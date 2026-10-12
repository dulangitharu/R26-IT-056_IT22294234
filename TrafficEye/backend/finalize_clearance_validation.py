"""
Prototype observed-clearance validation - final run for clip_01, clip_02,
clip_11 (clip_09 excluded - confirmed cross-class duplicate/class-flip).

Does NOT modify main.py, traffic_calculations.py,
generate_final_validation_snapshots.py, roi_config.json, YOLO weights,
confidence threshold, NMS, or any calibration/lookup CSV. Reuses:
- find_densest_frame / order_queue / bottom_center from
  generate_final_validation_snapshots.py (the current corrected ROI-aware
  runtime methodology - not reimplemented)
- build_queue_result from traffic_calculations.py (read-only call)
- side_of_line / distance_point_to_line / CONF_THRESHOLD from
  track_crossings.py (the SAME line-crossing convention already used by the
  project's calibration pipeline: a tracked vehicle's bottom-center sign
  flips across the reference line -> that frame is the crossing frame)
- the project's existing ByteTrack pipeline (model.track with persist=True,
  tracker='bytetrack.yaml') as an observation aid to follow one vehicle's
  identity forward from the snapshot frame - not a new tracker.

Writes only: outputs/final_validation/validation_clip_XX_snapshot.jpg
(regenerated for clip_01/02/11), outputs/final_validation/observed_evidence/*,
outputs/final_validation/final_clearance_validation.csv (updated per Step 7),
and a JSON run report. clip_09's old snapshot file is left untouched (not
part of this run).
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from generate_final_validation_snapshots import (
    BASE_DIR, UPLOAD_DIR, VALIDATION_DIR, model,
    find_densest_frame, order_queue, annotate_snapshot, bottom_center,
)
from traffic_calculations import build_queue_result
from track_crossings import side_of_line, distance_point_to_line, CONF_THRESHOLD

EVIDENCE_DIR = VALIDATION_DIR / "observed_evidence"
EVIDENCE_DIR.mkdir(exist_ok=True)

VALIDATION_CLIPS = ["clip_01.mp4", "clip_02.mp4", "clip_11.mp4"]

# How far past the snapshot frame to search for the crossing before giving up.
MAX_SEARCH_FRAMES_AHEAD = 900  # generous margin; loop also stops at video end

# Max pixel distance between the snapshot-frame Pn detection and a tracked
# box at the same frame for us to accept them as the same physical vehicle.
MATCH_DISTANCE_PX = 60


def identify_and_track_last_vehicle(video_path, snapshot_frame_idx, pn_bottom_center, line_b, fps):
    """
    Runs the project's existing ByteTrack pipeline from frame 0 (tracks need
    continuity to build a stable identity), finds which track corresponds to
    the last initial-queue vehicle (Pn) at the snapshot frame by nearest
    bottom-center match, then follows that same track_id forward and applies
    the SAME side_of_line sign-flip rule track_crossings.py already uses to
    find the first frame after the snapshot where it crosses line_b.

    Returns a dict describing the outcome - never invents a crossing time.
    """
    results_gen = model.track(
        source=str(video_path),
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONF_THRESHOLD,
        verbose=False,
    )

    target_track_id = None
    match_dist = None
    prev_side = None
    last_seen_frame = None
    last_seen_bbox = None
    crossing_frame = None
    crossing_side = None
    before_crossing_frame = None
    before_crossing_bbox = None
    frames_missing_streak = 0
    MAX_MISSING_STREAK = 30  # ~1s at 30fps / ~3s at 10fps - beyond this, treat identity as lost

    frame_idx = 0
    search_limit = snapshot_frame_idx + MAX_SEARCH_FRAMES_AHEAD

    for results in results_gen:
        if frame_idx > search_limit:
            break
        boxes = results.boxes
        this_frame_tracks = []
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().tolist()
            xyxys = boxes.xyxy.tolist()
            for track_id, (x1, y1, x2, y2) in zip(ids, xyxys):
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                bc = ((x1 + x2) / 2, y2)
                this_frame_tracks.append((track_id, (x1, y1, x2, y2), bc))

        if frame_idx == snapshot_frame_idx:
            # identify the track whose bottom-center best matches Pn's
            best = None
            best_d = None
            for track_id, bbox, bc in this_frame_tracks:
                d = ((bc[0] - pn_bottom_center[0]) ** 2 + (bc[1] - pn_bottom_center[1]) ** 2) ** 0.5
                if best_d is None or d < best_d:
                    best_d = d
                    best = (track_id, bbox, bc)
            if best is not None and best_d is not None and best_d <= MATCH_DISTANCE_PX:
                target_track_id, target_bbox, target_bc = best
                match_dist = round(best_d, 2)
                prev_side = side_of_line(target_bc, line_b[0], line_b[1])
                last_seen_frame = frame_idx
                last_seen_bbox = target_bbox
                before_crossing_frame = frame_idx
                before_crossing_bbox = target_bbox
            else:
                return {
                    "status": "AMBIGUOUS",
                    "reason": f"no ByteTrack detection within {MATCH_DISTANCE_PX}px of the snapshot-frame "
                              f"Pn bottom-center (closest was {best_d}px)" if best_d is not None
                              else "no ByteTrack detections at all at the snapshot frame",
                }

        elif target_track_id is not None and frame_idx > snapshot_frame_idx:
            found_this_frame = None
            for track_id, bbox, bc in this_frame_tracks:
                if track_id == target_track_id:
                    found_this_frame = (bbox, bc)
                    break

            if found_this_frame is not None:
                frames_missing_streak = 0
                bbox, bc = found_this_frame
                current_side = side_of_line(bc, line_b[0], line_b[1])
                if (
                    prev_side is not None and prev_side != 0
                    and current_side != 0 and current_side != prev_side
                ):
                    crossing_frame = frame_idx
                    crossing_side = current_side
                    crossing_bbox = bbox
                    break
                prev_side = current_side
                last_seen_frame = frame_idx
                last_seen_bbox = bbox
                before_crossing_frame = frame_idx
                before_crossing_bbox = bbox
            else:
                frames_missing_streak += 1
                if frames_missing_streak > MAX_MISSING_STREAK:
                    return {
                        "status": "AMBIGUOUS",
                        "reason": f"track_id {target_track_id} was lost for over {MAX_MISSING_STREAK} "
                                  f"consecutive frames after frame {last_seen_frame} without a confirmed "
                                  f"line_b crossing - cannot confidently follow the same physical vehicle "
                                  f"to a verified crossing",
                        "track_id": target_track_id,
                        "last_seen_frame": last_seen_frame,
                        "last_seen_time_s": round(last_seen_frame / fps, 3) if last_seen_frame is not None else None,
                    }

        frame_idx += 1

    if crossing_frame is None:
        return {
            "status": "AMBIGUOUS",
            "reason": f"track_id {target_track_id} did not cross line_b within "
                      f"{MAX_SEARCH_FRAMES_AHEAD} frames of the snapshot (search ended at frame {search_limit})",
            "track_id": target_track_id,
            "last_seen_frame": last_seen_frame,
        }

    return {
        "status": "OBSERVED",
        "track_id": target_track_id,
        "match_distance_px": match_dist,
        "before_crossing_frame": before_crossing_frame,
        "before_crossing_bbox": before_crossing_bbox,
        "crossing_frame": crossing_frame,
        "crossing_time_s": round(crossing_frame / fps, 3),
        "crossing_bbox": crossing_bbox,
    }


def save_evidence_image(video_path, frame_idx, bbox, line_b, roi_polygon, clip_name, label, out_path, fps, is_crossing):
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return False

    annotated = frame.copy()
    cv2.polylines(annotated, [np.array(roi_polygon, dtype=np.int32)], isClosed=True, color=(0, 140, 255), thickness=2)
    cv2.line(annotated, tuple(map(int, line_b[0])), tuple(map(int, line_b[1])), (0, 220, 120), 3)
    cv2.putText(annotated, "line_b", tuple(map(int, line_b[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 120), 2)

    x1, y1, x2, y2 = bbox
    color = (0, 0, 255) if is_crossing else (0, 200, 0)
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
    cv2.putText(annotated, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    frame_time_s = round(frame_idx / fps, 3)
    header = f"{clip_name}  frame={frame_idx}  t={frame_time_s:.3f}s  ({'CROSSING' if is_crossing else 'BEFORE CROSSING'})"
    cv2.putText(annotated, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imwrite(str(out_path), annotated)
    return True


def main():
    results = []

    for clip_name in VALIDATION_CLIPS:
        video_path = UPLOAD_DIR / clip_name
        stem = Path(clip_name).stem
        print(f"\n=== {clip_name} ===")

        scan = find_densest_frame(video_path)
        if scan["best_count"] == 0 or scan["best_frame"] is None:
            results.append({"clip_name": clip_name, "status": "NO_ROI_DETECTIONS"})
            continue

        ordered_vehicle_types, ranked = order_queue(scan["best_accepted"], scan["line_b"])
        queue_result = build_queue_result(ordered_vehicle_types)
        if not queue_result["available"]:
            results.append({"clip_name": clip_name, "status": "QUEUE_RESULT_UNAVAILABLE", "blocker": queue_result["blocker"]})
            continue

        snapshot_time_s = round(scan["best_frame_idx"] / scan["fps"], 3)

        # Regenerate the validation snapshot image (Step 1) using the exact
        # same annotate_snapshot function as before.
        annotated = annotate_snapshot(
            scan["best_frame"], scan["roi_polygon"], scan["line_b"], ranked, scan["best_rejected"],
            clip_name, scan["best_frame_idx"], snapshot_time_s
        )
        snapshot_name = f"validation_{stem}_snapshot.jpg"
        cv2.imwrite(str(VALIDATION_DIR / snapshot_name), annotated)

        pn = ranked[-1]  # last initial-queue vehicle
        pn_position = len(ranked)
        pn_bc = bottom_center(pn)

        print(f"Predicted: frame={scan['best_frame_idx']} t={snapshot_time_s}s "
              f"queue={ordered_vehicle_types} clearance={queue_result['queue_clearance_time_s']}s "
              f"green={queue_result['suggested_green_time_s']}s")
        print(f"Last initial-queue vehicle: P{pn_position} {pn['label']} bbox={pn['bbox']} bottom_center={pn_bc}")

        track_result = identify_and_track_last_vehicle(
            video_path, scan["best_frame_idx"], pn_bc, scan["line_b"], scan["fps"]
        )
        print("Tracking/crossing result:", json.dumps(track_result, default=str, indent=2))

        case = {
            "clip_name": clip_name,
            "snapshot_frame": scan["best_frame_idx"],
            "snapshot_time_s": snapshot_time_s,
            "fps": round(scan["fps"], 3),
            "vehicle_count": len(ordered_vehicle_types),
            "ordered_vehicle_types": ordered_vehicle_types,
            "position_groups": [d["position_group"] for d in queue_result["ordered_queue"]],
            "predicted_clearance_s": queue_result["queue_clearance_time_s"],
            "predicted_green_s": queue_result["suggested_green_time_s"],
            "last_queue_position": f"P{pn_position}",
            "last_vehicle_type": pn["label"],
            "last_vehicle_bbox_at_snapshot": pn["bbox"],
            "last_vehicle_bottom_center_at_snapshot": [round(v, 1) for v in pn_bc],
            "snapshot_file": snapshot_name,
            "tracking_result": track_result,
        }

        if track_result["status"] == "OBSERVED":
            crossing_time_s = track_result["crossing_time_s"]
            observed_clearance_s = round(crossing_time_s - snapshot_time_s, 3)
            signed_error_s = round(queue_result["queue_clearance_time_s"] - observed_clearance_s, 3)
            absolute_error_s = round(abs(signed_error_s), 3)

            case.update({
                "observed_crossing_frame": track_result["crossing_frame"],
                "observed_crossing_time_s": crossing_time_s,
                "observed_clearance_s": observed_clearance_s,
                "signed_error_s": signed_error_s,
                "absolute_error_s": absolute_error_s,
                "measurement_status": "OBSERVED",
            })

            label = f"P{pn_position} {pn['label']} (last initial-queue vehicle)"
            before_path = EVIDENCE_DIR / f"{stem}_before_crossing.jpg"
            crossing_path = EVIDENCE_DIR / f"{stem}_crossing.jpg"
            save_evidence_image(video_path, track_result["before_crossing_frame"], track_result["before_crossing_bbox"],
                                 scan["line_b"], scan["roi_polygon"], clip_name, label, before_path, scan["fps"], False)
            save_evidence_image(video_path, track_result["crossing_frame"], track_result["crossing_bbox"],
                                 scan["line_b"], scan["roi_polygon"], clip_name, label, crossing_path, scan["fps"], True)
            case["evidence_before"] = str(before_path.relative_to(BASE_DIR))
            case["evidence_crossing"] = str(crossing_path.relative_to(BASE_DIR))
        else:
            case.update({
                "observed_crossing_frame": "",
                "observed_crossing_time_s": "",
                "observed_clearance_s": "",
                "signed_error_s": "",
                "absolute_error_s": "",
                "measurement_status": "AMBIGUOUS_MANUAL_REVIEW_REQUIRED",
            })

        results.append(case)

    # Step 7: write the final validation CSV (exactly clip_01/02/11)
    csv_path = VALIDATION_DIR / "final_clearance_validation.csv"
    fieldnames = [
        "clip_name", "snapshot_frame", "snapshot_time_s", "vehicle_count",
        "ordered_vehicle_types", "predicted_clearance_s", "predicted_green_s",
        "observed_crossing_frame", "observed_crossing_time_s", "observed_clearance_s",
        "signed_error_s", "absolute_error_s", "measurement_status", "notes",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            if "predicted_clearance_s" not in r:
                continue
            notes = (
                f"last initial-queue vehicle P{r['last_queue_position'][1:]} {r['last_vehicle_type']}; "
                f"ByteTrack id={r['tracking_result'].get('track_id')}; "
            )
            if r["measurement_status"] == "OBSERVED":
                notes += "crossing visually verified via side_of_line sign-flip on the tracked identity."
            else:
                notes += f"NOT verified - {r['tracking_result'].get('reason', 'ambiguous')}."
            writer.writerow({
                "clip_name": r["clip_name"],
                "snapshot_frame": r["snapshot_frame"],
                "snapshot_time_s": r["snapshot_time_s"],
                "vehicle_count": r["vehicle_count"],
                "ordered_vehicle_types": "|".join(r["ordered_vehicle_types"]),
                "predicted_clearance_s": r["predicted_clearance_s"],
                "predicted_green_s": r["predicted_green_s"],
                "observed_crossing_frame": r["observed_crossing_frame"],
                "observed_crossing_time_s": r["observed_crossing_time_s"],
                "observed_clearance_s": r["observed_clearance_s"],
                "signed_error_s": r["signed_error_s"],
                "absolute_error_s": r["absolute_error_s"],
                "measurement_status": r["measurement_status"],
                "notes": notes,
            })

    report_path = VALIDATION_DIR / "finalize_clearance_validation_report.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nWrote {csv_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
