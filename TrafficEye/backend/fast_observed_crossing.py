"""
Full-stream observed-crossing check for ONE clip (clip_02 or clip_11 only -
clip_01's crossing is already confirmed and is not rerun here), using the
LOCKED snapshot values the user already verified.

Usage: python fast_observed_crossing.py <clip_key>
where <clip_key> is one of: clip_02, clip_11

Reuses identify_and_track_last_vehicle() from finalize_clearance_validation.py
unmodified - the exact method that already produced a correct, sensible
result for clip_01 (model.track(source=video_path, stream=True, persist=True,
tracker='bytetrack.yaml') - ByteTrack needs frame-0 continuity to build
stable track identities, so this starts at frame 0, but BREAKS OUT of the
generator the instant the target vehicle's line_b crossing is detected - it
does not process any frame after that.

Run as an independent process per clip (this file, invoked separately for
clip_02 and clip_11) specifically to avoid the cross-clip model-state
carryover already diagnosed as the cause of the earlier corrupted run.

Does not modify main.py, traffic_calculations.py,
generate_final_validation_snapshots.py, roi_config.json, YOLO weights,
confidence threshold, or any calibration/lookup CSV.
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from generate_final_validation_snapshots import (
    BASE_DIR, UPLOAD_DIR, VALIDATION_DIR, model,
    detect_vehicles_yolo, load_roi_geometry, bottom_center,
)
from track_crossings import distance_point_to_line
from finalize_clearance_validation import identify_and_track_last_vehicle

EVIDENCE_DIR = VALIDATION_DIR / "observed_evidence"
EVIDENCE_DIR.mkdir(exist_ok=True)

LOCKED = {
    "clip_02": {
        "clip_name": "clip_02.mp4", "frame": 4090, "time_s": 136.333,
        "queue": ["Car", "Motorcycle", "Motorcycle", "Motorcycle"],
        "predicted_clearance_s": 4.4132, "predicted_green_s": 7.53,
        "last_pos": 4, "last_label": "Motorcycle",
    },
    "clip_11": {
        "clip_name": "clip_11.mp4", "frame": 3040, "time_s": 101.322,
        "queue": ["Car", "Car", "Car", "Car", "Car"],
        "predicted_clearance_s": 8.7887, "predicted_green_s": 11.91,
        "last_pos": 5, "last_label": "Car",
    },
}


def bbox_xyxy(det):
    x, y, w, h = det["bbox"]
    return [x, y, x + w, y + h]


def main():
    clip_key = sys.argv[1]
    locked = LOCKED[clip_key]
    clip_name = locked["clip_name"]
    video_path = UPLOAD_DIR / clip_name

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    frame_idx = 0
    snapshot_frame_img = None
    while frame_idx <= locked["frame"]:
        ret, frame = cap.read()
        if not ret:
            cap.release()
            result = {"clip_key": clip_key, "status": "AMBIGUOUS",
                      "reason": f"video ended before reaching locked snapshot frame {locked['frame']}"}
            print(json.dumps(result, indent=2))
            (VALIDATION_DIR / f"{clip_key}_fast_result.json").write_text(json.dumps(result, indent=2))
            return
        if frame_idx == locked["frame"]:
            snapshot_frame_img = frame.copy()
        frame_idx += 1
    cap.release()

    roi_polygon, roi_polygon_cv, line_b = load_roi_geometry(video_w, video_h)

    _, dets = detect_vehicles_yolo(snapshot_frame_img)
    accepted = [d for d in dets if cv2.pointPolygonTest(roi_polygon_cv, bottom_center(d), False) >= 0]
    ranked = sorted(accepted, key=lambda d: distance_point_to_line(bottom_center(d), line_b[0], line_b[1]))
    ranked_labels = [d["label"] for d in ranked]
    sanity_match = ranked_labels == locked["queue"]

    print(f"[{clip_key}] sanity_match(ranked labels == locked queue): {sanity_match}")
    print(f"[{clip_key}] ranked_labels_now={ranked_labels}  locked_queue={locked['queue']}")

    if len(ranked) < locked["last_pos"]:
        result = {"clip_key": clip_key, "status": "AMBIGUOUS",
                  "reason": f"single-frame re-detection at the locked snapshot found only {len(ranked)} "
                            f"ROI_IN vehicles, fewer than the locked P{locked['last_pos']}",
                  "ranked_labels_now": ranked_labels, "locked_queue": locked["queue"]}
        print(json.dumps(result, indent=2))
        (VALIDATION_DIR / f"{clip_key}_fast_result.json").write_text(json.dumps(result, indent=2))
        return

    pn = ranked[locked["last_pos"] - 1]
    pn_bc = bottom_center(pn)
    print(f"[{clip_key}] P{locked['last_pos']} re-detected as {pn['label']} (locked label: {locked['last_label']}) "
          f"bbox={bbox_xyxy(pn)} bottom_center={pn_bc}")

    track_result = identify_and_track_last_vehicle(video_path, locked["frame"], pn_bc, line_b, fps)
    print(f"[{clip_key}] tracking result:", json.dumps(track_result, default=str, indent=2))

    if track_result["status"] != "OBSERVED":
        result = {"clip_key": clip_key, "status": "AMBIGUOUS",
                  "reason": track_result.get("reason", "unknown"), "sanity_match": sanity_match,
                  "track_result": track_result}
        print(json.dumps(result, indent=2, default=str))
        (VALIDATION_DIR / f"{clip_key}_fast_result.json").write_text(json.dumps(result, indent=2, default=str))
        return

    crossing_frame = track_result["crossing_frame"]
    crossing_time_s = track_result["crossing_time_s"]
    crossing_bbox = track_result["crossing_bbox"]
    observed_clearance_s = round(crossing_time_s - locked["time_s"], 3)
    signed_error_s = round(locked["predicted_clearance_s"] - observed_clearance_s, 3)
    absolute_error_s = round(abs(signed_error_s), 3)

    # evidence image at the crossing frame only
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, crossing_frame)
    ret, ev_frame = cap.read()
    cap.release()
    evidence_path = None
    if ret:
        annotated = ev_frame.copy()
        cv2.polylines(annotated, [np.array(roi_polygon, dtype=np.int32)], isClosed=True, color=(0, 140, 255), thickness=2)
        cv2.line(annotated, tuple(map(int, line_b[0])), tuple(map(int, line_b[1])), (0, 220, 120), 3)
        cv2.putText(annotated, "line_b", tuple(map(int, line_b[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 120), 2)
        x1, y1, x2, y2 = crossing_bbox
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
        label = f"P{locked['last_pos']} {pn['label']} (last initial-queue vehicle) - CROSSING"
        cv2.putText(annotated, label, (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        header = f"{clip_name}  frame={crossing_frame}  t={crossing_time_s:.3f}s"
        cv2.putText(annotated, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        evidence_path = EVIDENCE_DIR / f"{clip_key}_crossing.jpg"
        cv2.imwrite(str(evidence_path), annotated)

    result = {
        "clip_key": clip_key,
        "status": "OBSERVED",
        "sanity_match": sanity_match,
        "clip_name": clip_name,
        "snapshot_frame": locked["frame"],
        "snapshot_time_s": locked["time_s"],
        "vehicle_count": len(locked["queue"]),
        "ordered_vehicle_types": locked["queue"],
        "predicted_clearance_s": locked["predicted_clearance_s"],
        "predicted_green_s": locked["predicted_green_s"],
        "last_queue_position": f"P{locked['last_pos']}",
        "last_vehicle_type": pn["label"],
        "track_id": track_result["track_id"],
        "match_distance_px": track_result["match_distance_px"],
        "observed_crossing_frame": crossing_frame,
        "observed_crossing_time_s": crossing_time_s,
        "observed_clearance_s": observed_clearance_s,
        "signed_error_s": signed_error_s,
        "absolute_error_s": absolute_error_s,
        "measurement_status": "OBSERVED",
        "evidence_image": str(evidence_path.relative_to(BASE_DIR)) if evidence_path else None,
    }
    print(json.dumps(result, indent=2))
    (VALIDATION_DIR / f"{clip_key}_fast_result.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
