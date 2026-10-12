"""
READ-ONLY forensic reproduction of the CURRENT /api/analysis pipeline for
clip_04.mp4 (the record behind the user's screenshot), reusing main.py's
ACTUAL functions (select_camera_profile, detect_vehicles_yolo, VEHICLE_CLASSES,
ANNOTATED_RECORDING_TARGET_FPS) unmodified - not reimplemented - so results
are proof about the real running system, not a guess.

Reproduces, independently and side by side:
  1. The normal densest-ROI-frame scan (vehicle_count/vehicle_breakdown).
  2. The separate emergency-candidate scan (every sampled frame with an
     Ambulance in ROI).
  3. generate_annotated_recording()'s OWN independent frame-sampling grid
     (its own VideoCapture, its own sample_interval), to check whether the
     emergency winning frame is even included in what the user can see.

Writes only new diagnostic files under outputs/final_validation/. Does not
modify main.py, any model, any config, any database record.
"""
import json
from pathlib import Path

import cv2
import numpy as np

import main as m  # reuse the EXACT live code - not reimplemented

BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "uploads" / "clip_04.mp4"
OUT_DIR = BASE_DIR / "outputs" / "final_validation"
OUT_DIR.mkdir(exist_ok=True)


def bottom_center(det):
    return (det["bbox"][0] + det["bbox"][2] / 2, det["bbox"][1] + det["bbox"][3])


def main():
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    selected_profile, roi_config, scale_fn, profile_scores = m.select_camera_profile(cap)
    print("Profile scores:", profile_scores)
    print("Selected profile:", selected_profile["id"] if selected_profile else None)

    roi_usable = False
    line_b = None
    roi_polygon_cv = None
    if selected_profile is not None:
        scale_x, scale_y = scale_fn(video_w, video_h)
        line_b = m.roi_scale_points(roi_config["line_b"], scale_x, scale_y)
        roi_polygon = m.roi_scale_points(roi_config["roi_points"], scale_x, scale_y)
        roi_polygon_cv = np.array(roi_polygon, dtype=np.float32)
        roi_usable = True

    best_count = 0
    best_frame_idx = None
    best_breakdown = {}
    best_detections = []
    best_total_detected = 0

    emergency_best_count = 0
    emergency_best_frame_idx = None
    emergency_best_detections = []

    all_ambulance_candidates = []  # every sampled frame with an Ambulance in ROI

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % 10 == 0:
            raw_count, raw_detections = m.detect_vehicles_yolo(frame)
            detections = raw_detections
            count = raw_count
            if roi_usable:
                detections = [
                    det for det in raw_detections
                    if cv2.pointPolygonTest(roi_polygon_cv, bottom_center(det), False) >= 0
                ]
                count = len(detections)
            if count > best_count:
                breakdown = {}
                for det in detections:
                    breakdown[det["label"]] = breakdown.get(det["label"], 0) + 1
                best_count = count
                best_breakdown = breakdown
                best_detections = detections
                best_frame_idx = frame_idx
                best_total_detected = raw_count

            amb_dets = [d for d in detections if d["label"] == "Ambulance"]
            if roi_usable and amb_dets:
                for d in amb_dets:
                    all_ambulance_candidates.append({
                        "frame_idx": frame_idx,
                        "time_s": round(frame_idx / source_fps, 3),
                        "conf": d["conf"],
                        "bbox": d["bbox"],
                        "bottom_center": [round(c, 1) for c in bottom_center(d)],
                        "roi_vehicle_count_this_frame": count,
                        "all_labels_this_frame": [x["label"] for x in detections],
                    })
                if roi_usable and count > emergency_best_count:
                    emergency_best_count = count
                    emergency_best_frame_idx = frame_idx
                    emergency_best_detections = detections
        frame_idx += 1
    cap.release()

    print(f"\nNormal densest ROI frame: idx={best_frame_idx} t={best_frame_idx/source_fps:.3f}s "
          f"count={best_count} breakdown={best_breakdown} total_detected(raw)={best_total_detected}")

    print(f"\nAll Ambulance-in-ROI candidate frames ({len(all_ambulance_candidates)}):")
    for c in all_ambulance_candidates:
        print(f"  frame={c['frame_idx']} t={c['time_s']}s conf={c['conf']} bbox={c['bbox']} "
              f"roi_count={c['roi_vehicle_count_this_frame']} labels={c['all_labels_this_frame']}")

    emergency_result = None
    if emergency_best_detections:
        ranked = sorted(emergency_best_detections,
                         key=lambda d: m.distance_point_to_line(bottom_center(d), line_b[0], line_b[1]))
        ordered = [d["label"] for d in ranked]
        qr = m.build_queue_result(ordered)
        emergency_result = {
            "emergency_best_frame_idx": emergency_best_frame_idx,
            "emergency_best_time_s": round(emergency_best_frame_idx / source_fps, 3),
            "ordered_vehicle_types": ordered,
            "queue_result_available": qr["available"],
        }
        if qr["available"]:
            emergency_result["emergency_detected_from_this_frame"] = qr["emergency_detected"]
            if qr["emergency_detected"]:
                emergency_result.update({
                    "emergency_vehicle_type": qr["emergency_vehicle_type"],
                    "emergency_queue_position": qr["emergency_queue_position"],
                    "emergency_position_group": qr["emergency_position_group"],
                    "emergency_clearance_time_s": qr["emergency_clearance_time_s"],
                    "emergency_priority_green_s": qr["emergency_priority_green_s"],
                })
        else:
            emergency_result["blocker"] = qr["blocker"]
        print("\nWinning emergency frame result:")
        print(json.dumps(emergency_result, indent=2))

    # ── Reproduce generate_annotated_recording()'s OWN independent sampling
    # grid to check whether the emergency winning frame (or ANY ambulance
    # candidate frame) is even included in what the user sees.
    sample_interval = max(1, round(source_fps / m.ANNOTATED_RECORDING_TARGET_FPS))
    annotated_grid_frames = set(range(0, total_frame_count, sample_interval))
    print(f"\nAnnotated-video sample_interval={sample_interval} (source_fps={source_fps}, "
          f"target_fps={m.ANNOTATED_RECORDING_TARGET_FPS})")

    normal_frame_in_annotated_grid = best_frame_idx in annotated_grid_frames if best_frame_idx is not None else None
    print(f"Normal densest frame {best_frame_idx} in annotated grid: {normal_frame_in_annotated_grid}")
    if best_frame_idx is not None:
        nearest_grid_frame = min(annotated_grid_frames, key=lambda f: abs(f - best_frame_idx))
        print(f"  nearest annotated-grid frame to normal densest frame: {nearest_grid_frame} "
              f"(t={nearest_grid_frame/source_fps:.3f}s, encoded_playback_t={ (nearest_grid_frame//sample_interval)/m.ANNOTATED_RECORDING_TARGET_FPS:.3f}s)")

    for c in all_ambulance_candidates:
        in_grid = c["frame_idx"] in annotated_grid_frames
        encoded_playback_t = (c["frame_idx"] // sample_interval) / m.ANNOTATED_RECORDING_TARGET_FPS if in_grid else None
        c["in_annotated_video_grid"] = in_grid
        c["annotated_video_playback_time_s"] = round(encoded_playback_t, 3) if encoded_playback_t is not None else None
        print(f"  Ambulance candidate frame={c['frame_idx']} in_annotated_grid={in_grid} "
              f"annotated_playback_t={c['annotated_video_playback_time_s']}")

    report = {
        "video": "clip_04.mp4",
        "source_fps": source_fps,
        "profile_scores": profile_scores,
        "selected_profile": selected_profile["id"] if selected_profile else None,
        "normal_densest_frame": {
            "frame_idx": best_frame_idx,
            "time_s": round(best_frame_idx / source_fps, 3) if best_frame_idx is not None else None,
            "vehicle_count": best_count,
            "vehicle_breakdown": best_breakdown,
            "total_detected_raw": best_total_detected,
            "in_annotated_video_grid": normal_frame_in_annotated_grid,
        },
        "all_ambulance_candidates": all_ambulance_candidates,
        "emergency_winning_frame": emergency_result,
        "annotated_video_sample_interval": sample_interval,
    }
    report_path = OUT_DIR / "clip04_emergency_forensic_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
