"""
Prototype observed-clearance validation - Step 1 & Step 3 (snapshot generation).

Read-only w.r.t. all protected/core files. Reproduces the EXACT ROI-aware
densest-frame sampling and queue-ordering logic used by POST /api/analysis
in main.py (same "every 10th frame" scan, same detect_vehicles_yolo, same
bottom-center-inside-roi_points membership test via cv2.pointPolygonTest,
same distance-to-line_b ordering) - not a different/independent method.
Writes only new files under outputs/final_validation/.

This is a prototype observed-clearance validation (clips overlap the existing
calibration set), NOT an independent held-out validation.
"""
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from ultralytics import YOLO
from track_crossings import (
    FIRST_FRAME_PATH as ROI_FIRST_FRAME_PATH,
    ROI_CONFIG_PATH,
    compute_scale as roi_compute_scale,
    distance_point_to_line,
    scale_points as roi_scale_points,
)
from traffic_calculations import build_queue_result

UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
VALIDATION_DIR = OUTPUT_DIR / "final_validation"
VALIDATION_DIR.mkdir(exist_ok=True)

VALIDATION_CLIPS = ["clip_01.mp4", "clip_09.mp4", "clip_11.mp4"]

VEHICLE_CLASSES = {
    0: "Ambulance",
    1: "Bicycle",
    2: "Bus",
    3: "Car",
    4: "Motorcycle",
    5: "Truck",
    6: "Tuktuk",
}

model = YOLO(str(BASE_DIR / "runs2" / "detect" / "traffic_monitor" / "weights" / "best.pt"))


def detect_vehicles_yolo(frame):
    """Identical to main.py's detect_vehicles_yolo."""
    results = model(frame, verbose=False, conf=0.4)[0]
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id in VEHICLE_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "label": VEHICLE_CLASSES[cls_id],
                "conf": round(float(box.conf[0]), 2),
            })
    return len(detections), detections


def load_roi_geometry(video_w, video_h):
    """Identical to main.py's ROI-usability block: bottom-center-inside-
    roi_points membership (cv2.pointPolygonTest) is the ONLY geometry
    convention used - reused here, not reinvented."""
    with open(ROI_CONFIG_PATH) as f:
        roi_config = json.load(f)
    scale_x, scale_y = roi_compute_scale(video_w, video_h)
    line_b = roi_scale_points(roi_config["line_b"], scale_x, scale_y)
    roi_polygon = roi_scale_points(roi_config["roi_points"], scale_x, scale_y)
    roi_polygon_cv = np.array(roi_polygon, dtype=np.float32)
    return roi_polygon, roi_polygon_cv, line_b


def bottom_center(det):
    return (det["bbox"][0] + det["bbox"][2] / 2, det["bbox"][1] + det["bbox"][3])


def find_densest_frame(video_path: Path):
    """Identical scan to main.py's create_analysis: every 10th frame, keep the
    frame with the most ROI_IN detections (not most detections anywhere).
    Also keeps that winning frame's ROI_OUT detections for diagnostic
    overlay visibility only - they never receive a queue position."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")

    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    roi_polygon, roi_polygon_cv, line_b = load_roi_geometry(video_w, video_h)

    best_count = 0
    best_accepted = []
    best_rejected = []
    best_frame = None
    best_frame_idx = None
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % 10 == 0:
                _, detections = detect_vehicles_yolo(frame)
                accepted, rejected = [], []
                for det in detections:
                    if cv2.pointPolygonTest(roi_polygon_cv, bottom_center(det), False) >= 0:
                        accepted.append(det)
                    else:
                        rejected.append(det)
                if len(accepted) > best_count:
                    best_count = len(accepted)
                    best_accepted = accepted
                    best_rejected = rejected
                    best_frame = frame.copy()
                    best_frame_idx = frame_idx
            frame_idx += 1
    finally:
        cap.release()

    return {
        "video_w": video_w,
        "video_h": video_h,
        "fps": fps,
        "best_count": best_count,
        "best_accepted": best_accepted,
        "best_rejected": best_rejected,
        "best_frame": best_frame,
        "best_frame_idx": best_frame_idx,
        "roi_polygon": roi_polygon,
        "line_b": line_b,
    }


def order_queue(best_accepted, line_b):
    """Identical to main.py's queue-ordering: rank the already ROI-filtered
    detections by distance to line_b. No ROI_OUT detection is ever passed
    in here."""
    ranked = sorted(
        best_accepted,
        key=lambda det: distance_point_to_line(bottom_center(det), line_b[0], line_b[1]),
    )
    ordered_vehicle_types = [det["label"] for det in ranked]
    return ordered_vehicle_types, ranked


def annotate_snapshot(frame, roi_polygon, line_b, ranked_accepted, rejected, clip_name, frame_idx, snapshot_time_s):
    annotated = frame.copy()
    cv2.polylines(annotated, [np.array(roi_polygon, dtype=np.int32)], isClosed=True, color=(0, 140, 255), thickness=2)
    cv2.putText(annotated, "ROI", tuple(roi_polygon[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
    cv2.line(annotated, tuple(map(int, line_b[0])), tuple(map(int, line_b[1])), (0, 220, 120), 2)
    cv2.putText(annotated, "line_b", tuple(map(int, line_b[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 120), 2)

    for position, det in enumerate(ranked_accepted, start=1):
        x, y, w, h = det["bbox"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)
        label = f"P{position}:{det['label']}"
        cv2.putText(annotated, label, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 0), 2)

    for det in rejected:
        x, y, w, h = det["bbox"]
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 0, 255), 2)
        label = f"REJECTED ROI_OUT:{det['label']}"
        cv2.putText(annotated, label, (x, max(0, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    header = f"{clip_name}  frame={frame_idx}  t={snapshot_time_s:.3f}s  queue={len(ranked_accepted)}  rejected={len(rejected)}"
    cv2.putText(annotated, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return annotated


def main():
    rows = []
    report_cases = []

    for clip_name in VALIDATION_CLIPS:
        video_path = UPLOAD_DIR / clip_name
        if not video_path.exists():
            report_cases.append({"clip_name": clip_name, "status": "MISSING_FILE"})
            continue

        scan = find_densest_frame(video_path)
        if scan["best_count"] == 0 or scan["best_frame"] is None:
            report_cases.append({
                "clip_name": clip_name,
                "status": "NO_ROI_DETECTIONS",
                "note": "no vehicle detections inside the calibrated ROI in any sampled frame",
            })
            continue

        ordered_vehicle_types, ranked = order_queue(scan["best_accepted"], scan["line_b"])
        queue_result = build_queue_result(ordered_vehicle_types)
        if not queue_result["available"]:
            report_cases.append({
                "clip_name": clip_name, "status": "QUEUE_RESULT_UNAVAILABLE",
                "blocker": queue_result["blocker"],
            })
            continue

        # vehicle_count must equal len(ordered_queue) - both come from the
        # same ROI-filtered best_accepted list, so this holds by construction.
        assert len(ordered_vehicle_types) == len(queue_result["ordered_queue"])

        snapshot_time_s = round(scan["best_frame_idx"] / scan["fps"], 3)
        annotated = annotate_snapshot(
            scan["best_frame"], scan["roi_polygon"], scan["line_b"], ranked, scan["best_rejected"],
            clip_name, scan["best_frame_idx"], snapshot_time_s
        )
        snapshot_name = f"validation_{Path(clip_name).stem}_snapshot.jpg"
        cv2.imwrite(str(VALIDATION_DIR / snapshot_name), annotated)

        position_groups = [d["position_group"] for d in queue_result["ordered_queue"]]

        case = {
            "clip_name": clip_name,
            "status": "OK",
            "snapshot_frame": scan["best_frame_idx"],
            "snapshot_time_s": snapshot_time_s,
            "fps": round(scan["fps"], 3),
            "vehicle_count": len(ordered_vehicle_types),
            "rejected_roi_out_count": len(scan["best_rejected"]),
            "ordered_vehicle_types": ordered_vehicle_types,
            "position_groups": position_groups,
            "predicted_clearance_s": queue_result["queue_clearance_time_s"],
            "predicted_green_s": queue_result["suggested_green_time_s"],
            "snapshot_file": snapshot_name,
        }
        report_cases.append(case)

        rows.append({
            "clip_name": clip_name,
            "snapshot_frame": scan["best_frame_idx"],
            "snapshot_time_s": snapshot_time_s,
            "vehicle_count": len(ordered_vehicle_types),
            "ordered_vehicle_types": "|".join(ordered_vehicle_types),
            "predicted_clearance_s": queue_result["queue_clearance_time_s"],
            "predicted_green_s": queue_result["suggested_green_time_s"],
            "observed_clearance_s": "",
            "absolute_error_s": "",
            "measurement_status": "PENDING_MANUAL_OBSERVATION",
            "notes": f"line_b location per calibration/roi_config.json (scaled to this clip's resolution); "
                     f"ROI-filtered queue ({len(scan['best_rejected'])} off-ROI detection(s) rejected this frame); "
                     f"watch the vehicle in position P{len(ordered_vehicle_types)} "
                     f"({ordered_vehicle_types[-1] if ordered_vehicle_types else ''}) cross line_b.",
        })

    csv_path = VALIDATION_DIR / "final_clearance_validation.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "clip_name", "snapshot_frame", "snapshot_time_s", "vehicle_count",
            "ordered_vehicle_types", "predicted_clearance_s", "predicted_green_s",
            "observed_clearance_s", "absolute_error_s", "measurement_status", "notes",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    report_path = VALIDATION_DIR / "step1_snapshot_report.json"
    with open(report_path, "w") as f:
        json.dump(report_cases, f, indent=2)

    print(json.dumps(report_cases, indent=2))
    print(f"\nWrote {csv_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
