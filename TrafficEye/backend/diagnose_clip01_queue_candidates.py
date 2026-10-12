"""
READ-ONLY diagnostic: audit every YOLO detection in clip_01.mp4 frame 280
(the exact frame /api/analysis's densest-frame scan selected for the
prototype observed-clearance validation snapshot).

Goal: determine (a) whether ROI-polygon membership is currently checked
before a detection is allowed into the ordered queue (it is NOT, in
main.py's create_analysis - only distance_point_to_line is used, unlike
track_crossings.py's offline pipeline which DOES gate every detection with
cv2.pointPolygonTest before doing anything else with it), and (b) whether
the P5 candidate is a real, persistent vehicle or a short-lived false
positive.

Does not modify main.py, traffic_calculations.py, roi_config.json, YOLO
weights, confidence thresholds, or any calibration CSV. Reuses the exact
model, conf threshold, bbox/bottom-center formula, ROI test (cv2.pointPolygonTest,
copied verbatim from track_crossings.py's existing usage - not a new
implementation), and distance_point_to_line already used elsewhere in this
project. Writes only new diagnostic files under outputs/final_validation/.
"""
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
    CONF_THRESHOLD,
    compute_scale,
    scale_points,
    distance_point_to_line,
)

UPLOAD_DIR = BASE_DIR / "uploads"
VALIDATION_DIR = BASE_DIR / "outputs" / "final_validation"
VALIDATION_DIR.mkdir(exist_ok=True)

VIDEO_PATH = UPLOAD_DIR / "clip_01.mp4"
TARGET_FRAME_IDX = 280
TEMPORAL_WINDOW = (250, 310)

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


def load_roi_geometry(video_w, video_h):
    with open(ROI_CONFIG_PATH) as f:
        config = json.load(f)
    scale_x, scale_y = compute_scale(video_w, video_h)
    roi_polygon = scale_points(config["roi_points"], scale_x, scale_y)
    line_b = scale_points(config["line_b"], scale_x, scale_y)
    roi_polygon_cv = np.array(roi_polygon, dtype=np.float32)
    return roi_polygon, roi_polygon_cv, line_b


def detect_all(frame):
    """Same model/conf as main.py's detect_vehicles_yolo, but keeps every
    field (not just count/label) for the audit."""
    results = model(frame, verbose=False, conf=0.4)[0]
    dets = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = VEHICLE_CLASSES.get(cls_id, f"class_{cls_id}")
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = round(float(box.conf[0]), 4)
        dets.append({"cls_id": cls_id, "label": label, "conf": conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return dets


def step2_audit_frame():
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {VIDEO_PATH}")
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_idx = 0
    target_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx == TARGET_FRAME_IDX:
            target_frame = frame.copy()
            break
        frame_idx += 1
    cap.release()

    if target_frame is None:
        raise RuntimeError(f"frame {TARGET_FRAME_IDX} not reachable in {VIDEO_PATH}")

    roi_polygon, roi_polygon_cv, line_b = load_roi_geometry(video_w, video_h)
    dets = detect_all(target_frame)

    audit_rows = []
    for i, det in enumerate(dets):
        bottom_center = ((det["x1"] + det["x2"]) / 2, det["y2"])
        inside_roi = cv2.pointPolygonTest(roi_polygon_cv, bottom_center, False) >= 0
        dist_line_b = round(distance_point_to_line(bottom_center, line_b[0], line_b[1]), 2)
        audit_rows.append({
            "index": i,
            "class": det["label"],
            "conf": det["conf"],
            "bbox_x1y1x2y2": [det["x1"], det["y1"], det["x2"], det["y2"]],
            "bottom_center": [round(bottom_center[0], 1), round(bottom_center[1], 1)],
            "roi_status": "ROI_IN" if inside_roi else "ROI_OUT",
            "distance_to_line_b_px": dist_line_b,
        })

    # Current runtime behavior: ALL detections ranked by distance to line_b,
    # with NO ROI-membership filter applied first (mirrors main.py exactly).
    ranked = sorted(audit_rows, key=lambda r: r["distance_to_line_b_px"])
    for rank, r in enumerate(ranked, start=1):
        r["current_assigned_queue_position"] = f"P{rank}"
    # re-sort back to detection index order for the printed table
    audit_rows_by_index = sorted(ranked, key=lambda r: r["index"])

    return video_w, video_h, target_frame, roi_polygon, roi_polygon_cv, line_b, audit_rows_by_index


def step3_overlay(frame, roi_polygon, line_b, audit_rows):
    annotated = frame.copy()
    cv2.polylines(annotated, [np.array(roi_polygon, dtype=np.int32)], isClosed=True, color=(0, 140, 255), thickness=2)
    cv2.putText(annotated, "ROI", tuple(roi_polygon[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 140, 255), 2)
    cv2.line(annotated, tuple(map(int, line_b[0])), tuple(map(int, line_b[1])), (0, 220, 120), 2)
    cv2.putText(annotated, "line_b", tuple(map(int, line_b[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 120), 2)

    for r in audit_rows:
        x1, y1, x2, y2 = r["bbox_x1y1x2y2"]
        color = (0, 255, 0) if r["roi_status"] == "ROI_IN" else (0, 0, 255)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        bc = tuple(int(v) for v in r["bottom_center"])
        cv2.circle(annotated, bc, 5, (255, 0, 255), -1)
        label = (f"#{r['index']} {r['current_assigned_queue_position']} {r['class']} "
                 f"{r['conf']:.2f} {r['roi_status']}")
        cv2.putText(annotated, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    out_path = VALIDATION_DIR / "clip01_queue_candidate_audit.jpg"
    cv2.imwrite(str(out_path), annotated)
    return out_path


def step4_temporal_check(suspect_bottom_center, video_w, video_h):
    """Reuses the project's EXISTING ByteTrack pipeline (model.track with
    persist=True, tracker='bytetrack.yaml', same conf as track_crossings.py)
    - not a new tracking implementation. Runs from frame 0 so track IDs are
    established continuously (ByteTrack needs preceding frames to assign a
    stable identity), but only records detail for the requested window."""
    lo, hi = TEMPORAL_WINDOW
    results_gen = model.track(
        source=str(VIDEO_PATH),
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONF_THRESHOLD,
        verbose=False,
    )

    roi_polygon, roi_polygon_cv, line_b = load_roi_geometry(video_w, video_h)

    window_frames = {}
    frame_idx = 0
    best_track_id_at_target = None
    best_dist_at_target = None

    for results in results_gen:
        if frame_idx > hi:
            break
        boxes = results.boxes
        frame_dets = []
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().tolist()
            clss = boxes.cls.int().tolist()
            confs = boxes.conf.tolist()
            xyxys = boxes.xyxy.tolist()
            for track_id, cls_id, conf, (x1, y1, x2, y2) in zip(ids, clss, confs, xyxys):
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                bottom_center = ((x1 + x2) / 2, y2)
                inside_roi = cv2.pointPolygonTest(roi_polygon_cv, bottom_center, False) >= 0
                label = VEHICLE_CLASSES.get(cls_id, f"class_{cls_id}")
                frame_dets.append({
                    "track_id": track_id, "class": label, "conf": round(float(conf), 4),
                    "bottom_center": [round(bottom_center[0], 1), round(bottom_center[1], 1)],
                    "roi_status": "ROI_IN" if inside_roi else "ROI_OUT",
                })
                if frame_idx == TARGET_FRAME_IDX:
                    d = ((bottom_center[0] - suspect_bottom_center[0]) ** 2
                         + (bottom_center[1] - suspect_bottom_center[1]) ** 2) ** 0.5
                    if best_dist_at_target is None or d < best_dist_at_target:
                        best_dist_at_target = d
                        best_track_id_at_target = track_id

        if lo <= frame_idx <= hi:
            window_frames[frame_idx] = frame_dets
        frame_idx += 1

    matched_track_history = []
    if best_track_id_at_target is not None:
        for f_idx in sorted(window_frames):
            for d in window_frames[f_idx]:
                if d["track_id"] == best_track_id_at_target:
                    matched_track_history.append({"frame": f_idx, **d})

    return {
        "suspect_track_id": best_track_id_at_target,
        "match_distance_px_at_target_frame": round(best_dist_at_target, 2) if best_dist_at_target is not None else None,
        "track_history_in_window": matched_track_history,
        "window": [lo, hi],
    }


def main():
    video_w, video_h, frame, roi_polygon, roi_polygon_cv, line_b, audit_rows = step2_audit_frame()

    print("=== STEP 2: Frame 280 detection audit ===")
    for r in audit_rows:
        print(r)

    img_path = step3_overlay(frame, roi_polygon, line_b, audit_rows)
    print(f"\n=== STEP 3: overlay written to {img_path} ===")

    # Identify the P5 candidate (current_assigned_queue_position == "P5")
    p5 = next((r for r in audit_rows if r["current_assigned_queue_position"] == "P5"), None)
    p1 = next((r for r in audit_rows if r["current_assigned_queue_position"] == "P1"), None)

    temporal_result = None
    if p5 is not None:
        print(f"\n=== STEP 4: temporal check for P5 candidate {p5} ===")
        temporal_result = step4_temporal_check(tuple(p5["bottom_center"]), video_w, video_h)
        print(json.dumps(temporal_result, indent=2))

    report = {
        "frame_audit": audit_rows,
        "p1_candidate": p1,
        "p5_candidate": p5,
        "p5_temporal_check": temporal_result,
        "overlay_image": str(img_path.name),
    }
    report_path = VALIDATION_DIR / "clip01_queue_candidate_audit_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
