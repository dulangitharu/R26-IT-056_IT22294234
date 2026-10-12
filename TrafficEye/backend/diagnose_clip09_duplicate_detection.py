"""
READ-ONLY diagnostic: is clip_09.mp4 frame 1230's P1 (Truck) / P2 (Bus) pair
two distinct physical vehicles, or one physical vehicle detected twice under
two class labels?

Does not modify main.py, traffic_calculations.py,
generate_final_validation_snapshots.py, roi_config.json, YOLO weights,
confidence threshold, NMS settings, or any calibration/validation CSV.
Reuses the exact model, conf threshold, ROI-membership rule
(cv2.pointPolygonTest, as already used in track_crossings.py and main.py's
corrected runtime), and distance_point_to_line already used elsewhere in
this project - no new tracker/geometry convention. Writes only new
diagnostic files under outputs/final_validation/.
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
    ROI_CONFIG_PATH,
    CONF_THRESHOLD,
    compute_scale,
    scale_points,
    distance_point_to_line,
)

UPLOAD_DIR = BASE_DIR / "uploads"
VALIDATION_DIR = BASE_DIR / "outputs" / "final_validation"
VALIDATION_DIR.mkdir(exist_ok=True)

VIDEO_PATH = UPLOAD_DIR / "clip_09.mp4"
TARGET_FRAME_IDX = 1230
TEMPORAL_WINDOW = (1170, 1290)

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
print(f"model default predict args (NMS-relevant): "
      f"iou={model.overrides.get('iou')}, agnostic_nms={model.overrides.get('agnostic_nms')}, "
      f"conf(default)={model.overrides.get('conf')}, max_det={model.overrides.get('max_det')}")


def load_roi_geometry(video_w, video_h):
    with open(ROI_CONFIG_PATH) as f:
        config = json.load(f)
    scale_x, scale_y = compute_scale(video_w, video_h)
    roi_polygon = scale_points(config["roi_points"], scale_x, scale_y)
    line_b = scale_points(config["line_b"], scale_x, scale_y)
    roi_polygon_cv = np.array(roi_polygon, dtype=np.float32)
    return roi_polygon, roi_polygon_cv, line_b


def bottom_center_of(x1, y1, x2, y2):
    return ((x1 + x2) / 2, y2)


def iou_and_overlap(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0
    pct_a_covered = inter / area_a if area_a > 0 else 0.0
    pct_b_covered = inter / area_b if area_b > 0 else 0.0
    center_a = ((ax1 + ax2) / 2, (ay1 + ay2) / 2)
    center_b = ((bx1 + bx2) / 2, (by1 + by2) / 2)
    center_dist = ((center_a[0] - center_b[0]) ** 2 + (center_a[1] - center_b[1]) ** 2) ** 0.5
    bc_a = bottom_center_of(*a)
    bc_b = bottom_center_of(*b)
    bc_dist = ((bc_a[0] - bc_b[0]) ** 2 + (bc_a[1] - bc_b[1]) ** 2) ** 0.5
    return {
        "iou": round(iou, 4),
        "intersection_area": round(inter, 1),
        "area_a": round(area_a, 1),
        "area_b": round(area_b, 1),
        "pct_a_covered_by_b": round(pct_a_covered * 100, 1),
        "pct_b_covered_by_a": round(pct_b_covered * 100, 1),
        "center_distance_px": round(center_dist, 2),
        "bottom_center_distance_px": round(bc_dist, 2),
    }


def detect_all(frame):
    """Full raw model output at the SAME conf threshold as detect_vehicles_yolo
    (0.4) - keeps every field for the audit, not just count/label."""
    results = model(frame, verbose=False, conf=0.4)[0]
    dets = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = VEHICLE_CLASSES.get(cls_id, f"class_{cls_id}")
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = round(float(box.conf[0]), 4)
        dets.append({"cls_id": cls_id, "label": label, "conf": conf, "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return dets


def step1_frame_audit():
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
    all_dets = detect_all(target_frame)

    roi_in = []
    roi_out = []
    for i, det in enumerate(all_dets):
        bc = bottom_center_of(det["x1"], det["y1"], det["x2"], det["y2"])
        inside = cv2.pointPolygonTest(roi_polygon_cv, bc, False) >= 0
        row = {
            "index": i,
            "class": det["label"],
            "conf": det["conf"],
            "bbox_x1y1x2y2": [det["x1"], det["y1"], det["x2"], det["y2"]],
            "bbox_width": det["x2"] - det["x1"],
            "bbox_height": det["y2"] - det["y1"],
            "bbox_area": (det["x2"] - det["x1"]) * (det["y2"] - det["y1"]),
            "bottom_center": [round(bc[0], 1), round(bc[1], 1)],
        }
        if inside:
            row["distance_to_line_b_px"] = round(distance_point_to_line(bc, line_b[0], line_b[1]), 2)
            roi_in.append(row)
        else:
            roi_out.append(row)

    roi_in.sort(key=lambda r: r["distance_to_line_b_px"])
    for rank, r in enumerate(roi_in, start=1):
        r["assigned_queue_position"] = f"P{rank}"

    return video_w, video_h, target_frame, roi_polygon, line_b, roi_in, roi_out


def step2_pairwise_iou(roi_in):
    pairs = []
    for i in range(len(roi_in)):
        for j in range(i + 1, len(roi_in)):
            a, b = roi_in[i], roi_in[j]
            stats = iou_and_overlap(a["bbox_x1y1x2y2"], b["bbox_x1y1x2y2"])
            pairs.append({
                "pair": f"{a['assigned_queue_position']}({a['class']}) vs {b['assigned_queue_position']}({b['class']})",
                **stats,
            })
    return pairs


def step3_overlay(frame, roi_polygon, line_b, roi_in, roi_out):
    annotated = frame.copy()
    cv2.polylines(annotated, [np.array(roi_polygon, dtype=np.int32)], isClosed=True, color=(0, 140, 255), thickness=2)
    cv2.line(annotated, tuple(map(int, line_b[0])), tuple(map(int, line_b[1])), (0, 220, 120), 2)
    cv2.putText(annotated, "line_b", tuple(map(int, line_b[0])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 120), 2)

    colors = {"P1": (0, 0, 255), "P2": (255, 0, 255), "P3": (0, 200, 0)}
    for r in roi_in:
        x1, y1, x2, y2 = r["bbox_x1y1x2y2"]
        color = colors.get(r["assigned_queue_position"], (0, 200, 200))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        bc = tuple(int(v) for v in r["bottom_center"])
        cv2.circle(annotated, bc, 6, color, -1)
        label1 = f"{r['assigned_queue_position']} {r['class']} {r['conf']:.2f}"
        label2 = f"[{x1},{y1},{x2},{y2}]"
        cv2.putText(annotated, label1, (x1, max(0, y1 - 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(annotated, label2, (x1, max(0, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    header = f"clip_09.mp4 frame={TARGET_FRAME_IDX} - duplicate-detection audit (P1/P2/P3 only)"
    cv2.putText(annotated, header, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    out_path = VALIDATION_DIR / "clip09_duplicate_detection_audit.jpg"
    cv2.imwrite(str(out_path), annotated)
    return out_path


def step4_temporal_check(p1_bc, p2_bc, video_w, video_h):
    """Reuses the project's EXISTING ByteTrack pipeline (model.track with
    persist=True, tracker='bytetrack.yaml', same conf as track_crossings.py)
    - not a new tracking implementation."""
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

    window_records = []
    frame_idx = 0
    # region of interest for "near P1/P2" = within this pixel radius of
    # frame 1230's P1/P2 bottom-centers, tracked independently per candidate
    radius = 120

    for results in results_gen:
        if frame_idx > hi:
            break
        boxes = results.boxes
        if lo <= frame_idx <= hi and boxes is not None and boxes.id is not None:
            ids = boxes.id.int().tolist()
            clss = boxes.cls.int().tolist()
            confs = boxes.conf.tolist()
            xyxys = boxes.xyxy.tolist()
            for track_id, cls_id, conf, (x1, y1, x2, y2) in zip(ids, clss, confs, xyxys):
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                bc = bottom_center_of(x1, y1, x2, y2)
                near_p1 = ((bc[0] - p1_bc[0]) ** 2 + (bc[1] - p1_bc[1]) ** 2) ** 0.5 <= radius
                near_p2 = ((bc[0] - p2_bc[0]) ** 2 + (bc[1] - p2_bc[1]) ** 2) ** 0.5 <= radius
                if near_p1 or near_p2:
                    label = VEHICLE_CLASSES.get(cls_id, f"class_{cls_id}")
                    window_records.append({
                        "frame": frame_idx, "track_id": track_id, "class": label,
                        "conf": round(float(conf), 4), "bbox": [x1, y1, x2, y2],
                        "bottom_center": [round(bc[0], 1), round(bc[1], 1)],
                        "near_p1_region": near_p1, "near_p2_region": near_p2,
                    })
        frame_idx += 1

    return window_records


def main():
    video_w, video_h, frame, roi_polygon, line_b, roi_in, roi_out = step1_frame_audit()

    print("=== STEP 1: ROI_IN detection table (frame 1230) ===")
    for r in roi_in:
        print(r)
    print(f"\nROI_OUT count (for context only, not part of queue): {len(roi_out)}")

    print("\n=== STEP 2: pairwise IoU/overlap ===")
    pairs = step2_pairwise_iou(roi_in)
    for p in pairs:
        print(p)

    img_path = step3_overlay(frame, roi_polygon, line_b, roi_in, roi_out)
    print(f"\n=== STEP 3: overlay written to {img_path} ===")

    p1 = next((r for r in roi_in if r["assigned_queue_position"] == "P1"), None)
    p2 = next((r for r in roi_in if r["assigned_queue_position"] == "P2"), None)
    p3 = next((r for r in roi_in if r["assigned_queue_position"] == "P3"), None)

    temporal = None
    if p1 and p2:
        print("\n=== STEP 4: temporal check frames 1170-1290 near P1/P2 ===")
        temporal = step4_temporal_check(tuple(p1["bottom_center"]), tuple(p2["bottom_center"]), video_w, video_h)
        for rec in temporal:
            print(rec)

    report = {
        "roi_in": roi_in,
        "roi_out_count": len(roi_out),
        "pairwise_iou": pairs,
        "p1": p1, "p2": p2, "p3": p3,
        "temporal_window_records": temporal,
        "overlay_image": img_path.name,
        "model_nms_settings": {
            "iou": model.overrides.get("iou"),
            "agnostic_nms": model.overrides.get("agnostic_nms"),
            "max_det": model.overrides.get("max_det"),
        },
    }
    report_path = VALIDATION_DIR / "clip09_duplicate_detection_audit_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
