"""
READ-ONLY dual-model veto study. Reuses the EXISTING ambulance_overlap_sweep.json
candidate list (44 frames: 40 false across clip_01-11, 4 real in ambulense.mp4)
- no rescanning beyond what's needed. For each exact candidate frame, runs the
OLD best_before_ambulance_fix.pt (read-only, separate YOLO instance - the live
backend model path is never touched) and compares every old-model non-Ambulance
detection against the CURRENT model's Ambulance bbox via IoU + center distance,
keeping the best same-object match.

Does not modify main.py, any model, any config, any database record.
Writes only outputs/final_validation/dual_model_veto_study.json.
"""
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUT_DIR = BASE_DIR / "outputs" / "final_validation"

SWEEP_PATH = OUT_DIR / "ambulance_overlap_sweep.json"
OLD_WEIGHTS = BASE_DIR / "runs2" / "detect" / "traffic_monitor" / "weights" / "best_before_ambulance_fix.pt"

VEHICLE_CLASSES = {
    0: "Ambulance", 1: "Bicycle", 2: "Bus", 3: "Car",
    4: "Motorcycle", 5: "Truck", 6: "Tuktuk",
}

old_model = YOLO(str(OLD_WEIGHTS))


def bbox_xyxy(bbox_xywh):
    x, y, w, h = bbox_xywh
    return (x, y, x + w, y + h)


def center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def get_exact_frame(video_path, target_idx):
    cap = cv2.VideoCapture(str(video_path))
    frame_idx = 0
    target = None
    while frame_idx <= target_idx:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx == target_idx:
            target = frame.copy()
        frame_idx += 1
    cap.release()
    return target


def old_model_detect(frame):
    results = old_model(frame, verbose=False, conf=0.4)[0]
    dets = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = VEHICLE_CLASSES.get(cls_id, f"class_{cls_id}")
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = round(float(box.conf[0]), 3)
        dets.append({"label": label, "conf": conf, "bbox_xyxy": [x1, y1, x2, y2]})
    return dets


def main():
    sweep = json.load(open(SWEEP_PATH))

    results = []
    for video_entry in sweep:
        if video_entry.get("status") != "OK":
            continue
        video_name = video_entry["video"]
        is_real_ambulance_video = (video_name == "ambulense.mp4")
        video_path = UPLOAD_DIR / video_name

        for cand in video_entry["ambulance_candidates"]:
            frame_idx = cand["frame_idx"]
            time_s = cand["time_s"]
            current_conf = cand["ambulance_conf"]
            current_bbox = bbox_xyxy(cand["ambulance_bbox"])

            frame = get_exact_frame(video_path, frame_idx)
            if frame is None:
                results.append({
                    "video": video_name, "frame_idx": frame_idx, "time_s": time_s,
                    "group": "REAL" if is_real_ambulance_video else "FALSE",
                    "status": "FRAME_UNREACHABLE",
                })
                continue

            old_dets = old_model_detect(frame)
            non_amb = [d for d in old_dets if d["label"] != "Ambulance"]
            old_says_ambulance = [d for d in old_dets if d["label"] == "Ambulance"]

            best_match = None
            best_iou = -1
            for d in non_amb:
                i = iou(current_bbox, d["bbox_xyxy"])
                cd = ((center(current_bbox)[0] - center(d["bbox_xyxy"])[0]) ** 2
                      + (center(current_bbox)[1] - center(d["bbox_xyxy"])[1]) ** 2) ** 0.5
                if i > best_iou:
                    best_iou = i
                    best_match = {"old_class": d["label"], "old_conf": d["conf"],
                                  "old_bbox": d["bbox_xyxy"], "iou": round(i, 4),
                                  "center_distance_px": round(cd, 1)}

            results.append({
                "video": video_name,
                "frame_idx": frame_idx,
                "time_s": time_s,
                "group": "REAL" if is_real_ambulance_video else "FALSE",
                "status": "OK",
                "current_ambulance_conf": current_conf,
                "current_ambulance_bbox": current_bbox,
                "old_model_all_detections": old_dets,
                "old_model_says_ambulance_too": bool(old_says_ambulance),
                "old_model_ambulance_conf": old_says_ambulance[0]["conf"] if old_says_ambulance else None,
                "best_nonambulance_match": best_match,
            })

            tag = "REAL" if is_real_ambulance_video else "FALSE"
            match_desc = (f"{best_match['old_class']}(conf={best_match['old_conf']},iou={best_match['iou']},"
                          f"cdist={best_match['center_distance_px']})") if best_match else "NO_NONAMBULANCE_DETECTION"
            old_amb = f" OLD_ALSO_SAYS_AMBULANCE(conf={old_says_ambulance[0]['conf']})" if old_says_ambulance else ""
            print(f"[{tag}] {video_name} f={frame_idx} t={time_s}s cur_amb_conf={current_conf} "
                  f"-> old_best_match={match_desc}{old_amb}")

    out_path = OUT_DIR / "dual_model_veto_study.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
