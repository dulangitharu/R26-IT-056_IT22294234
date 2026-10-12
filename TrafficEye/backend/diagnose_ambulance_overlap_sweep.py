"""
READ-ONLY sweep: for every video in uploads/clip_01.mp4..clip_11.mp4 and
ambulense.mp4, reproduce the CURRENT live emergency-candidate scan (reusing
main.py's select_camera_profile/detect_vehicles_yolo unmodified - same
model, same conf=0.4, same every-10th-frame sampling, same ROI membership
test), and for every sampled frame containing an ROI-filtered "Ambulance"
detection, compute IoU/overlap evidence against every OTHER ROI-filtered
detection in that same frame (especially Bus/Truck).

Does not modify main.py, any model, any config, any database record, or any
video file. Writes only outputs/final_validation/ambulance_overlap_sweep.json.
"""
import json
from pathlib import Path

import cv2
import numpy as np

import main as m  # reuse the EXACT live code

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUT_DIR = BASE_DIR / "outputs" / "final_validation"
OUT_DIR.mkdir(exist_ok=True)

VIDEOS = [f"clip_{i:02d}.mp4" for i in range(1, 12)] + ["ambulense.mp4"]


def bottom_center(det):
    return (det["bbox"][0] + det["bbox"][2] / 2, det["bbox"][1] + det["bbox"][3])


def bbox_xyxy(det):
    x, y, w, h = det["bbox"]
    return (x, y, x + w, y + h)


def iou_and_overlap(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0
    return {
        "iou": round(iou, 4),
        "pct_ambulance_covered": round((inter / area_a * 100) if area_a > 0 else 0, 1),
        "pct_other_covered": round((inter / area_b * 100) if area_b > 0 else 0, 1),
    }


def scan_video(video_name):
    video_path = UPLOAD_DIR / video_name
    if not video_path.exists():
        return {"video": video_name, "status": "MISSING_FILE"}

    cap = cv2.VideoCapture(str(video_path))
    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25

    selected_profile, roi_config, scale_fn, profile_scores = m.select_camera_profile(cap)
    if selected_profile is None:
        cap.release()
        return {"video": video_name, "status": "NO_CAMERA_PROFILE_MATCH", "profile_scores": profile_scores}

    scale_x, scale_y = scale_fn(video_w, video_h)
    line_b = m.roi_scale_points(roi_config["line_b"], scale_x, scale_y)
    roi_polygon = m.roi_scale_points(roi_config["roi_points"], scale_x, scale_y)
    roi_polygon_cv = np.array(roi_polygon, dtype=np.float32)

    ambulance_candidates = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % 10 == 0:
            _, raw_detections = m.detect_vehicles_yolo(frame)
            detections = [
                det for det in raw_detections
                if cv2.pointPolygonTest(roi_polygon_cv, bottom_center(det), False) >= 0
            ]
            amb_dets = [d for d in detections if d["label"] == "Ambulance"]
            for amb in amb_dets:
                overlaps = []
                for other in detections:
                    if other is amb:
                        continue
                    stats = iou_and_overlap(bbox_xyxy(amb), bbox_xyxy(other))
                    overlaps.append({
                        "other_label": other["label"],
                        "other_conf": other["conf"],
                        "other_bbox": other["bbox"],
                        **stats,
                    })
                overlaps.sort(key=lambda o: o["iou"], reverse=True)
                ambulance_candidates.append({
                    "frame_idx": frame_idx,
                    "time_s": round(frame_idx / source_fps, 3),
                    "ambulance_conf": amb["conf"],
                    "ambulance_bbox": amb["bbox"],
                    "roi_vehicle_count_this_frame": len(detections),
                    "all_labels_this_frame": [x["label"] for x in detections],
                    "overlaps_with_others": overlaps,
                    "best_overlap": overlaps[0] if overlaps else None,
                })
        frame_idx += 1
    cap.release()

    return {
        "video": video_name,
        "status": "OK",
        "selected_profile": selected_profile["id"],
        "profile_scores": profile_scores,
        "source_fps": source_fps,
        "ambulance_candidate_count": len(ambulance_candidates),
        "ambulance_candidates": ambulance_candidates,
    }


def main():
    all_results = []
    for video in VIDEOS:
        print(f"=== Scanning {video} ===")
        result = scan_video(video)
        status = result.get("status")
        print(f"  status={status}", end="")
        if status == "OK":
            print(f"  profile={result['selected_profile']}  ambulance_candidates={result['ambulance_candidate_count']}")
            for c in result["ambulance_candidates"]:
                best = c["best_overlap"]
                best_desc = f"{best['other_label']}(conf={best['other_conf']},iou={best['iou']})" if best else "NONE (isolated)"
                print(f"    frame={c['frame_idx']} t={c['time_s']}s amb_conf={c['ambulance_conf']} "
                      f"labels={c['all_labels_this_frame']} best_overlap={best_desc}")
        else:
            print(f"  {result}")
        all_results.append(result)

    out_path = OUT_DIR / "ambulance_overlap_sweep.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
