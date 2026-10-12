"""
READ-ONLY: generate ROI-aware snapshot candidates for clip_02-clip_08 and
clip_10, to select a replacement for clip_09 in the prototype
observed-clearance validation (clip_09 excluded - confirmed
CLASS_FLIP_SAME_TRACK cross-class duplicate).

Reuses the EXACT corrected runtime methodology already implemented and
verified in generate_final_validation_snapshots.py (which itself mirrors
main.py's create_analysis: every 10th frame -> YOLO -> ROI_IN filtering ->
densest-by-ROI_IN-count frame -> distance-to-line_b ordering) - imported,
not reimplemented. Does not modify main.py, traffic_calculations.py,
generate_final_validation_snapshots.py, roi_config.json, YOLO weights,
confidence threshold, NMS settings, or any calibration/validation CSV.
Does not overwrite outputs/final_validation/final_clearance_validation.csv.
Writes only new files under outputs/final_validation/replacement_candidates/.

Selection is NOT decided by this script - it only produces the
technical-validity evidence (ROI membership, duplicate/IoU screen, visual
snapshot) needed to apply Step 4/5's criteria by hand, in clip_02..clip_10
numerical order.
"""
import json
from pathlib import Path

import cv2
import numpy as np

from generate_final_validation_snapshots import (
    BASE_DIR, UPLOAD_DIR,
    find_densest_frame, order_queue, annotate_snapshot, bottom_center,
)
from traffic_calculations import build_queue_result

CANDIDATE_CLIPS = ["clip_02.mp4", "clip_03.mp4", "clip_04.mp4", "clip_05.mp4",
                   "clip_06.mp4", "clip_07.mp4", "clip_08.mp4", "clip_10.mp4"]

OUT_DIR = BASE_DIR / "outputs" / "final_validation" / "replacement_candidates"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IOU_FLAG_THRESHOLD = 0.80


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
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    iou = inter / union if union > 0 else 0.0
    return {
        "iou": round(iou, 4),
        "pct_a_covered_by_b": round((inter / area_a * 100) if area_a > 0 else 0, 1),
        "pct_b_covered_by_a": round((inter / area_b * 100) if area_b > 0 else 0, 1),
    }


def pairwise_iou_screen(ranked_accepted):
    flags = []
    for i in range(len(ranked_accepted)):
        for j in range(i + 1, len(ranked_accepted)):
            a, b = ranked_accepted[i], ranked_accepted[j]
            stats = iou_and_overlap(bbox_xyxy(a), bbox_xyxy(b))
            pair_label = f"P{i+1}({a['label']}) vs P{j+1}({b['label']})"
            entry = {"pair": pair_label, **stats,
                     "different_classes": a["label"] != b["label"]}
            if stats["iou"] > IOU_FLAG_THRESHOLD:
                entry["FLAGGED"] = True
            flags.append(entry)
    return flags


def main():
    report = []

    for clip_name in CANDIDATE_CLIPS:
        video_path = UPLOAD_DIR / clip_name
        if not video_path.exists():
            report.append({"clip_name": clip_name, "status": "MISSING_FILE"})
            continue

        scan = find_densest_frame(video_path)
        if scan["best_count"] == 0 or scan["best_frame"] is None:
            report.append({
                "clip_name": clip_name, "status": "NO_ROI_DETECTIONS",
                "note": "no vehicle detections inside the calibrated ROI in any sampled frame",
            })
            continue

        ordered_vehicle_types, ranked = order_queue(scan["best_accepted"], scan["line_b"])
        queue_result = build_queue_result(ordered_vehicle_types)
        if not queue_result["available"]:
            report.append({
                "clip_name": clip_name, "status": "QUEUE_RESULT_UNAVAILABLE",
                "blocker": queue_result["blocker"],
            })
            continue

        iou_flags = pairwise_iou_screen(ranked)
        high_iou_flags = [f for f in iou_flags if f.get("FLAGGED")]

        snapshot_time_s = round(scan["best_frame_idx"] / scan["fps"], 3)
        annotated = annotate_snapshot(
            scan["best_frame"], scan["roi_polygon"], scan["line_b"], ranked, scan["best_rejected"],
            clip_name, scan["best_frame_idx"], snapshot_time_s
        )
        snapshot_name = f"candidate_{Path(clip_name).stem}_snapshot.jpg"
        cv2.imwrite(str(OUT_DIR / snapshot_name), annotated)

        position_groups = [d["position_group"] for d in queue_result["ordered_queue"]]

        report.append({
            "clip_name": clip_name,
            "status": "OK",
            "snapshot_frame": scan["best_frame_idx"],
            "snapshot_time_s": snapshot_time_s,
            "fps": round(scan["fps"], 3),
            "roi_valid_vehicle_count": len(ordered_vehicle_types),
            "ordered_vehicle_types": ordered_vehicle_types,
            "position_groups": position_groups,
            "rejected_roi_out_count": len(scan["best_rejected"]),
            "rejected_roi_out_classes": [d["label"] for d in scan["best_rejected"]],
            "predicted_clearance_s": queue_result["queue_clearance_time_s"],
            "predicted_green_s": queue_result["suggested_green_time_s"],
            "pairwise_iou_screen": iou_flags,
            "high_iou_flagged_pairs": high_iou_flags,
            "snapshot_file": f"replacement_candidates/{snapshot_name}",
        })

    report_path = OUT_DIR / "replacement_candidates_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nWrote {report_path}")


if __name__ == "__main__":
    main()
