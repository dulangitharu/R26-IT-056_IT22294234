"""
TrafficEye - Module 3: YOLO tracking + two-line crossing/speed logger.

Runs YOLOv8 + ByteTrack (via model.track(..., persist=True)) over a video,
keeps only vehicles whose bottom-center point falls inside the calibrated
ROI polygon, and logs each vehicle's crossing of line_a and line_b. The
line_b crossing time is used as the final crossing event (crossing_log CSV,
kept for backward compatibility); the time between line_a and line_b, plus
the calibrated real-world distance, produces a per-vehicle speed (speed_log
CSV). Also writes an annotated debug video for visual verification.

Standalone script - does not import or touch main.py / the /ws/stream
WebSocket endpoint.

Run from the backend/ directory:
    python track_crossings.py clip_01.mp4
    python track_crossings.py            # uses roi_config.json's video_name
"""

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
CALIBRATION_DIR = BASE_DIR / "calibration"
OUTPUT_DIR = BASE_DIR / "outputs"
ROI_CONFIG_PATH = CALIBRATION_DIR / "roi_config.json"
FIRST_FRAME_PATH = CALIBRATION_DIR / "first_frame.jpg"
MODEL_PATH = BASE_DIR / "runs2" / "detect" / "traffic_monitor" / "weights" / "best.pt"


CONF_THRESHOLD = 0.4
FLASH_FRAMES = 10          # how many frames the crossing highlight stays visible in the debug video
DEBUG_TARGET_FPS = 12      # write the debug video at a lower fps to keep file size reasonable
DEBUG_MAX_DIM = 960        # cap the longest side of the debug video for file size

ROI_COLOR = (0, 140, 255)      # orange (BGR)
LINE_A_COLOR = (60, 60, 255)   # red
LINE_B_COLOR = (0, 220, 120)   # green
IN_ROI_COLOR = (0, 200, 100)   # green
OUT_ROI_COLOR = (140, 140, 140)  # gray
POINT_COLOR = (255, 255, 0)    # cyan
FLASH_COLOR = (0, 255, 255)    # bright yellow


def side_of_line(point, line_start, line_end):
    """Sign of the cross product -> which side of the reference line `point` is on."""
    px, py = point
    x1, y1 = line_start
    x2, y2 = line_end
    cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
    if cross > 0:
        return 1
    if cross < 0:
        return -1
    return 0


def load_roi_config():
    if not ROI_CONFIG_PATH.exists():
        print(f"ERROR: {ROI_CONFIG_PATH} not found. Run mark_roi.py first.")
        sys.exit(1)
    with open(ROI_CONFIG_PATH) as f:
        return json.load(f)


def validate_roi_config(config):
    """Ensure roi_config.json uses the line_a/line_b/real_world_distance_m schema."""
    if "reference_line" in config and "line_a" not in config:
        print("ERROR: roi_config.json still uses the old 'reference_line' schema.")
        print("Please re-run mark_roi.py to regenerate roi_config.json with line_a, line_b, and real_world_distance_m.")
        sys.exit(1)

    missing = [k for k in ("line_a", "line_b", "real_world_distance_m") if k not in config]
    if missing:
        print(f"ERROR: roi_config.json is missing required key(s): {', '.join(missing)}")
        print("Please re-run mark_roi.py to regenerate roi_config.json with line_a, line_b, and real_world_distance_m.")
        sys.exit(1)

    distance = config.get("real_world_distance_m")
    if not isinstance(distance, (int, float)) or distance <= 0:
        print(f"ERROR: real_world_distance_m must be a positive number, got: {distance!r}")
        sys.exit(1)


def compute_scale(video_w, video_h):
    """
    ROI/reference-line points were clicked against first_frame.jpg's raw
    pixel dimensions (mark_roi.py displays the extracted frame unscaled).
    If the video we're tracking now has different dimensions, scale the
    calibrated points to match.
    """
    if not FIRST_FRAME_PATH.exists():
        print(f"ERROR: {FIRST_FRAME_PATH} not found. Run mark_roi.py first.")
        sys.exit(1)

    calib_img = cv2.imread(str(FIRST_FRAME_PATH))
    calib_h, calib_w = calib_img.shape[:2]

    if calib_w == video_w and calib_h == video_h:
        print(f"Calibration frame size: {calib_w}x{calib_h}  |  Video frame size: {video_w}x{video_h}  |  no scaling needed")
        return 1.0, 1.0

    scale_x = video_w / calib_w
    scale_y = video_h / calib_h
    print(f"Calibration frame size: {calib_w}x{calib_h}  |  Video frame size: {video_w}x{video_h}  |  "
          f"scale factor applied: ({scale_x:.4f}, {scale_y:.4f})")
    return scale_x, scale_y


def scale_points(points, scale_x, scale_y):
    return [[round(x * scale_x), round(y * scale_y)] for x, y in points]


def draw_debug_frame(frame, roi_polygon, line_a, line_b, tracks_this_frame, frame_idx, flash_until):
    annotated = frame.copy()

    cv2.polylines(annotated, [np.array(roi_polygon, dtype=np.int32)], isClosed=True, color=ROI_COLOR, thickness=2)

    cv2.line(annotated, tuple(line_a[0]), tuple(line_a[1]), LINE_A_COLOR, 2)
    cv2.putText(annotated, "line_a", tuple(line_a[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, LINE_A_COLOR, 2)

    cv2.line(annotated, tuple(line_b[0]), tuple(line_b[1]), LINE_B_COLOR, 2)
    cv2.putText(annotated, "line_b", tuple(line_b[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, LINE_B_COLOR, 2)

    for track_id, bbox, label, bottom_center, inside_roi in tracks_this_frame:
        x1, y1, x2, y2 = bbox
        flashing = frame_idx <= flash_until.get(track_id, -1)

        if flashing:
            color, thickness = FLASH_COLOR, 3
        elif inside_roi:
            color, thickness = IN_ROI_COLOR, 2
        else:
            color, thickness = OUT_ROI_COLOR, 1

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        tag = f"ID {track_id} {label}" + (" CROSSED!" if flashing else "")
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(annotated, tag, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.circle(annotated, (int(bottom_center[0]), int(bottom_center[1])), 5, POINT_COLOR, -1)

    return annotated


def run(video_name: str):
    config = load_roi_config()
    validate_roi_config(config)
    config_video_name = config.get("video_name")
    if config_video_name and config_video_name != video_name:
        print(f"WARNING: roi_config.json was calibrated for '{config_video_name}', "
              f"but running against '{video_name}'. Geometry may not line up correctly.")

    video_path = UPLOAD_DIR / video_name
    if not video_path.exists():
        print(f"ERROR: video '{video_path}' not found.")
        sys.exit(1)

    if not MODEL_PATH.exists():
        print(f"ERROR: model weights not found at '{MODEL_PATH}'.")
        sys.exit(1)

    cap_meta = cv2.VideoCapture(str(video_path))
    if not cap_meta.isOpened():
        print(f"ERROR: could not open video '{video_path}'.")
        sys.exit(1)
    orig_fps = cap_meta.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap_meta.get(cv2.CAP_PROP_FRAME_COUNT))
    video_w = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap_meta.release()

    scale_x, scale_y = compute_scale(video_w, video_h)
    roi_polygon = scale_points(config["roi_points"], scale_x, scale_y)
    line_a = scale_points(config["line_a"], scale_x, scale_y)
    line_b = scale_points(config["line_b"], scale_x, scale_y)
    real_world_distance_m = config["real_world_distance_m"]
    roi_polygon_cv = np.array(roi_polygon, dtype=np.float32)

    print(f"Loading model from {MODEL_PATH} ...")
    model = YOLO(str(MODEL_PATH))
    class_names = model.names  # authoritative class id -> label mapping, straight from the trained model
    print(f"model.names: {class_names}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    video_stem = Path(video_name).stem
    csv_path = OUTPUT_DIR / f"crossing_log_{video_stem}.csv"
    speed_csv_path = OUTPUT_DIR / f"speed_log_{video_stem}.csv"
    debug_video_path = OUTPUT_DIR / f"debug_{video_stem}.mp4"

    # Debug video sizing: cap longest side + reduce output fps for file size,
    # while tracking itself still runs on every full-resolution frame.
    debug_scale = min(1.0, DEBUG_MAX_DIM / max(video_w, video_h))
    debug_w, debug_h = round(video_w * debug_scale), round(video_h * debug_scale)
    write_every_n = max(1, round(orig_fps / DEBUG_TARGET_FPS))
    debug_out_fps = orig_fps / write_every_n

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(debug_video_path), fourcc, debug_out_fps, (debug_w, debug_h))

    last_side_a: dict[int, int] = {}     # track_id -> side of line_a on last frame seen inside ROI
    last_side_b: dict[int, int] = {}     # track_id -> side of line_b on last frame seen inside ROI
    line_a_crossings: dict[int, dict] = {}   # track_id -> {"frame": int, "time_s": float}
    line_b_crossings: dict[int, dict] = {}   # track_id -> {"frame": int, "time_s": float}
    vehicle_type_by_id: dict[int, str] = {}  # track_id -> last-seen class label
    counted_ids: set[int] = set()        # track_id -> already logged in crossing_log (line_b event)
    flash_until: dict[int, int] = {}     # track_id -> last frame index to keep the crossing highlight on
    crossings: list[dict] = []           # crossing_log rows (line_b crossing = final event)
    speed_rows: list[dict] = []          # speed_log rows (valid line_a -> line_b pairs)
    skipped_speed_count = 0              # line_b crossings that could not produce a valid speed row

    print(f"Tracking vehicles in '{video_name}' ({video_w}x{video_h} @ {orig_fps:.2f}fps, {total_frames} frames)...")

    results_gen = model.track(
        source=str(video_path),
        stream=True,
        persist=True,
        tracker="bytetrack.yaml",
        conf=CONF_THRESHOLD,
        verbose=False,
    )

    frame_idx = 0
    for results in results_gen:
        frame = results.orig_img
        tracks_this_frame = []

        boxes = results.boxes
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.int().tolist()
            clss = boxes.cls.int().tolist()
            xyxys = boxes.xyxy.tolist()

            for track_id, cls_id, (x1, y1, x2, y2) in zip(ids, clss, xyxys):
                label = class_names.get(cls_id, f"class_{cls_id}")
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                bottom_center = ((x1 + x2) / 2, y2)

                inside_roi = cv2.pointPolygonTest(roi_polygon_cv, bottom_center, False) >= 0

                if inside_roi:
                    vehicle_type_by_id[track_id] = label

                    # line_a crossing (first-time only, guarded by presence in line_a_crossings)
                    current_side_a = side_of_line(bottom_center, line_a[0], line_a[1])
                    prev_side_a = last_side_a.get(track_id)
                    if (
                        prev_side_a is not None
                        and prev_side_a != 0
                        and current_side_a != 0
                        and current_side_a != prev_side_a
                        and track_id not in line_a_crossings
                    ):
                        line_a_crossings[track_id] = {
                            "frame": frame_idx,
                            "time_s": round(frame_idx / orig_fps, 3),
                        }
                    last_side_a[track_id] = current_side_a

                    # line_b crossing (first-time only, guarded by presence in line_b_crossings)
                    current_side_b = side_of_line(bottom_center, line_b[0], line_b[1])
                    prev_side_b = last_side_b.get(track_id)
                    if (
                        prev_side_b is not None
                        and prev_side_b != 0
                        and current_side_b != 0
                        and current_side_b != prev_side_b
                        and track_id not in line_b_crossings
                    ):
                        line_b_crossings[track_id] = {
                            "frame": frame_idx,
                            "time_s": round(frame_idx / orig_fps, 3),
                        }
                        flash_until[track_id] = frame_idx + FLASH_FRAMES

                        # crossing_log (backward compatible): line_b crossing = final crossing event
                        if track_id not in counted_ids:
                            counted_ids.add(track_id)
                            crossings.append({
                                "vehicle_id": track_id,
                                "vehicle_type": label,
                                "frame_number": frame_idx,
                                "pass_time_seconds": line_b_crossings[track_id]["time_s"],
                            })

                        # speed_log: only valid if this vehicle crossed line_a first, in order,
                        # with a positive travel time and a positive calibrated distance.
                        a = line_a_crossings.get(track_id)
                        if a is not None:
                            travel_time_s = round(line_b_crossings[track_id]["time_s"] - a["time_s"], 3)
                            if travel_time_s > 0 and real_world_distance_m > 0:
                                speed_rows.append({
                                    "video_name": video_name,
                                    "vehicle_id": track_id,
                                    "vehicle_type": vehicle_type_by_id.get(track_id, label),
                                    "line_a_frame": a["frame"],
                                    "line_a_time_s": a["time_s"],
                                    "line_b_frame": line_b_crossings[track_id]["frame"],
                                    "line_b_time_s": line_b_crossings[track_id]["time_s"],
                                    "travel_time_s": travel_time_s,
                                    "speed_mps": round(real_world_distance_m / travel_time_s, 3),
                                })
                            else:
                                # travel_time_s <= 0 (same-frame double cross, or clock anomaly)
                                skipped_speed_count += 1
                        else:
                            # crossed line_b without a prior line_a crossing (wrong order, or
                            # line_a was never reached - e.g. vehicle entered ROI past line_a)
                            skipped_speed_count += 1

                    last_side_b[track_id] = current_side_b

                tracks_this_frame.append((track_id, (x1, y1, x2, y2), label, bottom_center, inside_roi))

        if frame_idx % write_every_n == 0:
            annotated = draw_debug_frame(frame, roi_polygon, line_a, line_b, tracks_this_frame, frame_idx, flash_until)
            if debug_scale != 1.0:
                annotated = cv2.resize(annotated, (debug_w, debug_h))
            writer.write(annotated)

        if frame_idx % 100 == 0:
            pct = f" ({frame_idx / total_frames * 100:.1f}%)" if total_frames else ""
            print(f"  frame {frame_idx}{f'/{total_frames}' if total_frames else ''}{pct} - "
                  f"{len(line_a_crossings)} line_a / {len(line_b_crossings)} line_b crossings so far")

        frame_idx += 1

    writer.release()

    crossings.sort(key=lambda r: r["pass_time_seconds"])
    with open(csv_path, "w", newline="") as f:
        writer_csv = csv.DictWriter(f, fieldnames=["vehicle_id", "vehicle_type", "frame_number", "pass_time_seconds"])
        writer_csv.writeheader()
        writer_csv.writerows(crossings)

    speed_rows.sort(key=lambda r: r["line_b_time_s"])
    with open(speed_csv_path, "w", newline="") as f:
        writer_speed_csv = csv.DictWriter(f, fieldnames=[
            "video_name", "vehicle_id", "vehicle_type",
            "line_a_frame", "line_a_time_s",
            "line_b_frame", "line_b_time_s",
            "travel_time_s", "speed_mps",
        ])
        writer_speed_csv.writeheader()
        writer_speed_csv.writerows(speed_rows)

    breakdown = Counter(r["vehicle_type"] for r in crossings)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Frames processed:      {frame_idx}")
    print(f"Total crossings:       {len(crossings)}")
    print("Breakdown by type:")
    if breakdown:
        for vtype, count in breakdown.most_common():
            print(f"  {vtype:12s} {count}")
    else:
        print("  (none)")
    print(f"Total line_a crossings: {len(line_a_crossings)}")
    print(f"Total line_b crossings: {len(line_b_crossings)}")
    print(f"Valid speed rows:       {len(speed_rows)}")
    print(f"Skipped/invalid speed rows: {skipped_speed_count}")
    print(f"CSV log saved to:      {csv_path}")
    print(f"Speed CSV saved to:    {speed_csv_path}")
    print(f"Debug video saved:     {debug_video_path}")


if __name__ == "__main__":
    config_for_default = load_roi_config() if ROI_CONFIG_PATH.exists() else {}
    default_video = config_for_default.get("video_name")

    parser = argparse.ArgumentParser(description="Track vehicles and log line_a/line_b crossings and speed for a video.")
    parser.add_argument("video_name", nargs="?", default=default_video,
                        help="Video filename inside backend/uploads/. Defaults to roi_config.json's video_name.")
    args = parser.parse_args()

    if not args.video_name:
        print("ERROR: no video_name given and none found in roi_config.json.")
        sys.exit(1)

    run(args.video_name)
