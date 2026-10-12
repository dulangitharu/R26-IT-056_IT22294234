"""
TrafficEye - ROI / speed-line verification tool.

Reads backend/calibration/roi_config.json, draws the saved ROI polygon and
line_a / line_b on top of backend/calibration/first_frame.jpg, and writes the
result to backend/calibration/roi_line_check.jpg. Non-interactive - no
window is opened.

Run from the backend/ directory:
    python verify_roi.py
"""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
CALIBRATION_DIR = BASE_DIR / "calibration"
FIRST_FRAME_PATH = CALIBRATION_DIR / "first_frame.jpg"
CONFIG_PATH = CALIBRATION_DIR / "roi_config.json"
OUTPUT_PATH = CALIBRATION_DIR / "roi_line_check.jpg"

ROI_COLOR = (0, 140, 255)     # orange (BGR)
LINE_A_COLOR = (60, 60, 255)  # red (BGR)
LINE_B_COLOR = (0, 220, 120)  # green (BGR)
TEXT_COLOR = (255, 255, 255)


def main():
    if not CONFIG_PATH.exists():
        print(f"ERROR: config file not found -> {CONFIG_PATH}")
        sys.exit(1)
    if not FIRST_FRAME_PATH.exists():
        print(f"ERROR: first frame not found -> {FIRST_FRAME_PATH}")
        sys.exit(1)

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    roi_points = config["roi_points"]
    line_a = config["line_a"]
    line_b = config["line_b"]
    real_world_distance_m = config.get("real_world_distance_m")

    frame = cv2.imread(str(FIRST_FRAME_PATH))
    if frame is None:
        print(f"ERROR: could not load image -> {FIRST_FRAME_PATH}")
        sys.exit(1)

    # ROI polygon
    for i, (x, y) in enumerate(roi_points):
        cv2.circle(frame, (x, y), 5, ROI_COLOR, -1)
        cv2.putText(frame, str(i + 1), (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, ROI_COLOR, 2)
    cv2.polylines(frame, [np.array(roi_points, dtype=int)], isClosed=True,
                  color=ROI_COLOR, thickness=2)

    # line_a
    for i, (x, y) in enumerate(line_a):
        cv2.circle(frame, (x, y), 5, LINE_A_COLOR, -1)
        cv2.putText(frame, str(i + 1), (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, LINE_A_COLOR, 2)
    cv2.line(frame, tuple(line_a[0]), tuple(line_a[1]), LINE_A_COLOR, 2)
    cv2.putText(frame, "line_a", tuple(line_a[0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, LINE_A_COLOR, 2)

    # line_b
    for i, (x, y) in enumerate(line_b):
        cv2.circle(frame, (x, y), 5, LINE_B_COLOR, -1)
        cv2.putText(frame, str(i + 1), (x + 8, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, LINE_B_COLOR, 2)
    cv2.line(frame, tuple(line_b[0]), tuple(line_b[1]), LINE_B_COLOR, 2)
    cv2.putText(frame, "line_b", tuple(line_b[0]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, LINE_B_COLOR, 2)

    CALIBRATION_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(OUTPUT_PATH), frame)
    print(f"Video: {config.get('video_name')}")
    print(f"ROI points: {roi_points}")
    print(f"line_a: {line_a}")
    print(f"line_b: {line_b}")
    print(f"real_world_distance_m: {real_world_distance_m}")
    print(f"Saved annotated check image -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
