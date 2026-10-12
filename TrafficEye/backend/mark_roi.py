"""
TrafficEye - Semi-automatic ROI / Speed-Line marking tool.

Auto-extracts the first frame of a video, then lets the user click:
  - 4 points to define the ROI polygon (the road/lane area to analyze)
  - 2 points to define line_a (first speed-measurement line)
  - 2 points to define line_b (second speed-measurement line)

Before saving, prompts for the real-world distance (in meters) between
line_a and line_b, used later for speed calculation.

Run from the backend/ directory:
    python mark_roi.py clip_01.mp4

Controls:
    left click  - place the next point
    s           - save roi_config.json (once 4 ROI points + line_a + line_b are placed)
    r           - reset all points and start over
    q / ESC     - quit without saving
"""

import argparse
import json
import sys
from pathlib import Path

import cv2

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
CALIBRATION_DIR = BASE_DIR / "calibration"
FIRST_FRAME_PATH = CALIBRATION_DIR / "first_frame.jpg"
CONFIG_PATH = CALIBRATION_DIR / "roi_config.json"

WINDOW_NAME = "TrafficEye - Mark ROI + Reference Line"

ROI_POINT_COLOR = (0, 200, 255)
ROI_LINE_COLOR = (0, 200, 255)
LINE_A_COLOR = (60, 60, 255)
LINE_B_COLOR = (0, 220, 120)
TEXT_COLOR = (255, 255, 255)


def extract_first_frame(video_path: Path) -> "cv2.Mat":
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: could not open video '{video_path}'.")
        sys.exit(1)

    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"ERROR: could not read the first frame from '{video_path}'.")
        sys.exit(1)

    CALIBRATION_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(FIRST_FRAME_PATH), frame)
    print(f"Saved first frame -> {FIRST_FRAME_PATH}")
    return frame


class RoiMarker:
    def __init__(self, base_frame):
        self.base_frame = base_frame
        self.roi_points: list[list[int]] = []
        self.line_a_points: list[list[int]] = []
        self.line_b_points: list[list[int]] = []

    def reset(self):
        self.roi_points.clear()
        self.line_a_points.clear()
        self.line_b_points.clear()

    def on_mouse(self, event, x, y, flags, userdata):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if len(self.roi_points) < 4:
            self.roi_points.append([x, y])
        elif len(self.line_a_points) < 2:
            self.line_a_points.append([x, y])
        elif len(self.line_b_points) < 2:
            self.line_b_points.append([x, y])
        # else: all sets already complete, ignore further clicks

    def status_text(self) -> str:
        if len(self.roi_points) < 4:
            return f"Step 1/3: click ROI point {len(self.roi_points) + 1}/4"
        if len(self.line_a_points) < 2:
            return f"Step 2/3: click line_a point {len(self.line_a_points) + 1}/2"
        if len(self.line_b_points) < 2:
            return f"Step 3/3: click line_b point {len(self.line_b_points) + 1}/2"
        return "All points placed - press 's' to save or 'r' to reset"

    def render(self):
        frame = self.base_frame.copy()

        # ROI points + polygon
        for i, (x, y) in enumerate(self.roi_points):
            cv2.circle(frame, (x, y), 5, ROI_POINT_COLOR, -1)
            cv2.putText(frame, str(i + 1), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, ROI_POINT_COLOR, 2)
        if len(self.roi_points) == 4:
            cv2.polylines(frame, [self._np(self.roi_points)], isClosed=True,
                          color=ROI_LINE_COLOR, thickness=2)
        elif len(self.roi_points) > 1:
            cv2.polylines(frame, [self._np(self.roi_points)], isClosed=False,
                          color=ROI_LINE_COLOR, thickness=2)

        # line_a points + line
        for i, (x, y) in enumerate(self.line_a_points):
            cv2.circle(frame, (x, y), 5, LINE_A_COLOR, -1)
            cv2.putText(frame, str(i + 1), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, LINE_A_COLOR, 2)
        if len(self.line_a_points) == 2:
            cv2.line(frame, tuple(self.line_a_points[0]), tuple(self.line_a_points[1]),
                      LINE_A_COLOR, 2)
            cv2.putText(frame, "line_a", tuple(self.line_a_points[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, LINE_A_COLOR, 2)

        # line_b points + line
        for i, (x, y) in enumerate(self.line_b_points):
            cv2.circle(frame, (x, y), 5, LINE_B_COLOR, -1)
            cv2.putText(frame, str(i + 1), (x + 8, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, LINE_B_COLOR, 2)
        if len(self.line_b_points) == 2:
            cv2.line(frame, tuple(self.line_b_points[0]), tuple(self.line_b_points[1]),
                      LINE_B_COLOR, 2)
            cv2.putText(frame, "line_b", tuple(self.line_b_points[0]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, LINE_B_COLOR, 2)

        # HUD
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 32), (15, 15, 25), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame, self.status_text(), (10, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1)

        return frame

    @staticmethod
    def _np(points):
        import numpy as np
        return np.array(points, dtype=int)

    def is_complete(self) -> bool:
        return (len(self.roi_points) == 4
                and len(self.line_a_points) == 2
                and len(self.line_b_points) == 2)


def prompt_real_world_distance() -> float:
    while True:
        raw = input("Enter real-world distance between line_a and line_b (meters): ").strip()
        try:
            value = float(raw)
        except ValueError:
            print("Please enter a number, e.g. 10 or 12.5")
            continue
        if value <= 0:
            print("Distance must be greater than 0.")
            continue
        return value


def run(video_name: str):
    video_path = UPLOAD_DIR / video_name
    if not video_path.exists():
        print(f"ERROR: video '{video_path}' not found.")
        sys.exit(1)

    frame = extract_first_frame(video_path)
    marker = RoiMarker(frame)

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, marker.on_mouse)

    print("\nClick 4 ROI points, then 2 line_a points, then 2 line_b points.")
    print("Press 's' to save, 'r' to reset, 'q'/ESC to quit without saving.\n")

    while True:
        cv2.imshow(WINDOW_NAME, marker.render())
        key = cv2.waitKey(20) & 0xFF

        if key in (ord('q'), 27):  # ESC
            print("Quit without saving.")
            break

        elif key == ord('r'):
            marker.reset()
            print("Reset. Click 4 ROI points, then 2 line_a points, then 2 line_b points.")

        elif key == ord('s'):
            if not marker.is_complete():
                print(f"Cannot save yet - {marker.status_text()}")
                continue

            real_world_distance_m = prompt_real_world_distance()

            config = {
                "video_name": video_name,
                "roi_points": marker.roi_points,
                "line_a": marker.line_a_points,
                "line_b": marker.line_b_points,
                "real_world_distance_m": real_world_distance_m,
            }
            CALIBRATION_DIR.mkdir(exist_ok=True)
            with open(CONFIG_PATH, "w") as f:
                json.dump(config, f, indent=2)
            print(f"Saved ROI config -> {CONFIG_PATH}")
            print(json.dumps(config, indent=2))
            break

        # Window closed via the 'X' button
        if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mark ROI polygon + reference line on a video's first frame.")
    parser.add_argument("video_name", help="Video filename inside backend/uploads/, e.g. clip_01.mp4")
    args = parser.parse_args()

    run(args.video_name)
