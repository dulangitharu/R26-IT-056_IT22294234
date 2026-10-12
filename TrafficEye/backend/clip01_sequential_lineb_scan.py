"""
READ-ONLY: sequentially read clip_01.mp4 (no seeking - cv2.set(POS_FRAMES) has
already been shown to be unreliable for this file) from frame 0, drawing the
KNOWN static line_b/ROI (from calibration/roi_config.json, no detection)
onto every frame in a target range, to visually locate the true crossing of
the P4 Car (the maroon/dark-red van) by eye. No YOLO, no ByteTrack.
"""
from pathlib import Path
import json
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "uploads" / "clip_01.mp4"
OUT_DIR = BASE_DIR / "outputs" / "final_validation" / "observed_evidence"

FPS = 10.47261777182907
START_FRAME = 400
END_FRAME = 480
STEP = 4

with open(BASE_DIR / "calibration" / "roi_config.json") as f:
    config = json.load(f)
LINE_B = config["line_b"]
ROI_POINTS = config["roi_points"]

TILE_W, TILE_H = 320, 569
COLS = 6


def main():
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    tiles = []
    frame_idx = 0
    while frame_idx <= END_FRAME:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= START_FRAME and (frame_idx - START_FRAME) % STEP == 0:
            annotated = frame.copy()
            cv2.polylines(annotated, [np.array(ROI_POINTS, dtype=np.int32)], isClosed=True, color=(0, 140, 255), thickness=2)
            cv2.line(annotated, tuple(LINE_B[0]), tuple(LINE_B[1]), (0, 220, 120), 3)
            t_s = frame_idx / FPS
            cv2.putText(annotated, f"f={frame_idx} t={t_s:.3f}s", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 3)
            resized = cv2.resize(annotated, (TILE_W, TILE_H))
            tiles.append((frame_idx, resized))
        frame_idx += 1
    cap.release()

    print(f"Extracted {len(tiles)} sequential+line_b tiles from frame {START_FRAME} to {END_FRAME} (step {STEP})")
    rows = (len(tiles) + COLS - 1) // COLS
    sheet = np.full((rows * TILE_H, COLS * TILE_W, 3), 40, dtype=np.uint8)
    for i, (f_idx, tile) in enumerate(tiles):
        r, c = divmod(i, COLS)
        sheet[r * TILE_H:(r + 1) * TILE_H, c * TILE_W:(c + 1) * TILE_W] = tile

    out_path = OUT_DIR / "clip_01_sequential_lineb_scan_400_480.jpg"
    cv2.imwrite(str(out_path), sheet)
    print(f"Wrote {out_path}")
    print("Frames:", [f for f, _ in tiles])


if __name__ == "__main__":
    main()
