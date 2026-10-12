"""
READ-ONLY visual audit of clip_01.mp4, frames 280-460 - RAW FRAMES ONLY.

No YOLO, no ByteTrack, no detection/tracking inference of any kind. Reads
frames directly with OpenCV and builds a contact sheet for human visual
inspection. The only "known" coordinate used is the already-logged P4 Car
bbox at frame 280 (bbox=[390,430,489,532], from the earlier detection run),
drawn as a static reference box on the frame-280 tile only, purely to help
a human locate which vehicle is P4 - this is not a new detection.

Writes only:
  outputs/final_validation/observed_evidence/clip_01_P4_visual_audit_contact_sheet.jpg
"""
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "uploads" / "clip_01.mp4"
OUT_DIR = BASE_DIR / "outputs" / "final_validation" / "observed_evidence"
OUT_DIR.mkdir(exist_ok=True)

FPS = 10.47261777182907
START_FRAME = 280
END_FRAME = 460
STEP = 10

# Already-logged P4 Car bbox at frame 280 (from the earlier detection run) -
# drawn only as a static visual reference, not a new detection.
P4_BBOX_AT_280 = (390, 430, 489, 532)  # x1,y1,x2,y2

TILE_W, TILE_H = 280, 498  # downscaled from 576x1024, keep aspect
COLS = 5


def main():
    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {VIDEO_PATH}")

    tiles = []
    frame_idx = 0
    while frame_idx <= END_FRAME:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx >= START_FRAME and (frame_idx - START_FRAME) % STEP == 0:
            annotated = frame.copy()
            if frame_idx == START_FRAME:
                x1, y1, x2, y2 = P4_BBOX_AT_280
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 4)
                cv2.putText(annotated, "P4 Car (known @280)", (x1, max(0, y1 - 12)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            t_s = frame_idx / FPS
            cv2.putText(annotated, f"f={frame_idx} t={t_s:.3f}s", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 0), 3)
            resized = cv2.resize(annotated, (TILE_W, TILE_H))
            tiles.append((frame_idx, resized))
        frame_idx += 1
    cap.release()

    print(f"Extracted {len(tiles)} raw tiles from frame {START_FRAME} to {END_FRAME} (step {STEP})")

    rows = (len(tiles) + COLS - 1) // COLS
    sheet = np.full((rows * TILE_H, COLS * TILE_W, 3), 40, dtype=np.uint8)
    for i, (f_idx, tile) in enumerate(tiles):
        r, c = divmod(i, COLS)
        sheet[r * TILE_H:(r + 1) * TILE_H, c * TILE_W:(c + 1) * TILE_W] = tile

    out_path = OUT_DIR / "clip_01_P4_visual_audit_contact_sheet.jpg"
    cv2.imwrite(str(out_path), sheet)
    print(f"Wrote {out_path}")
    print("Frame indices included:", [f for f, _ in tiles])


if __name__ == "__main__":
    main()
