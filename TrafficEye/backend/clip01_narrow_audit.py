"""
READ-ONLY: extract raw frames 405-435 of clip_01.mp4 individually (no
detection/tracking) as a tighter contact sheet, to pin down exactly what
happens to the P4 Car around the disputed frame 421.
"""
from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
VIDEO_PATH = BASE_DIR / "uploads" / "clip_01.mp4"
OUT_DIR = BASE_DIR / "outputs" / "final_validation" / "observed_evidence"

FPS = 10.47261777182907
START_FRAME = 405
END_FRAME = 435

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
        if frame_idx >= START_FRAME:
            annotated = frame.copy()
            t_s = frame_idx / FPS
            cv2.putText(annotated, f"f={frame_idx}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
            cv2.putText(annotated, f"t={t_s:.3f}s", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            resized = cv2.resize(annotated, (TILE_W, TILE_H))
            tiles.append((frame_idx, resized))
        frame_idx += 1
    cap.release()

    print(f"Extracted {len(tiles)} raw tiles from frame {START_FRAME} to {END_FRAME}")
    rows = (len(tiles) + COLS - 1) // COLS
    sheet = np.full((rows * TILE_H, COLS * TILE_W, 3), 40, dtype=np.uint8)
    for i, (f_idx, tile) in enumerate(tiles):
        r, c = divmod(i, COLS)
        sheet[r * TILE_H:(r + 1) * TILE_H, c * TILE_W:(c + 1) * TILE_W] = tile

    out_path = OUT_DIR / "clip_01_narrow_audit_405_435.jpg"
    cv2.imwrite(str(out_path), sheet)
    print(f"Wrote {out_path}")
    print("Frames:", [f for f, _ in tiles])


if __name__ == "__main__":
    main()
