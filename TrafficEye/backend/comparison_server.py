"""
YOLOv5 vs YOLOv8 — ISOLATED video-inference comparison demo backend.

This file is ADDITIVE and completely separate from the production TrafficEye
backend (main.py). It:
  * runs on its OWN port (default 8001), never 8000
  * has its OWN upload dir (backend/comparison_uploads/), never backend/uploads/
  * loads the EXISTING trained YOLOv5n + YOLOv8n comparison weights READ-ONLY
  * never imports main.py, never touches MongoDB / auth / ROI / queue / green-time
    / emergency / calibration / crossing logs / production model weights
  * exposes only  GET /  ,  POST /upload  ,  WS /ws/comparison

Small helpers (letterbox, jpeg-b64, traffic level, annotation style, class map)
are intentionally DUPLICATED from main.py rather than imported, so importing this
module can never load or disturb the production app.

Run:
    cd backend
    myfirstproject\\Scripts\\python.exe -m uvicorn comparison_server:app --host 127.0.0.1 --port 8001
"""

import asyncio
import base64
import os
import pathlib
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

# ── Existing trained comparison weights (READ-ONLY — never written/moved) ─────
BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
YOLOV5_WEIGHTS = PROJECT_ROOT / "experiments" / "model_comparison" / "yolov5" / "evidence" / "weights" / "best.pt"
YOLOV8_WEIGHTS = PROJECT_ROOT / "experiments" / "model_comparison" / "yolov8" / "weights" / "best.pt"

# The project's intended vehicle-detection inference confidence (matches
# detect_vehicles_yolo() in main.py). Applied IDENTICALLY to both models — the
# comparison is not tilted by per-model threshold tweaks. Each model keeps its
# own default NMS IoU (its standard/intended setting).
INFERENCE_CONF = 0.4

# TrafficEye's 7 vehicle classes (same mapping main.py uses). Both comparison
# checkpoints were trained with this exact class order (verified at load).
VEHICLE_CLASSES = {
    0: "Ambulance",
    1: "Bicycle",
    2: "Bus",
    3: "Car",
    4: "Motorcycle",
    5: "Truck",
    6: "Tuktuk",
}

COMPARISON_UPLOAD_DIR = BACKEND_DIR / "comparison_uploads"
COMPARISON_UPLOAD_DIR.mkdir(exist_ok=True)

CANVAS_W, CANVAS_H = 720, 480

# ── Load both models ONCE ───────────────────────────────────────────────────
_model_v5 = None
_model_v8 = None
_load_error = None


def _load_models() -> None:
    """Load YOLOv5n (torch.hub 'custom') and YOLOv8n (ultralytics) exactly once."""
    global _model_v5, _model_v8, _load_error
    if _model_v5 is not None and _model_v8 is not None:
        return
    try:
        import torch
        from ultralytics import YOLO

        # YOLOv5n — classic ultralytics/yolov5 checkpoint (trained on Colab).
        # Needs the yolov5 repo code (torch.hub cache) to deserialize; the
        # checkpoint pickles a PosixPath, so patch it to WindowsPath for the
        # load on Windows, then restore. No pretrained weights are downloaded —
        # only the repo source, and our own best.pt is what gets loaded.
        _orig_posix = pathlib.PosixPath
        pathlib.PosixPath = pathlib.WindowsPath
        try:
            m5 = torch.hub.load(
                "ultralytics/yolov5", "custom",
                path=str(YOLOV5_WEIGHTS), _verbose=False, trust_repo=True,
            )
        finally:
            pathlib.PosixPath = _orig_posix
        m5.conf = INFERENCE_CONF          # same conf as YOLOv8 below
        m5.classes = None                 # keep all 7 trained classes
        m5.eval() if hasattr(m5, "eval") else None
        _model_v5 = m5

        # YOLOv8n — fair-retrain comparison checkpoint.
        _model_v8 = YOLO(str(YOLOV8_WEIGHTS))

        print(f"✅ YOLOv5 loaded  classes={_model_v5.names}")
        print(f"✅ YOLOv8 loaded  classes={_model_v8.names}")
    except Exception as e:  # pragma: no cover - surfaced to the UI, not fatal at import
        _load_error = f"{type(e).__name__}: {e}"
        print(f"❌ Comparison model load failed: {_load_error}")


# ── Small self-contained helpers (duplicated from main.py on purpose) ─────────
def resize_letterboxed(frame, target_w=CANVAS_W, target_h=CANVAS_H):
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    resized = cv2.resize(frame, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def get_traffic_level(count: int) -> str:
    if count == 0:
        return "CLEAR"
    if count <= 4:
        return "LOW"
    if count <= 10:
        return "MEDIUM"
    return "HIGH"


def frame_to_jpeg_b64(frame, quality=72) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()


def annotate_comparison_frame(frame, detections, model_tag, fps):
    """Same visual style as main.py's annotate_frame(): GREEN boxes (0,200,100),
    '<Class> <conf%>' label with filled background + black text, plus a compact
    per-model HUD strip. Identical for both panels."""
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    for det in detections:
        x, y, bw, bh = det["bbox"]
        label = f"{det['label']} {det['conf']:.0%}"
        color = (0, 200, 100)
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ly = max(th + 6, y)
        cv2.rectangle(annotated, (x, ly - th - 6), (x + tw + 4, ly), color, -1)
        cv2.putText(annotated, label, (x + 2, ly - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 34), (15, 15, 25), -1)
    cv2.addWeighted(overlay, 0.78, annotated, 0.22, 0, annotated)
    cv2.putText(annotated, f"{model_tag}   Vehicles: {len(detections)}   {fps:.1f} FPS",
                (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return annotated


# ── Per-model inference → normalized detection list ─────────────────────────
def infer_v5(frame_bgr):
    """YOLOv5 AutoShape expects RGB numpy. Returns (detections, infer_ms)."""
    t0 = time.time()
    res = _model_v5(frame_bgr[..., ::-1])          # BGR -> RGB
    infer_ms = (time.time() - t0) * 1000.0
    dets = []
    arr = res.xyxy[0].cpu().numpy()                # x1,y1,x2,y2,conf,cls
    for x1, y1, x2, y2, conf, cls in arr:
        cid = int(cls)
        name = _model_v5.names.get(cid, VEHICLE_CLASSES.get(cid, str(cid)))
        dets.append({
            "bbox": [int(x1), int(y1), int(x2 - x1), int(y2 - y1)],
            "label": name,
            "conf": round(float(conf), 2),
        })
    return dets, infer_ms


def infer_v8(frame_bgr):
    """Ultralytics YOLO expects BGR numpy. Returns (detections, infer_ms)."""
    t0 = time.time()
    res = _model_v8(frame_bgr, verbose=False, conf=INFERENCE_CONF)[0]
    infer_ms = (time.time() - t0) * 1000.0
    dets = []
    for box in res.boxes:
        cid = int(box.cls[0])
        name = _model_v8.names.get(cid, VEHICLE_CLASSES.get(cid, str(cid)))
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        dets.append({
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "label": name,
            "conf": round(float(box.conf[0]), 2),
        })
    return dets, infer_ms


def class_counts(detections):
    out = {}
    for d in detections:
        out[d["label"]] = out.get(d["label"], 0) + 1
    return out


# ── FastAPI app ─────────────────────────────────────────────────────────────
app = FastAPI(title="YOLOv5 vs YOLOv8 Comparison Demo")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

_state = {"video_path": None}


@app.on_event("startup")
async def _startup():
    _load_models()


@app.get("/")
def root():
    return {
        "status": "YOLO comparison demo running",
        "models_loaded": _model_v5 is not None and _model_v8 is not None,
        "load_error": _load_error,
        "yolov5_weights": str(YOLOV5_WEIGHTS),
        "yolov8_weights": str(YOLOV8_WEIGHTS),
        "inference_conf": INFERENCE_CONF,
        "classes": VEHICLE_CLASSES,
    }


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Save ONE traffic video for the dual-model comparison. Isolated dir."""
    safe = Path(file.filename or "upload.mp4").name
    dest = COMPARISON_UPLOAD_DIR / safe
    with open(dest, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(str(dest))
    if not cap.isOpened():
        cap.release()
        return {"ok": False, "error": "Uploaded file is not a readable video."}
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.release()

    _state["video_path"] = str(dest)
    return {
        "ok": True,
        "filename": safe,
        "total_frames": total_frames,
        "fps": round(fps, 2),
        "duration_s": round(total_frames / fps, 1) if fps else None,
    }


@app.websocket("/ws/comparison")
async def ws_comparison(ws: WebSocket):
    """
    Synchronized dual-model stream. For EVERY source frame_idx the SAME frame is
    run through YOLOv5 AND YOLOv8 before the pair is sent, so neither model can
    ever run ahead of the other.

    Client → { action: "start"|"pause"|"resume"|"stop"|"seek", speed, frame }
    Server → {
        frame_idx, total_frames, video_time_s, progress,
        v5: { frame(b64), detections, count, class_counts, level, infer_ms, fps },
        v8: { frame(b64), detections, count, class_counts, level, infer_ms, fps },
    }
    """
    await ws.accept()

    if _load_error is not None or _model_v5 is None or _model_v8 is None:
        await ws.send_json({"error": f"Models not loaded: {_load_error or 'unknown error'}"})
        await ws.close()
        return

    video_path = _state.get("video_path")
    if not video_path or not Path(video_path).exists():
        await ws.send_json({"error": "No video uploaded yet — upload a video first."})
        await ws.close()
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        await ws.send_json({"error": "Cannot open the uploaded video."})
        await ws.close()
        return

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_period = 1.0 / src_fps

    paused = False
    speed = 1.0
    frame_idx = 0
    t_prev = time.time()

    try:
        while True:
            # ── control messages (non-blocking) ──────────────────────────────
            try:
                msg = await asyncio.wait_for(ws.receive_json(), timeout=0.001)
                action = msg.get("action", "")
                if action in ("pause",):
                    paused = True
                elif action in ("resume", "start"):
                    paused = False
                elif action == "stop":
                    break
                elif action == "seek":
                    frame_idx = max(0, int(msg.get("frame", 0)))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                if "speed" in msg and msg["speed"]:
                    speed = float(msg["speed"])
            except asyncio.TimeoutError:
                pass
            except Exception:
                break

            if paused:
                await asyncio.sleep(0.05)
                continue

            ret, frame = cap.read()
            if not ret:
                await ws.send_json({
                    "done": True, "frame_idx": frame_idx, "total_frames": total_frames,
                })
                # hold here until the client sends stop / seek
                paused = True
                continue

            # ONE canvas → BOTH models see the identical pixels for this frame_idx
            canvas = resize_letterboxed(frame, CANVAS_W, CANVAS_H)

            loop = asyncio.get_event_loop()
            (d5, ms5), (d8, ms8) = await asyncio.gather(
                loop.run_in_executor(None, infer_v5, canvas),
                loop.run_in_executor(None, infer_v8, canvas),
            )

            t_now = time.time()
            wall_fps = 1.0 / max(t_now - t_prev, 1e-3)
            t_prev = t_now
            v5_fps = 1000.0 / max(ms5, 1e-3)
            v8_fps = 1000.0 / max(ms8, 1e-3)

            a5 = annotate_comparison_frame(canvas, d5, "YOLOv5n", v5_fps)
            a8 = annotate_comparison_frame(canvas, d8, "YOLOv8n", v8_fps)

            payload = {
                "frame_idx": frame_idx,
                "total_frames": total_frames,
                "video_time_s": round(frame_idx * frame_period, 2),
                "progress": round(min(frame_idx / total_frames, 1.0) * 100, 1) if total_frames else 0,
                "wall_fps": round(wall_fps, 1),
                "v5": {
                    "frame": frame_to_jpeg_b64(a5),
                    "detections": d5,
                    "count": len(d5),
                    "class_counts": class_counts(d5),
                    "level": get_traffic_level(len(d5)),
                    "infer_ms": round(ms5, 1),
                    "fps": round(v5_fps, 1),
                },
                "v8": {
                    "frame": frame_to_jpeg_b64(a8),
                    "detections": d8,
                    "count": len(d8),
                    "class_counts": class_counts(d8),
                    "level": get_traffic_level(len(d8)),
                    "infer_ms": round(ms8, 1),
                    "fps": round(v8_fps, 1),
                },
            }
            await ws.send_json(payload)

            # advance by the SAME stride for both models
            stride = max(1, round(speed))
            if stride > 1:
                frame_idx += stride
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            else:
                frame_idx += 1

            # gentle pacing toward real time (both panels paced together)
            target_dt = frame_period * stride / max(speed, 1e-6)
            spent = time.time() - t_now
            if target_dt - spent > 0:
                await asyncio.sleep(min(target_dt - spent, 0.25))

    except WebSocketDisconnect:
        pass
    finally:
        cap.release()
