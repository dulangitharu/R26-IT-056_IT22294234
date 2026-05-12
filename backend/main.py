"""
Traffic Density Monitor - Backend
FastAPI + YOLOv8 + OpenCV
Run: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import asyncio
import base64
import json
import os
import time
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# ── YOLO setup ──────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    # model = YOLO("yolov8n.pt")          # downloads ~6 MB on first run
    model = YOLO("runs2/detect/traffic_monitor/weights/best.pt")
    YOLO_AVAILABLE = True
    print("✅ YOLOv8 loaded")
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️  ultralytics not installed – using HOG fallback detector")

# COCO class IDs for vehicles
# VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
VEHICLE_CLASSES = {
    0: "Ambulance",
    1: "Bicycle",
    2: "Bus",
    3: "Car",
    4: "Motorcycle",
    5: "Truck",
}

# ── Traffic thresholds (tune for your road) ──────────────────────────────────
def get_traffic_level(count: int) -> dict:
    if count == 0:
        return {"level": "CLEAR",  "color": "#4ade80", "code": 0}
    elif count <= 4:
        return {"level": "LOW",    "color": "#86efac", "code": 1}
    elif count <= 10:
        return {"level": "MEDIUM", "color": "#fbbf24", "code": 2}
    else:
        return {"level": "HIGH",   "color": "#f87171", "code": 3}

# ── Fallback HOG detector (when YOLO not available) ──────────────────────────
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def detect_vehicles_hog(frame):
    """Simple HOG people detector as vehicle proxy for demo."""
    gray = cv2.resize(frame, (640, 360))
    rects, _ = hog.detectMultiScale(gray, winStride=(8, 8), padding=(4, 4), scale=1.05)
    return len(rects), [{"bbox": list(map(int, r)), "label": "vehicle", "conf": 0.7} for r in rects]

def detect_vehicles_yolo(frame):
    """YOLOv8 vehicle detection."""
    results = model(frame, verbose=False, conf=0.4)[0]
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id in VEHICLE_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detections.append({
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "label": VEHICLE_CLASSES[cls_id],
                # "label": "vehicle",
                "conf": round(float(box.conf[0]), 2),
            })
    return len(detections), detections

def detect_vehicles(frame):
    if YOLO_AVAILABLE:
        return detect_vehicles_yolo(frame)
    return detect_vehicles_hog(frame)

# ── Draw annotations on frame ────────────────────────────────────────────────
def annotate_frame(frame, detections, traffic_info, fps):
    annotated = frame.copy()
    h, w = annotated.shape[:2]

    # Draw bounding boxes
    for det in detections:
        x, y, bw, bh = det["bbox"]
        label = f"{det['label']} {det['conf']:.0%}"
        color = (0, 200, 100)
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(annotated, (x, y - th - 6), (x + tw + 4, y), color, -1)
        cv2.putText(annotated, label, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    # HUD overlay
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (w, 50), (15, 15, 25), -1)
    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

    level = traffic_info["level"]
    count = len(detections)
    cv2.putText(annotated, f"Vehicles: {count}", (12, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(annotated, f"Traffic: {level}", (w // 2 - 80, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(annotated, f"FPS: {fps:.1f}", (w - 110, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (180, 180, 180), 1)

    return annotated

def frame_to_jpeg_b64(frame, quality=75) -> str:
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode()

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="Traffic Monitor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Simple state store
state = {
    "video_path": None,
    "history": deque(maxlen=60),   # last 60 data points
}

@app.get("/")
def root():
    return {"status": "Traffic Monitor API running", "yolo": YOLO_AVAILABLE}

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Upload a video file for processing."""
    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        content = await file.read()
        f.write(content)
    state["video_path"] = str(dest)
    return {"filename": file.filename, "path": str(dest), "ok": True}

@app.get("/history")
def get_history():
    return list(state["history"])

@app.websocket("/ws/stream")
async def stream(ws: WebSocket):
    """
    WebSocket endpoint — streams annotated frames + stats.
    Client sends: { "action": "start" | "pause" | "stop", "speed": 1.0 }
    Server sends JSON: { frame, count, level, color, fps, timestamp, detections }
    """
    await ws.accept()
    paused = False
    speed = 1.0

    video_path = state.get("video_path")
    if not video_path or not Path(video_path).exists():
        await ws.send_json({"error": "No video uploaded yet."})
        await ws.close()
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        await ws.send_json({"error": "Cannot open video file."})
        await ws.close()
        return

    orig_fps   = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_delay  = 1.0 / orig_fps

    try:
        frame_idx = 0
        t_prev = time.time()

        while True:
            # Handle incoming control messages (non-blocking)
            try:
                msg = await asyncio.wait_for(ws.receive_json(), timeout=0.001)
                action = msg.get("action", "")
                if action == "pause":  paused = True
                elif action == "resume": paused = False
                elif action == "stop":  break
                elif action == "seek":
                    seek_to = int(msg.get("frame", 0))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, seek_to)
                    frame_idx = seek_to
                speed = float(msg.get("speed", speed))
            except asyncio.TimeoutError:
                pass
            except Exception:
                break

            if paused:
                await asyncio.sleep(0.05)
                continue

            ret, frame = cap.read()
            if not ret:
                # Loop video
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                continue

            # Resize for speed
            frame = cv2.resize(frame, (854, 480))

            # Detect
            count, detections = detect_vehicles(frame)
            traffic = get_traffic_level(count)

            # FPS calc
            t_now = time.time()
            fps   = 1.0 / max(t_now - t_prev, 0.001)
            t_prev = t_now

            # Annotate
            annotated = annotate_frame(frame, detections, traffic, fps)
            b64 = frame_to_jpeg_b64(annotated)

            payload = {
                "frame":      b64,
                "count":      count,
                "level":      traffic["level"],
                "color":      traffic["color"],
                "code":       traffic["code"],
                "fps":        round(fps, 1),
                "frame_idx":  frame_idx,
                "total":      total_frames,
                "timestamp":  time.time(),
                "detections": detections,
            }

            # Store in history (without frame to save memory)
            state["history"].append({k: v for k, v in payload.items() if k != "frame"})

            await ws.send_json(payload)

            frame_idx += 1
            await asyncio.sleep(frame_delay / speed)

    except WebSocketDisconnect:
        pass
    finally:
        cap.release()
