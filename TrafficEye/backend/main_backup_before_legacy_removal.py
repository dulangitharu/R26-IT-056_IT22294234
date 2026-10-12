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
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from bson import ObjectId
from bson.errors import InvalidId
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from pymongo.errors import DuplicateKeyError

from auth import create_access_token, get_current_officer, hash_password, verify_password
from database import analysis_collection, ensure_indexes, officers_collection
from models import AnalysisRecord, Officer, OfficerRegister
from traffic_calculations import (
    build_queue_result,
    calculate_clearance_time,
    calculate_queue_length,
    calculate_suggested_green_time,
)
from track_crossings import (
    FIRST_FRAME_PATH as ROI_FIRST_FRAME_PATH,
    ROI_CONFIG_PATH,
    compute_scale as roi_compute_scale,
    distance_point_to_line,
    scale_points as roi_scale_points,
)

# ── YOLO setup ──────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    # model = YOLO("yolov8n.pt")          # downloads ~6 MB on first run
    model = YOLO("runs2/detect/traffic_monitor/weights/best.pt")
    YOLO_AVAILABLE = True
    print(f"✅ YOLOv8 loaded — {len(model.names)} classes: {model.names}")
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
    6: "Tuktuk",
}

# ── Live-stream adaptive real-time sync (speed/accuracy trade-off) ──────────
# Running YOLO on every single frame is the main bottleneck for dense,
# high-resolution real-world footage (e.g. phone-recorded clips) — it can
# drag live playback down to just a few FPS. The /ws/stream loop below
# measures actual YOLO latency in real time and dynamically decides how many
# frames to skip detection on (reusing the last known detections/count in
# between) so total playback time tracks the video's real wall-clock
# duration — neither slow motion nor fast-forward. That trades some
# detection accuracy (a vehicle that appears/moves between processed frames
# can be briefly missed or shown with a stale box) for real-time playback.
# This only affects the live WebSocket stream — POST /api/analysis
# ("New Analysis") is untouched and keeps sampling every 10th frame
# independently for its own detection pass.
DETECT_TIME_EMA_ALPHA = 0.3  # smoothing factor for the rolling average of measured YOLO latency

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

def resize_letterboxed(frame, target_w=854, target_h=480):
    """
    Scale the frame to fit within (target_w, target_h) while preserving its
    aspect ratio, then pad with black bars to exactly fill the canvas.
    Avoids the stretching/squashing a plain cv2.resize to a fixed size causes.
    """
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))

    resized = cv2.resize(frame, (new_w, new_h))

    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    x_off = (target_w - new_w) // 2
    y_off = (target_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized

    return canvas

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

@app.on_event("startup")
async def on_startup():
    await ensure_indexes()

# ── Officer authentication ────────────────────────────────────────────────────
@app.post("/api/auth/register")
async def register_officer(officer_in: OfficerRegister):
    """Register a new Traffic Officer account."""
    existing = await officers_collection.find_one({"username": officer_in.username})
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists.")

    officer = Officer(
        username=officer_in.username,
        password_hash=hash_password(officer_in.password),
        full_name=officer_in.full_name,
        badge_number=officer_in.badge_number,
        created_at=datetime.now(timezone.utc),
    )

    try:
        await officers_collection.insert_one(officer.model_dump())
    except DuplicateKeyError:
        raise HTTPException(status_code=400, detail="Username already exists.")

    return {
        "username": officer.username,
        "full_name": officer.full_name,
        "message": "Officer registered successfully.",
    }

@app.post("/api/auth/login")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Authenticate an officer and issue a JWT access token."""
    officer = await officers_collection.find_one({"username": form_data.username})
    if not officer or not verify_password(form_data.password, officer["password_hash"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = create_access_token(officer["username"])
    return {"access_token": token, "token_type": "bearer"}

@app.get("/api/auth/me")
async def get_me(current_officer: dict = Depends(get_current_officer)):
    """Return the currently authenticated officer's public profile."""
    return {"username": current_officer["username"], "full_name": current_officer["full_name"]}

# ── TrafficEye decision-support endpoints ─────────────────────────────────────
def _serialize_record(doc: dict) -> dict:
    """Convert a MongoDB document's ObjectId _id into a plain string for JSON."""
    doc["_id"] = str(doc["_id"])
    return doc

@app.post("/api/analysis")
async def create_analysis(
    file: UploadFile = File(...),
    location_name: str = Form(...),
    road: str = Form(...),
    direction: str = Form(...),
    gps_lat: float = Form(...),
    gps_lng: float = Form(...),
    recorded_date: str = Form(...),
    recorded_time: str = Form(...),
    notes: Optional[str] = Form(None),
    current_officer: dict = Depends(get_current_officer),
):
    """
    Upload a traffic video + location metadata, run YOLO detection on
    sampled frames, compute queue length / clearance time / suggested
    green time from the densest frame, and persist the result to MongoDB.
    """
    print(f"📹 Received video for analysis: {file.filename}")

    dest = UPLOAD_DIR / file.filename
    with open(dest, "wb") as f:
        content = await file.read()
        f.write(content)

    cap = cv2.VideoCapture(str(dest))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable video.")

    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("🔍 Sampling frames (every 10th frame) for detection...")

    best_count = 0
    best_breakdown: dict[str, int] = {}
    best_detections: list = []
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % 10 == 0:
                count, detections = detect_vehicles_yolo(frame)
                if count > best_count:
                    breakdown: dict[str, int] = {}
                    for det in detections:
                        breakdown[det["label"]] = breakdown.get(det["label"], 0) + 1
                    best_count = count
                    best_breakdown = breakdown
                    best_detections = detections
            frame_idx += 1
    finally:
        cap.release()

    print(f"✅ Detection complete — densest sampled frame had {best_count} vehicles: {best_breakdown}")

    traffic = get_traffic_level(best_count)

    # ── Queue-position based calculation (final research methodology) ──────
    # Determine spatial queue order (closest to line_b = position 1) from the
    # densest frame's detections, reusing the SAME ROI/line_b calibration and
    # distance-to-line geometry track_crossings.py uses - never a blind
    # ascending/descending pixel guess. Only attempted when the uploaded
    # video is geometrically the one roi_config.json was calibrated for;
    # otherwise this reports a clear blocker rather than inventing an order.
    ordered_vehicle_types: Optional[list] = None
    queue_ordering_blocker: Optional[str] = None

    if best_detections:
        if not ROI_CONFIG_PATH.exists():
            queue_ordering_blocker = "calibration/roi_config.json not found - cannot determine spatial queue order"
        elif not ROI_FIRST_FRAME_PATH.exists():
            queue_ordering_blocker = "calibration/first_frame.jpg not found - cannot scale ROI/line geometry to this video"
        else:
            with open(ROI_CONFIG_PATH) as f:
                roi_config = json.load(f)
            if "line_b" not in roi_config:
                queue_ordering_blocker = "roi_config.json has no line_b - cannot determine queue-discharge reference line"
            else:
                config_video_name = roi_config.get("video_name")
                if config_video_name and config_video_name != file.filename:
                    queue_ordering_blocker = (
                        f"roi_config.json was calibrated for '{config_video_name}', not the uploaded "
                        f"'{file.filename}' - spatial queue ordering would not be geometrically valid for this footage"
                    )
                else:
                    scale_x, scale_y = roi_compute_scale(video_w, video_h)
                    line_b = roi_scale_points(roi_config["line_b"], scale_x, scale_y)
                    ranked = sorted(
                        best_detections,
                        key=lambda det: distance_point_to_line(
                            (det["bbox"][0] + det["bbox"][2] / 2, det["bbox"][1] + det["bbox"][3]),
                            line_b[0], line_b[1],
                        ),
                    )
                    ordered_vehicle_types = [det["label"] for det in ranked]
    else:
        queue_ordering_blocker = "no vehicle detections in the densest sampled frame"

    new_queue_result = build_queue_result(ordered_vehicle_types) if ordered_vehicle_types else {
        "available": False, "blocker": queue_ordering_blocker,
    }

    if new_queue_result["available"]:
        queue_length_m = new_queue_result["ordered_queue"][-1]["cumulative_distance_m"]
        clearance_time_s = new_queue_result["queue_clearance_time_s"]
        suggested_green_s = new_queue_result["suggested_green_time_s"]
        print(f"📊 Queue-position calculation complete — {len(ordered_vehicle_types)}-vehicle ordered queue, "
              f"clearance={clearance_time_s}s, suggested_green={suggested_green_s}s")
    else:
        # Legacy fallback - unchanged old behavior, kept for compatibility when
        # spatial ordering isn't available (e.g. ROI mismatch, no detections).
        queue_length_m = calculate_queue_length(best_breakdown)
        clearance_time_s = calculate_clearance_time(best_breakdown)
        suggested_green_s = calculate_suggested_green_time(clearance_time_s)
        print(f"⚠️  Queue-position calculation unavailable ({new_queue_result['blocker']}) - "
              f"using legacy count-based calculation. queue={queue_length_m}m, "
              f"clearance={clearance_time_s}s, suggested_green={suggested_green_s}s")

    record = AnalysisRecord(
        location_name=location_name,
        road=road,
        direction=direction,
        gps_lat=gps_lat,
        gps_lng=gps_lng,
        recorded_date=recorded_date,
        recorded_time=recorded_time,
        notes=notes,
        video_filename=file.filename,
        vehicle_count=best_count,
        vehicle_breakdown=best_breakdown,
        traffic_level=traffic["level"],
        queue_length_m=queue_length_m,
        clearance_time_s=clearance_time_s,
        suggested_green_s=suggested_green_s,
        status="completed",
        created_at=datetime.now(timezone.utc),
    )

    try:
        result = await analysis_collection.insert_one(record.model_dump())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not save analysis to MongoDB: {e}")

    print(f"💾 MongoDB save complete — record id {result.inserted_id}")

    saved = record.model_dump()
    saved["_id"] = str(result.inserted_id)

    # ── Additional queue-position based fields (response-only, not persisted
    # to MongoDB - AnalysisRecord's schema is unchanged). ordered_queue's raw/
    # FIFO-adjusted per-vehicle detail is for research/debug verification;
    # the frontend's normal UI only surfaces the emergency_* summary fields.
    if new_queue_result["available"]:
        saved["queue_calculation_method"] = "queue_position_based"
        saved["ordered_queue"] = new_queue_result["ordered_queue"]
        saved["average_gap_m"] = new_queue_result["average_gap_m"]
        saved["safety_buffer_s"] = new_queue_result["safety_buffer_s"]
        saved["emergency_detected"] = new_queue_result["emergency_detected"]
        if new_queue_result["emergency_detected"]:
            saved["emergency_vehicle_type"] = new_queue_result["emergency_vehicle_type"]
            saved["emergency_queue_position"] = new_queue_result["emergency_queue_position"]
            saved["emergency_position_group"] = new_queue_result["emergency_position_group"]
            saved["emergency_clearance_time_s"] = new_queue_result["emergency_clearance_time_s"]
            saved["emergency_priority_green_s"] = new_queue_result["emergency_priority_green_s"]
            saved["emergency_priority"] = new_queue_result["emergency_priority"]
            saved["emergency_recommendation"] = new_queue_result["emergency_recommendation"]
    else:
        saved["queue_calculation_method"] = "legacy_fallback"
        saved["queue_calculation_blocker"] = new_queue_result["blocker"]
        saved["emergency_detected"] = False

    return saved

@app.get("/api/analysis")
async def list_analysis():
    """Return all analysis records, newest first."""
    cursor = analysis_collection.find().sort("created_at", -1)
    return [_serialize_record(doc) async for doc in cursor]

@app.get("/api/analysis/{record_id}")
async def get_analysis(record_id: str):
    """Return one analysis record by its MongoDB _id."""
    try:
        oid = ObjectId(record_id)
    except InvalidId:
        raise HTTPException(status_code=404, detail=f"No record found with id '{record_id}'.")

    doc = await analysis_collection.find_one({"_id": oid})
    if not doc:
        raise HTTPException(status_code=404, detail=f"No record found with id '{record_id}'.")
    return _serialize_record(doc)

@app.get("/api/stats")
async def get_stats():
    """Aggregate stats across all analysis records."""
    total_cases = await analysis_collection.count_documents({})
    total_locations = len(await analysis_collection.distinct("location_name"))
    high_traffic_count = await analysis_collection.count_documents({"traffic_level": "HIGH"})

    avg_clearance_time = 0
    if total_cases > 0:
        pipeline = [{"$group": {"_id": None, "avg": {"$avg": "$clearance_time_s"}}}]
        result = await analysis_collection.aggregate(pipeline).to_list(length=1)
        if result:
            avg_clearance_time = round(result[0]["avg"], 1)

    return {
        "total_locations": total_locations,
        "total_cases": total_cases,
        "high_traffic_count": high_traffic_count,
        "avg_clearance_time": avg_clearance_time,
    }

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

    orig_fps     = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_period = 1.0 / orig_fps          # real-world seconds one video frame represents at 1x speed
    total_duration = total_frames / orig_fps  # video's actual real-time length, per requirement 1

    # ── Real-time sync anchors ───────────────────────────────────────────────
    # "elapsed real time since streaming started" (excluding time spent
    # paused) is compared against "where the video SHOULD be" to decide
    # whether to skip detection / catch up, or hold back with a small delay.
    stream_start_time = time.time()
    paused_time_total = 0.0
    pause_started_at: Optional[float] = None

    # Rolling estimate of how long a single detect_vehicles() call actually
    # takes on this machine/footage, updated every time detection runs.
    avg_detect_time: Optional[float] = None
    frames_since_detect = 0

    try:
        frame_idx = 0
        t_prev = time.time()
        last_count = 0
        last_detections: list = []
        last_traffic = get_traffic_level(0)

        while True:
            # Handle incoming control messages (non-blocking)
            try:
                msg = await asyncio.wait_for(ws.receive_json(), timeout=0.001)
                action = msg.get("action", "")
                if action == "pause":
                    if not paused:
                        paused = True
                        pause_started_at = time.time()
                elif action == "resume":
                    if paused and pause_started_at is not None:
                        paused_time_total += time.time() - pause_started_at
                        pause_started_at = None
                    paused = False
                elif action == "stop":
                    break
                elif action == "seek":
                    seek_to = int(msg.get("frame", 0))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, seek_to)
                    frame_idx = seek_to
                    # Re-anchor the sync clock so playback doesn't think it's
                    # instantly miles behind/ahead of the seeked position.
                    stream_start_time = time.time() - (frame_idx * frame_period) / max(speed, 1e-6)
                    paused_time_total = 0.0
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
                # Loop video — restart the sync clock too, so the next
                # play-through re-syncs cleanly instead of inheriting drift.
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx = 0
                stream_start_time = time.time()
                paused_time_total = 0.0
                continue

            # Resize for speed, preserving aspect ratio (letterboxed to a
            # consistent 854x480 canvas instead of stretching/squashing)
            frame = resize_letterboxed(frame, 854, 480)

            # ── Where SHOULD playback be right now? ─────────────────────────
            now = time.time()
            elapsed_real = now - stream_start_time - paused_time_total
            target_video_time = elapsed_real * speed
            current_video_time = frame_idx * frame_period

            # ── Dynamically decide whether to run detection this frame ─────
            # Speed/accuracy trade-off: derive how many frames one detection
            # call "costs" from the measured average latency vs. the
            # real-time budget per frame at the current speed (rounded, not
            # truncated, so e.g. a detection costing 1.8 frame-periods
            # correctly skips 1 frame between runs instead of rounding down
            # to "skip 0"). Slower detection (or higher speed) -> skip more
            # frames; fast detection -> skip few or none. On top of that
            # baseline, add a small, CAPPED extra skip proportional to how
            # far real elapsed time has drifted ahead of the video's decode
            # position (encode/send overhead isn't captured by avg latency
            # alone, so this closes that gap and keeps playback tracking
            # real time). The cap matters: an earlier version of this used
            # an uncapped hard veto ("skip whenever behind by more than one
            # frame-period") which — since a single detection call almost
            # always costs several frame-periods — stayed permanently true
            # and blocked detection almost entirely (the "stuck at 0
            # vehicles" regression). Capping the catch-up skip means
            # detection is always guaranteed to run at least once every
            # few frames, no matter how far behind we've fallen.
            MAX_EXTRA_CATCHUP_SKIP = 4
            budget = frame_period / max(speed, 1e-6)
            lag = target_video_time - current_video_time  # >0 = behind schedule
            if avg_detect_time is None:
                baseline_skip = 0  # unknown yet - measure it on this frame
            else:
                baseline_skip = max(0, round(avg_detect_time / budget) - 1)
            catchup_skip = min(MAX_EXTRA_CATCHUP_SKIP, max(0, int(lag / budget)))
            frames_to_skip = baseline_skip + catchup_skip

            if avg_detect_time is None or frames_since_detect >= frames_to_skip:
                t0 = time.time()
                count, detections = detect_vehicles(frame)
                measured = time.time() - t0
                avg_detect_time = measured if avg_detect_time is None else (
                    DETECT_TIME_EMA_ALPHA * measured + (1 - DETECT_TIME_EMA_ALPHA) * avg_detect_time
                )
                traffic = get_traffic_level(count)
                last_count, last_detections, last_traffic = count, detections, traffic
                frames_since_detect = 0
            else:
                # Skipped frame: reuse the last real detection result so the
                # dashboard keeps showing a sensible count/level instead of
                # dropping to 0/empty.
                count, detections, traffic = last_count, last_detections, last_traffic
                frames_since_detect += 1

            # FPS calc
            t_now = time.time()
            fps   = 1.0 / max(t_now - t_prev, 0.001)
            t_prev = t_now

            # Annotate
            annotated = annotate_frame(frame, detections, traffic, fps)
            b64 = frame_to_jpeg_b64(annotated)

            payload = {
                "frame":          b64,
                "count":          count,
                "level":          traffic["level"],
                "color":          traffic["color"],
                "code":           traffic["code"],
                "fps":            round(fps, 1),
                "frame_idx":      frame_idx,
                "total":          total_frames,
                "timestamp":      time.time(),
                "detections":     detections,
                "video_time_s":   round(current_video_time, 2),
                "real_elapsed_s": round(elapsed_real, 2),
                "total_duration_s": round(total_duration, 2),
            }

            # Store in history (without frame to save memory)
            state["history"].append({k: v for k, v in payload.items() if k != "frame"})

            await ws.send_json(payload)

            frame_idx += 1

            # ── Re-sync to wall clock ────────────────────────────────────────
            # If we're ahead of where the video should be at this point in
            # real time, hold back with a small delay; if behind, don't
            # sleep at all so the loop can catch up on the next iteration.
            now2 = time.time()
            elapsed_real2 = now2 - stream_start_time - paused_time_total
            target_video_time2 = elapsed_real2 * speed
            ahead_by = current_video_time - target_video_time2
            if ahead_by > 0:
                await asyncio.sleep(ahead_by)

    except WebSocketDisconnect:
        pass
    finally:
        cap.release()
