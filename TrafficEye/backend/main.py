"""
Traffic Density Monitor - Backend
FastAPI + YOLOv8 + OpenCV
Run: uvicorn main:app --reload --host 0.0.0.0 --port 8000
"""

import asyncio
import base64
import json
import os
import subprocess
import tempfile
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import imageio_ffmpeg
import numpy as np
from bson import ObjectId
from bson.errors import InvalidId
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from pymongo.errors import DuplicateKeyError

from auth import create_access_token, get_current_officer, hash_password, verify_password
from database import analysis_collection, ensure_indexes, officers_collection
from models import AnalysisRecord, Officer, OfficerRegister
from traffic_calculations import build_queue_result
from track_crossings import (
    FIRST_FRAME_PATH as ROI_FIRST_FRAME_PATH,
    ROI_CONFIG_PATH,
    compute_scale as roi_compute_scale,
    distance_point_to_line,
    scale_points as roi_scale_points,
)
# Visualization-only class-name helper (annotated preview video). Never used by
# the research calculation - see display_label_helper.py and
# generate_annotated_recording() below.
from display_label_helper import remap_display_labels

# ── Camera/view compatibility (replaces the old clip_01..clip_11 filename
# allowlist) ──────────────────────────────────────────────────────────────
# The queue-position calibration (ROI polygon, line_b, discharge geometry) was
# hand-marked on ONE fixed camera's frame (calibration/first_frame.jpg, the
# frame roi_config.json's video_name was calibrated against). Applying that
# geometry to a different camera/angle would silently produce a meaningless
# queue order, so every upload must be visually verified against that same
# physical scene - a filename match/mismatch was never real evidence of that,
# it just happened to be true for the 11 clips originally tested. This does
# NOT introduce a new calibration system: it only decides whether the
# *existing* ROI/line_b/scaling logic (roi_compute_scale/roi_scale_points,
# still imported unchanged from track_crossings.py below) is allowed to run
# on a given upload.
#
# Method: ORB keypoint matching + RANSAC homography between the candidate
# video's frames (resized to calibration/first_frame.jpg's pixel dimensions -
# resolution match alone is NOT treated as proof of the same camera, only as
# a normalization step before real content comparison) and that calibration
# frame. Matching only a video's exact first frame proved unreliable across
# real lighting/exposure differences - one already-supported clip scored as
# few as 9 RANSAC inliers on frame 0 alone under harsh backlighting. Sampling
# a handful of frames spread across the clip and keeping the best-scoring one
# (after CLAHE contrast normalization) cleanly separates every known-
# compatible clip (>=122 inliers observed) from synthetic non-matches such as
# a rotated/flipped/random frame (<=9 inliers observed) - see
# inspect_orb_v2.py in the dev scratchpad for the measurements this was tuned
# against. The thresholds below sit well inside that margin.
#
# Fails closed: a video that cannot be confidently matched to the calibrated
# view is rejected as calibration_required, regardless of its filename.
CAMERA_MATCH_SAMPLE_FRACTIONS = (0.0, 0.15, 0.35, 0.55, 0.75)
CAMERA_MATCH_MIN_GOOD_MATCHES = 60
CAMERA_MATCH_MIN_RANSAC_INLIERS = 40

_camera_match_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
_camera_match_orb = cv2.ORB_create(nfeatures=3000)
_camera_match_bf = cv2.BFMatcher(cv2.NORM_HAMMING)

def _camera_match_descriptors(gray_frame):
    normalized = _camera_match_clahe.apply(gray_frame)
    return _camera_match_orb.detectAndCompute(normalized, None)

def _camera_match_score(kp_ref, des_ref, gray_candidate):
    kp, des = _camera_match_descriptors(gray_candidate)
    if des is None or des_ref is None or len(des) < 2 or len(des_ref) < 2:
        return 0, 0
    matches = _camera_match_bf.knnMatch(des_ref, des, k=2)
    good = []
    for pair in matches:
        if len(pair) == 2 and pair[0].distance < 0.8 * pair[1].distance:
            good.append(pair[0])
    if len(good) < 4:
        return len(good), 0
    src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 6.0)
    inliers = int(mask.sum()) if mask is not None else 0
    return len(good), inliers

# ── Multiple calibrated camera-view profiles ────────────────────────────────
# The fixed camera at this junction has been recorded with more than one
# zoom/tilt/framing (confirmed via a real E2E test: a real ambulance video
# visually matched the same physical junction strongly - 276 good / 227
# RANSAC inliers - yet every detection fell outside profile A's ROI, because
# that ROI's pixel coordinates were hand-marked for profile A's specific
# framing only). Each *framing* needs its own ROI polygon/line_b in image
# space; the underlying research pipeline (YOLO, vehicle lengths, gap,
# discharge-speed lookup, FIFO, clearance, buffer, suggested-green,
# emergency logic) is identical for every profile - only which pixel
# geometry gates/orders detections differs.
#
# Profile A (profile_a_original) is exactly the original, single calibration
# this file always had - its roi_config_path/reference_frame_path are the
# same ROI_CONFIG_PATH/ROI_FIRST_FRAME_PATH imported from track_crossings.py
# above, untouched, and it is scored/scaled using track_crossings.py's own
# compute_scale() so its behavior is byte-identical to before this change.
#
# Profile B (profile_b_ambulance_framing) is a second, separate config/
# reference-frame pair (calibration/roi_config_profile_b.json,
# calibration/first_frame_profile_b.jpg) - roi_config.json itself is never
# read or written by this profile.
CAMERA_PROFILES = [
    {"id": "profile_a_original", "roi_config_path": ROI_CONFIG_PATH, "reference_frame_path": ROI_FIRST_FRAME_PATH},
    {"id": "profile_b_ambulance_framing",
     "roi_config_path": Path("calibration/roi_config_profile_b.json"),
     "reference_frame_path": Path("calibration/first_frame_profile_b.jpg")},
]

def _best_match_for_reference(cap, kp_ref, des_ref, calib_w, calib_h):
    """Same multi-frame-sample-then-best-of-N matching used for the single
    profile before this change, now reusable per profile. Does not touch
    `cap`'s position beyond seeking during sampling - caller resets it."""
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    positions = sorted({max(0, int(total_frames * f)) for f in CAMERA_MATCH_SAMPLE_FRACTIONS}) or [0]
    best_good, best_inliers = 0, 0
    for pos in positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if not ok:
            continue
        resized = cv2.resize(frame, (calib_w, calib_h), interpolation=cv2.INTER_AREA)
        good, inliers = _camera_match_score(kp_ref, des_ref, cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY))
        if inliers > best_inliers:
            best_good, best_inliers = good, inliers
    return best_good, best_inliers

def select_camera_profile(cap: "cv2.VideoCapture"):
    """
    Scores `cap` (already open, at any read position) against EVERY entry in
    CAMERA_PROFILES with the same ORB/RANSAC visual match used for the single
    profile before this change - never chosen from filename. Returns
    (profile_dict, roi_config, scale_fn, scores) for the highest-RANSAC-
    inlier profile that both clears the compatibility threshold and has a
    valid roi_config (line_b + roi_points present), or (None, None, None,
    scores) if no profile qualifies - fails closed. Always leaves `cap`
    positioned at frame 0 when it returns.
    """
    scores: dict[str, str] = {}
    best = None  # (profile, roi_config, scale_fn, inliers)
    for profile in CAMERA_PROFILES:
        ref_path = profile["reference_frame_path"]
        cfg_path = profile["roi_config_path"]
        if not ref_path.exists() or not cfg_path.exists():
            scores[profile["id"]] = "unavailable (missing reference frame or config file)"
            continue
        with open(cfg_path) as f:
            roi_config = json.load(f)
        if "line_b" not in roi_config or "roi_points" not in roi_config:
            scores[profile["id"]] = "unavailable (roi_config missing line_b/roi_points)"
            continue
        calib_img = cv2.imread(str(ref_path)) 
        if calib_img is None:
            scores[profile["id"]] = "unavailable (unreadable reference frame)"
            continue
        calib_h, calib_w = calib_img.shape[:2]
        kp_ref, des_ref = _camera_match_descriptors(cv2.cvtColor(calib_img, cv2.COLOR_BGR2GRAY))
        good, inliers = _best_match_for_reference(cap, kp_ref, des_ref, calib_w, calib_h)
        scores[profile["id"]] = f"good={good} inliers={inliers}"

        if good >= CAMERA_MATCH_MIN_GOOD_MATCHES and inliers >= CAMERA_MATCH_MIN_RANSAC_INLIERS:
            if profile["id"] == "profile_a_original":
                scale_fn = roi_compute_scale
            else:
                def scale_fn(video_w, video_h, _cw=calib_w, _ch=calib_h):
                    if _cw == video_w and _ch == video_h:
                        return 1.0, 1.0
                    return video_w / _cw, video_h / _ch
            if best is None or inliers > best[3]:
                best = (profile, roi_config, scale_fn, inliers)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    selected_id = best[0]["id"] if best else None
    print(f"🎥 Camera/view profile scores: {scores} -> selected: {selected_id} "
          f"(need >={CAMERA_MATCH_MIN_GOOD_MATCHES}/{CAMERA_MATCH_MIN_RANSAC_INLIERS})")
    if best is None:
        return None, None, None, scores
    profile, roi_config, scale_fn, _ = best
    return profile, roi_config, scale_fn, scores

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

DETECT_CONF = 0.4  # production/research confidence gate - unchanged

def detect_vehicles_yolo(frame, conf: float = DETECT_CONF, imgsz=None):
    """
    YOLOv8 vehicle detection.

    `conf` and `imgsz` both default to the production settings (0.4 gate,
    Ultralytics' own default inference size) so every research caller -
    create_analysis()'s densest-frame loop, the scene-registration fallback,
    and the /ws/stream live path - is byte-identical to before. Only the
    visualization-only annotated-recording pass overrides them: a lower `conf`
    so the same model's genuine lower-confidence boxes are drawn instead of
    dropped, and a larger `imgsz` (the source frame's own resolution) so small
    vehicles in zoomed-out parts of a handheld clip are still found instead of
    the overlay vanishing. Neither override ever feeds vehicle_count / queue /
    green time / emergency, which still come from the default 0.4 / default
    imgsz pass.
    """
    predict_kwargs = {"verbose": False, "conf": conf}
    if imgsz is not None:
        predict_kwargs["imgsz"] = imgsz
    results = model(frame, **predict_kwargs)[0]
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


def normalize_to_calibration_frame(frame, calib_w: int, calib_h: int):
    """
    Map `frame` onto the calibration frame's exact pixel coordinate system so
    that YOLO detection (and therefore the ROI-membership test and the
    distance-to-line_b queue ordering) runs in the SAME coordinate space the
    ROI polygon / line_a / line_b were hand-calibrated in.

    The calibrated geometry (roi_config.json) was clicked against ONE fixed
    frame size (calibration/first_frame.jpg, calib_w x calib_h). Previously the
    analysis path scaled that geometry ONTO each upload with an independent
    per-axis factor (video_w/calib_w, video_h/calib_h) and ran detection on the
    raw upload. That silently distorts the ROI whenever a same-camera copy of
    the footage has a different aspect ratio, carries letterbox/pillarbox
    padding, or has non-square pixels (all common results of trimming /
    re-encoding / sending a clip through a messaging app). Normalizing the
    FRAME to the calibration size instead - aspect-ratio preserved, padded, and
    then comparing against the ROI at its native calibrated coordinates - keeps
    queue-position geometry valid for the same fixed camera at any capture
    resolution.

    A frame that is already exactly calib_w x calib_h is returned unchanged
    (no resize, no padding), so every video recorded natively at the
    calibration resolution behaves bit-for-bit as before this change.
    """
    h, w = frame.shape[:2]
    if w == calib_w and h == calib_h:
        return frame
    scale = min(calib_w / w, calib_h / h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (new_w, new_h), interpolation=interp)
    canvas = np.zeros((calib_h, calib_w, 3), dtype=np.uint8)
    x_off = (calib_w - new_w) // 2
    y_off = (calib_h - new_h) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


# ── Scene registration (same fixed camera, different zoom/crop/framing) ──────
# Fallback ONLY - used by create_analysis when the direct ROI test found no
# vehicle inside the calibrated ROI in any sampled frame. A genuine recording
# from the same fixed camera can still be framed differently from the one the
# ROI polygon was hand-marked on (a phone zoom mid-clip, a crop, a trimmed
# export at a wider view). The scene content is the same but its pixels sit
# elsewhere in the image, so the fixed ROI polygon no longer overlays the road.
# ORB keypoints + RANSAC on the RIGID background (buildings, signage, poles)
# recover the projective transform between the upload's frame and the
# calibration reference frame; RANSAC discards the moving-vehicle keypoints.
# This realigns the coordinates the ROI test / line_b ordering receive - it
# never changes YOLO, the ROI shape, line_b, the ordering rule, or any research
# value, and it never runs for an upload that already produced an ROI detection.
_scene_reg_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
_scene_reg_orb = cv2.ORB_create(nfeatures=4000)
_scene_reg_bf = cv2.BFMatcher(cv2.NORM_HAMMING)


def _scene_reg_descriptors(bgr_image):
    gray = _scene_reg_clahe.apply(cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY))
    return _scene_reg_orb.detectAndCompute(gray, None)


def scene_registration_homography(frame, kp_ref, des_ref):
    """
    Projective transform (3x3 H) mapping `frame`'s pixel coordinates onto the
    calibration reference frame's pixel coordinates, plus the RANSAC inlier
    count. Returns (None, n_good) when too few correspondences to be trusted.
    """
    kp, des = _scene_reg_descriptors(frame)
    if des is None or des_ref is None or len(kp) < 8 or len(kp_ref) < 8:
        return None, 0
    matches = _scene_reg_bf.knnMatch(des, des_ref, k=2)
    good = [m[0] for m in matches if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
    if len(good) < 12:
        return None, len(good)
    src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 6.0)
    inliers = int(mask.sum()) if mask is not None else 0
    return H, inliers


def _project_point(pt, H):
    """Map a single (x, y) through homography H -> (x', y') floats."""
    p = np.array([[[float(pt[0]), float(pt[1])]]], dtype=np.float32)
    out = cv2.perspectiveTransform(p, H)[0][0]
    return float(out[0]), float(out[1])

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

# ── Annotated-recording visualization (New Analysis → Analysis Results only) ──
# Entirely separate from annotate_frame()/the /ws/stream live path above -
# nothing here is called from or shared with the live WebSocket stream, so
# VideoAnalysis.jsx's behavior is unaffected by any of this.
def annotate_recorded_frame(frame, detections, display_labels=None):
    """
    Draws real YOLO bounding boxes + class labels for the recorded-analysis
    visualization video only. Deliberately does NOT draw the live vehicle-
    count/traffic-level/FPS HUD that annotate_frame() draws for /ws/stream -
    this is a recorded analysis result, not a live feed, so no LIVE/ROI
    ACTIVE/FPS text belongs here. Confidence is intentionally not displayed.

    Box coordinates ALWAYS come from `detections` (production best.pt).
    `display_labels`, when given, is a same-length list of visualization-only
    class-name strings (see display_label_helper.remap_display_labels); it can
    change ONLY the drawn text, never the box, never the count.
    """
    annotated = frame.copy()
    for i, det in enumerate(detections):
        x, y, bw, bh = det["bbox"]
        label = det["label"] if display_labels is None else display_labels[i]
        color = (0, 200, 100)
        cv2.rectangle(annotated, (x, y), (x + bw, y + bh), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        label_y = max(th + 6, y)
        cv2.rectangle(annotated, (x, label_y - th - 6), (x + tw + 4, label_y), color, -1)
        cv2.putText(annotated, label, (x + 2, label_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return annotated


# Analysis-preview output rate for generate_annotated_recording(). Full-frame
# CPU YOLO proved far too slow (see the clip_02 controlled E2E timing) - this
# trades smoothness for speed by only detecting/drawing/encoding a frame
# every ~1/4th of a second of source video, never every source frame. Lowered
# 5.0 -> 4.0 to cut the emergency-overlay wait by ~20-25% (fewer sampled
# frames): the preview plays a touch choppier but every frame it DOES contain
# still gets its own full, real YOLO detection - no frame is faked, repeated,
# or interpolated, and no research value is touched.
ANNOTATED_RECORDING_TARGET_FPS = 4.0

# Hard wall-clock cap on a single annotated-preview render. A pathologically
# large upload, a slow machine, or a stuck ffmpeg must never leave the
# EmergencyAlert page loading forever: past this many seconds the render stops
# with whatever real frames it has produced so far (still real detections, just
# fewer) and the file is published as-is. For the demo clips a full render is
# well under this.
ANNOTATED_RENDER_DEADLINE_S = 240

# Confidence gate for the VISUALIZATION-ONLY annotated preview video. Lower
# than the 0.4 research gate (DETECT_CONF) so genuine lower-confidence boxes
# from the SAME model - the distant cars / motorcycles / tuktuks alongside a
# nearer vehicle in a wide, low-resolution phone clip - are drawn instead of
# dropped. Measured on clip_07: 0.40 -> {Car,Moto,Tuktuk,Bus} but sparse;
# 0.30 roughly doubles the Motorcycle/Tuktuk boxes and keeps Car/Bus coverage,
# while staying above the band (<=0.25) where best.pt begins emitting
# misclassified low-confidence "Ambulance"/"Truck" labels on a normal clip.
# This value is used ONLY by generate_annotated_recording(); it never reaches
# vehicle_count, ROI membership, queue ordering, clearance/green time, or the
# emergency path, which all keep using DETECT_CONF.
ANNOTATED_DETECT_CONF = 0.30

# Sparse-frame retry for the VISUALIZATION preview only. When the primary
# annotated pass finds fewer than ANNOTATED_SPARSE_RETRY_TRIGGER vehicles in a
# sampled frame, run ONE more inference of the SAME model at the SAME size with
# a lower confidence (ANNOTATED_SPARSE_RETRY_CONF) and union the real boxes
# (IoU-deduped). Handheld zoom-out stretches push small vehicles below the
# primary 0.30 gate; the retry recovers the SAME model's genuine weaker
# detections on exactly those frames (measured on ambulense.mp4: frame coverage
# 70% -> 86%, longest un-boxed run 3.5s -> 2.1s, ~+8s render - the retry runs
# only on the sparse frames, same size so no inference re-warm-up). It never
# fabricates, moves, repeats, or hard-codes a box, and the research detector
# (detect_vehicles_yolo defaults - conf 0.4, default size) is untouched.
ANNOTATED_SPARSE_RETRY_TRIGGER = 2
ANNOTATED_SPARSE_RETRY_CONF = 0.20


def _iou_xywh(a, b):
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _merge_detections(primary, extra, iou_thr=0.5):
    """Union two detect_vehicles_yolo() outputs, dropping any `extra` box that
    already overlaps a kept box (IoU >= iou_thr). Real detections only - this
    never creates, moves, or duplicates a box."""
    merged = list(primary)
    for e in extra:
        if all(_iou_xywh(e["bbox"], k["bbox"]) < iou_thr for k in merged):
            merged.append(e)
    return merged


def generate_annotated_recording(video_path: Path, output_path: Path, emergency_hint: bool = False) -> bool:
    """
    Visualization-only pass for the immediate New Analysis result. Reopens
    the uploaded video in its OWN VideoCapture - completely independent of,
    and running after, the calculation loop in create_analysis() (which has
    already released its own capture by the time this is called).

    Runs at a fixed ~ANNOTATED_RECORDING_TARGET_FPS analysis-preview rate
    rather than every source frame (full-frame YOLO on CPU was too slow).
    Frames are sampled evenly across the WHOLE video based on the source's
    own real FPS (e.g. a 30 FPS source samples every round(30/5)=6th frame),
    and every frame that IS selected gets its own fresh, real YOLO
    detection - never a representative/neighboring frame's boxes, never
    interpolated, never fabricated. Encoding the sampled frames at
    ANNOTATED_RECORDING_TARGET_FPS (instead of the source rate) keeps total
    playback duration close to the source's, since fewer frames are shown
    for proportionally less time each.

    These per-frame detections are used ONLY to draw this video. They are
    never returned, aggregated, or combined with vehicle_count/
    vehicle_breakdown/queue_length_m/clearance_time_s/suggested_green_s/
    ordered_queue/traffic_level/emergency_* - all of those come exclusively
    from the existing every-10th-frame calculation loop and
    build_queue_result() in create_analysis(), which this function never
    touches or calls into.

    Returns True on success, False if the video can't be opened or ffmpeg
    fails - callers must treat False as "no annotated video available" and
    must not fabricate a URL for it.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if not source_fps or source_fps <= 0:
        source_fps = 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        return False

    # Evenly sample ~ANNOTATED_RECORDING_TARGET_FPS frames per second of
    # source video (e.g. 30 FPS source -> every 6th frame). Every sampled
    # frame still gets its own real, fresh YOLO detection below - nothing is
    # ever reused or interpolated between samples.
    sample_interval = max(1, round(source_fps / ANNOTATED_RECORDING_TARGET_FPS))

    # Visualization detector inference size: the source frame's own resolution
    # (multiple of 32, clamped to [640, 896]) instead of YOLO's 640 default.
    # A handheld clip that zooms OUT mid-recording shrinks every vehicle to a
    # few pixels; at 640 the model then finds nothing and the overlay silently
    # vanishes for that stretch even though the vehicles are still on screen.
    # Detecting at a larger size keeps them found and boxed. Capped at 896
    # (was 1024): ~25% less inference cost per frame for near-identical
    # small-object recall, cutting the emergency-overlay wait. Applied ONLY to
    # this preview pass - detect_vehicles_yolo()'s default (used by the research
    # loop, the fallback, and /ws/stream) is unchanged.
    annot_imgsz = int(min(896, max(640, round(max(width, height) / 32) * 32)))

    ffmpeg_cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(), "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{width}x{height}",
        "-r", str(ANNOTATED_RECORDING_TARGET_FPS),
        "-i", "-",
        "-an",
        # 'ultrafast' (was 'veryfast'): the SAME frames and the SAME real YOLO
        # detections are drawn and encoded - only the x264 speed preset changes,
        # so the annotated overlay is produced (and, since ultrafast also drops
        # CABAC / B-frames, starts decoding in the browser) noticeably sooner.
        # +faststart keeps the moov atom at the front so playback can begin
        # before the whole file has downloaded.
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]
    # ffmpeg's own progress output (frame=.../fps=.../speed=... lines) is
    # continuous and can be substantial over a multi-minute encode. Piping
    # stderr with subprocess.PIPE while nothing reads it deadlocks once the
    # OS pipe buffer fills: ffmpeg blocks writing stderr -> stops reading
    # stdin -> proc.stdin.write() below blocks forever -> both processes
    # stay alive but make no further progress (confirmed: a stalled real
    # run stopped growing its output file entirely, while ffmpeg's own
    # progress counter showed it had only muxed a handful of frames despite
    # the request running for over half an hour). Redirecting stderr to a
    # real temp file removes the bounded-pipe backpressure entirely - the
    # OS file system, not a fixed-size pipe buffer, absorbs ffmpeg's
    # output, so it can never block on writing it.
    with tempfile.TemporaryFile() as stderr_log:
        proc = subprocess.Popen(
            ffmpeg_cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=stderr_log,
        )

        render_started = time.monotonic()
        deadline_hit = False
        try:
            frame_idx = 0
            while True:
                if time.monotonic() - render_started > ANNOTATED_RENDER_DEADLINE_S:
                    deadline_hit = True
                    print(f"⏱️  Annotated render exceeded {ANNOTATED_RENDER_DEADLINE_S}s — "
                          f"publishing the {frame_idx // sample_interval} frames rendered so far.")
                    break
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % sample_interval == 0:
                    # Fresh detection on THIS sampled frame only - never
                    # best_detections from create_analysis()'s densest-frame
                    # selection, and never another sampled frame's detections.
                    # Visualization gate (ANNOTATED_DETECT_CONF) is lower than
                    # the 0.4 research gate so the SAME model's genuine
                    # lower-confidence boxes (e.g. distant cars/motorcycles
                    # beside a nearer tuktuk) are drawn rather than dropped, and
                    # annot_imgsz runs that detection at the frame's native
                    # resolution so small vehicles in zoomed-out stretches stay
                    # found. These detections are drawn only - never counted,
                    # ordered, or fed into any research value.
                    _, detections = detect_vehicles_yolo(
                        frame, conf=ANNOTATED_DETECT_CONF, imgsz=annot_imgsz
                    )
                    # Sparse-frame retry: only when this frame came back sparse,
                    # ask the SAME model (SAME size) once more at a lower
                    # confidence and union the real boxes. Recovers small
                    # zoomed-out vehicles the 0.30 gate dropped; adds no
                    # fabricated box.
                    if len(detections) < ANNOTATED_SPARSE_RETRY_TRIGGER:
                        _, extra = detect_vehicles_yolo(
                            frame, conf=ANNOTATED_SPARSE_RETRY_CONF, imgsz=annot_imgsz
                        )
                        detections = _merge_detections(detections, extra)
                    # Visualization-only hybrid labels: production best.pt owns
                    # every box (geometry + count); a second model may only
                    # correct the DISPLAYED class name. Fail-closed - any error
                    # here falls back to the detector's own labels, and every
                    # research value (already computed in create_analysis) is
                    # untouched.
                    try:
                        display_labels = remap_display_labels(
                            frame, detections, video_path.name, emergency_hint=emergency_hint
                        )
                    except Exception as e:
                        print(f"⚠️  hybrid display labels unavailable ({e}); using detector labels")
                        display_labels = None
                    annotated = annotate_recorded_frame(frame, detections, display_labels)
                    try:
                        proc.stdin.write(annotated.tobytes())
                    except BrokenPipeError:
                        break
                frame_idx += 1
        finally:
            cap.release()
            try:
                proc.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            # Bounded wait so a wedged ffmpeg can never hang the render thread
            # (and therefore the EmergencyAlert loading state) indefinitely.
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                print("⚠️  ffmpeg did not exit within 60s after stdin close — killing it.")
                proc.kill()
                proc.wait()

        if proc.returncode not in (0, None) and not deadline_hit:
            stderr_log.seek(0)
            stderr_tail = stderr_log.read().decode(errors="replace")[-500:]
            print(f"⚠️  Annotated-video encode failed (ffmpeg exit {proc.returncode}): {stderr_tail}")
            return False
    # stderr_log is closed (and its temp file removed) here, after any
    # diagnostic read above has already completed.

    # On a deadline stop, ffmpeg received a truncated raw stream; accept the
    # partial file only if it actually holds playable frames.
    return output_path.exists() and output_path.stat().st_size > 0


# ── Annotated-render status markers (visualization-only plumbing) ───────────
# Two tiny sidecar files can sit next to each reserved '<uuid>.mp4' in
# PROCESSED_DIR so the readiness probe can tell "still rendering" from
# "render finished" from "render gave up", and so a render interrupted by an
# `uvicorn --reload` restart can be resumed on the next startup instead of
# leaving the UI polling for the full ~10-minute ceiling.
#   <uuid>.mp4.job     JSON {source, emergency_hint, created} - a render that
#                      is supposed to exist. Removed once the render finishes
#                      or is declared failed.
#   <uuid>.mp4.failed  written when the render cannot produce a usable file
#                      (unreadable source, ffmpeg failure, or an interrupted
#                      render that could not be resumed).
# NONE of this reads or writes MongoDB, a detection, a count, a breakdown, or
# any ROI / queue / FIFO / green-time / emergency value. It only governs
# whether and when the green-box preview video is shown. The only files this
# code ever deletes are the '.job' markers it itself creates.
def _render_job_marker(final_output: Path) -> Path:
    return final_output.with_name(final_output.name + ".job")


def _render_failed_marker(final_output: Path) -> Path:
    return final_output.with_name(final_output.name + ".failed")


def _write_render_job(final_output: Path, source_video: Path, emergency_hint: bool) -> None:
    try:
        _render_job_marker(final_output).write_text(json.dumps({
            "source": str(source_video),
            "emergency_hint": bool(emergency_hint),
            "created": time.time(),
        }))
    except OSError as e:
        print(f"⚠️  Could not write annotated-render job marker for {final_output.name}: {e}")


def _clear_render_job(final_output: Path) -> None:
    try:
        _render_job_marker(final_output).unlink(missing_ok=True)
    except OSError:
        pass


def _mark_render_failed(final_output: Path) -> None:
    try:
        _render_failed_marker(final_output).write_text(str(time.time()))
    except OSError:
        pass
    _clear_render_job(final_output)


def _render_annotated_video_bg(source_video: Path, final_output: Path, emergency_hint: bool = False) -> None:
    """
    Background worker for the visualization-only annotated video. Scheduled via
    FastAPI BackgroundTasks so it runs in a worker thread AFTER the
    /api/analysis response has already been sent - the real research result is
    never made to wait for the ffmpeg encode.

    Renders with the EXISTING, unchanged generate_annotated_recording()
    pipeline, but writes to a temporary '<name>.part.mp4' first and only
    atomically renames it to `final_output` on success. That way
    GET /api/analysis-video-ready/{filename} can treat "the final file exists
    with non-zero size" as "fully rendered" and never serve a half-written MP4.

    Never raises, re-runs no calculation, and touches no research value. On
    success it clears the '<name>.job' marker; on any failure it writes a
    '<name>.failed' marker so the frontend can stop polling and fall back to
    the raw upload immediately instead of waiting out the full poll ceiling.
    """
    tmp_output = final_output.with_name(final_output.stem + ".part.mp4")
    try:
        ok = generate_annotated_recording(source_video, tmp_output, emergency_hint=emergency_hint)
        if ok and tmp_output.exists() and tmp_output.stat().st_size > 0:
            os.replace(tmp_output, final_output)
            _clear_render_job(final_output)
            print(f"🎬 Annotated video ready — /media/{final_output.name}")
        else:
            print(f"⚠️  Annotated video render produced no usable file for {final_output.name}")
            if tmp_output.exists():
                try:
                    tmp_output.unlink()
                except OSError:
                    pass
            _mark_render_failed(final_output)
    except Exception as e:
        print(f"⚠️  Annotated background render raised for {final_output.name}: {e}")
        try:
            if tmp_output.exists():
                tmp_output.unlink()
        except OSError:
            pass
        _mark_render_failed(final_output)


def _recover_orphaned_renders() -> None:
    """
    Startup-only recovery for annotated (green-box) preview renders that were
    interrupted - almost always by an `uvicorn --reload` restart while a
    render was still running. For each reserved '<uuid>.mp4.job':
      * final MP4 already present  -> drop the stale job marker;
      * source upload gone, job > 1 h old, or too many already queued this
        startup                    -> mark it failed so the UI stops polling
                                      on the ~10-minute ceiling;
      * otherwise                  -> re-dispatch the SAME unchanged render in
        a daemon thread (generate_annotated_recording runs ffmpeg with -y, so
        it just overwrites any leftover '<uuid>.part.mp4').

    Visualization plumbing only: never reads or writes MongoDB, a detection, a
    count, a breakdown, or any ROI / queue / FIFO / green-time / emergency
    value, and never loads or changes the model. The only files it removes are
    the '.job' status markers this feature itself creates.
    """
    resumed = 0
    MAX_RESUME = 3
    for job in sorted(PROCESSED_DIR.glob("*.mp4.job")):
        final_output = job.with_name(job.name[:-len(".job")])
        if final_output.is_file() and final_output.stat().st_size > 0:
            _clear_render_job(final_output)
            continue
        try:
            info = json.loads(job.read_text())
        except (OSError, ValueError):
            _mark_render_failed(final_output)
            continue
        source = Path(str(info.get("source", "")))
        created = float(info.get("created", 0) or 0)
        if not source.is_file() or (time.time() - created) > 3600:
            print(f"⚠️  Giving up on interrupted annotated render — /media/{final_output.name}")
            _mark_render_failed(final_output)
            continue
        if resumed >= MAX_RESUME:
            print(f"⚠️  Deferring interrupted annotated render — /media/{final_output.name} "
                  f"(re-run its analysis to retry)")
            _mark_render_failed(final_output)
            continue
        resumed += 1
        print(f"↻ Resuming interrupted annotated render — /media/{final_output.name}")
        threading.Thread(
            target=_render_annotated_video_bg,
            args=(source, final_output, bool(info.get("emergency_hint", False))),
            daemon=True,
        ).start()

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

# Annotated-recording output only - never the raw uploads directory, and
# nothing else on disk is exposed through this mount.
PROCESSED_DIR = Path("processed")
PROCESSED_DIR.mkdir(exist_ok=True)
app.mount("/media", StaticFiles(directory=str(PROCESSED_DIR)), name="media")

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
    # Visualization-only: re-dispatch or cleanly fail any green-box preview
    # render that an `uvicorn --reload` restart interrupted. Never touches
    # research data - see _recover_orphaned_renders.
    try:
        _recover_orphaned_renders()
    except Exception as e:
        print(f"⚠️  Orphaned annotated-render recovery skipped: {e}")

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
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    # Optional - the New Analysis UI no longer sends these (see
    # AnalysisRecord in models.py for why). recorded_date/recorded_time stay
    # required: the UI still collects and sends both of those.
    location_name: Optional[str] = Form(None),
    road: Optional[str] = Form(None),
    direction: Optional[str] = Form(None),
    gps_lat: Optional[float] = Form(None),
    gps_lng: Optional[float] = Form(None),
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
    # Stream the upload to disk in bounded chunks instead of buffering the whole
    # video into one in-memory bytes object before the analysis can start. Same
    # bytes on disk; this only trims the pre-analysis wait / peak memory for
    # large clips and keeps the event loop yielding while the file lands. No
    # detection / calculation / JSON behaviour is affected.
    with open(dest, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    cap = cv2.VideoCapture(str(dest))
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Uploaded file is not a readable video.")

    video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("🔍 Sampling frames (every 10th frame) for detection...")

    # ── Queue-position based calculation (final research methodology) ──────
    # Determine ROI/line_b usability BEFORE the sampling loop below, so the
    # SAME ROI-membership rule track_crossings.py's calibration pipeline
    # already uses (bottom-center inside roi_points, via
    # cv2.pointPolygonTest) can gate every detection during densest-frame
    # selection - not just after it. Only usable when the uploaded video is
    # geometrically the one roi_config.json was calibrated for; otherwise
    # this reports a clear blocker rather than inventing an order, and the
    # frame scan below falls back to an unfiltered count that is only ever
    # used for logging (a request for an unsupported video is rejected as
    # calibration_required below before any result is returned).
    roi_usable = False
    roi_polygon_cv = None
    line_b = None
    calib_w = calib_h = None
    queue_ordering_blocker: Optional[str] = None

    selected_profile, roi_config, scale_fn, profile_scores = select_camera_profile(cap)
    if selected_profile is None:
        queue_ordering_blocker = (
            f"'{file.filename}' does not visually match any calibrated fixed-camera view profile "
            f"({profile_scores}) - spatial queue ordering would not be geometrically valid for this footage"
        )
    else:
        # Normalize each sampled frame ONTO the calibration frame's coordinate
        # system (normalize_to_calibration_frame) and use the ROI polygon /
        # line_b at their NATIVE calibrated coordinates - do NOT scale the
        # calibrated geometry onto the (possibly trimmed / re-encoded /
        # differently-padded) upload. This keeps queue-position geometry valid
        # for the SAME fixed camera recorded at any resolution, and is an exact
        # no-op when the upload is already at the calibration resolution.
        calib_ref_img = cv2.imread(str(selected_profile["reference_frame_path"]))
        if calib_ref_img is None:
            # select_camera_profile already verified this file is readable, so
            # this is unreachable in practice - guard only so a disk/permission
            # regression fails loudly instead of raising AttributeError below.
            raise HTTPException(status_code=500, detail="Calibration reference frame became unreadable.")
        calib_h, calib_w = calib_ref_img.shape[:2]
        line_b = roi_scale_points(roi_config["line_b"], 1.0, 1.0)
        roi_polygon = roi_scale_points(roi_config["roi_points"], 1.0, 1.0)
        roi_polygon_cv = np.array(roi_polygon, dtype=np.float32)
        roi_usable = True
        print(f"✅ Using camera profile: {selected_profile['id']} — detection frames normalized "
              f"to calibration space {calib_w}x{calib_h} (upload is {video_w}x{video_h})")

    best_count = 0
    best_breakdown: dict[str, int] = {}
    best_detections: list = []
    best_total_detected = 0
    frame_idx = 0

    # Parallel, independent emergency-candidate tracking (see below the loop
    # for why): tracks whichever sampled ROI-filtered frame contains a real
    # Ambulance detection and has the MOST ROI vehicles among such frames -
    # entirely separate from best_count/best_breakdown/best_detections above,
    # which this never reads or writes. Ties keep the earliest frame, since
    # the loop below only overwrites on strict `>` while frame_idx increases.
    emergency_best_count = 0
    emergency_best_frame_idx = None
    emergency_best_detections: list = []

    try:
        while True:
            # Walk the video one frame at a time with grab() (advance only) and
            # pay the full decode (retrieve) ONLY for the frames this pass
            # actually samples. This is byte-identical to the previous
            # `ret, frame = cap.read()` every frame: same sequential walk, same
            # frame_idx, same "every 10th frame" rule, same pixels handed to
            # YOLO — the ~90% of frames that were decoded and then immediately
            # discarded are simply not decoded now. It is NOT a seek (a seek
            # would snap to keyframes and change which frame each index maps
            # to); it only removes wasted colour-convert/copy work.
            if not cap.grab():
                break
            if frame_idx % 10 == 0:
                ret, frame = cap.retrieve()
                if not ret:
                    break
                # When a calibrated profile matched, run detection in the
                # calibration frame's coordinate system so the ROI test and the
                # line_b ordering below compare like-for-like. No-op for an
                # upload already at the calibration resolution.
                det_frame = (
                    normalize_to_calibration_frame(frame, calib_w, calib_h)
                    if roi_usable else frame
                )
                raw_count, raw_detections = detect_vehicles_yolo(det_frame)
                detections = raw_detections
                count = raw_count
                if roi_usable:
                    # A detection is a queue candidate ONLY when its bbox
                    # bottom-center is inside/on the calibrated ROI polygon -
                    # the densest frame is now "most valid queue detections
                    # inside the ROI", not "most detections anywhere".
                    detections = [
                        det for det in raw_detections
                        if cv2.pointPolygonTest(
                            roi_polygon_cv,
                            (det["bbox"][0] + det["bbox"][2] / 2, det["bbox"][1] + det["bbox"][3]),
                            False,
                        ) >= 0
                    ]
                    count = len(detections)
                if count > best_count:
                    breakdown: dict[str, int] = {}
                    for det in detections:
                        breakdown[det["label"]] = breakdown.get(det["label"], 0) + 1
                    best_count = count
                    best_breakdown = breakdown
                    best_detections = detections
                    # Raw YOLO vehicle-class detections in this SAME sampled
                    # frame, before the ROI filter above - display/stat only,
                    # never fed into queue length/clearance/green-time below.
                    best_total_detected = raw_count
                # Same raw YOLO call, same ROI-filtered `detections` computed
                # above - no second detection pass. An Ambulance can appear
                # in a sampled frame that is NOT the normal densest frame
                # (e.g. it arrives alone, before/after the densest moment);
                # without this separate track, emergency detection would be
                # silently tied to whichever frame happens to win the normal
                # vehicle-count comparison above, which is not the same
                # question as "was an Ambulance ever seen in the ROI".
                if roi_usable and count > emergency_best_count and any(det["label"] == "Ambulance" for det in detections):
                    emergency_best_count = count
                    emergency_best_frame_idx = frame_idx
                    emergency_best_detections = detections
            frame_idx += 1
    finally:
        cap.release()

    print(f"✅ Detection complete — densest sampled frame had {best_count} vehicles: {best_breakdown}"
          + (" (ROI-filtered)" if roi_usable else ""))

    # ── Scene-registration fallback ──────────────────────────────────────────
    # The profile matched (same physical junction) but the direct ROI test
    # found no vehicle inside the calibrated ROI in ANY sampled frame. Before
    # reporting calibration_required, retry ONCE with each sampled frame
    # projectively registered onto the calibration reference frame, so a
    # genuine same-fixed-camera recording at a different zoom / crop / framing
    # still has its real YOLO detections tested against the real ROI polygon -
    # in the ROI's own coordinate space (see scene_registration_homography).
    #
    # This block cannot affect any upload that already produced an ROI
    # detection (`not best_detections` guard), so every currently-working
    # video is byte-identical. It does NOT bypass the ROI test (each recovered
    # detection must still fall inside roi_polygon_cv), does NOT fabricate
    # anything (real detect_vehicles_yolo output), and does NOT touch line_b,
    # the ordering rule, build_queue_result, green-time, or emergency logic -
    # only the pixel coordinates handed to pointPolygonTest / distance_point_
    # to_line are realigned.
    if roi_usable and not best_detections:
        try:
            kp_ref, des_ref = _scene_reg_descriptors(calib_ref_img)
        except Exception as e:  # pragma: no cover - defensive
            kp_ref, des_ref = None, None
            print(f"⚠️  Scene-registration fallback unavailable ({e})")
        if des_ref is not None:
            print("↻ No ROI detections in direct pass — retrying with scene registration "
                  "to the calibration frame…")
            cap_reg = cv2.VideoCapture(str(dest))
            reg_idx = 0
            reg_best_inliers = 0
            try:
                while True:
                    # Same grab()/retrieve() split as the main loop above:
                    # advance every frame, decode only the every-10th frames
                    # this fallback actually samples. Byte-identical sampling,
                    # no seek — just no wasted decode on skipped frames.
                    if not cap_reg.grab():
                        break
                    if reg_idx % 10 == 0:
                        ok, frame = cap_reg.retrieve()
                        if not ok:
                            break
                        H, inliers = scene_registration_homography(frame, kp_ref, des_ref)
                        if H is not None and inliers >= CAMERA_MATCH_MIN_RANSAC_INLIERS:
                            reg_best_inliers = max(reg_best_inliers, inliers)
                            raw_count, raw_detections = detect_vehicles_yolo(frame)
                            registered = []
                            for det in raw_detections:
                                bx = det["bbox"]
                                cx, cy = _project_point(
                                    (bx[0] + bx[2] / 2, bx[1] + bx[3]), H
                                )
                                if cv2.pointPolygonTest(roi_polygon_cv, (cx, cy), False) >= 0:
                                    # Rebuild the bbox around the calibration-space
                                    # bottom-center so the existing (x+w/2, y+h)
                                    # ordering math below is unchanged. Width/height
                                    # are only used for the type breakdown.
                                    registered.append({
                                        "label": det["label"],
                                        "conf": det.get("conf"),
                                        "bbox": [cx - bx[2] / 2, cy - bx[3], bx[2], bx[3]],
                                    })
                            if len(registered) > best_count:
                                best_count = len(registered)
                                best_detections = registered
                                best_total_detected = raw_count
                                bd: dict[str, int] = {}
                                for d in registered:
                                    bd[d["label"]] = bd.get(d["label"], 0) + 1
                                best_breakdown = bd
                            if (len(registered) > emergency_best_count
                                    and any(d["label"] == "Ambulance" for d in registered)):
                                emergency_best_count = len(registered)
                                emergency_best_frame_idx = reg_idx
                                emergency_best_detections = registered
                    reg_idx += 1
            finally:
                cap_reg.release()
            if best_detections:
                print(f"✅ Scene-registration fallback recovered {best_count} ROI vehicle(s) "
                      f"(best background alignment: {reg_best_inliers} RANSAC inliers).")
            else:
                print("⛔ Scene-registration fallback still found no vehicle inside the "
                      "calibrated ROI — reporting calibration_required.")

    traffic = get_traffic_level(best_count)

    # Queue order (closest to line_b = position 1), reusing the SAME
    # distance-to-line geometry track_crossings.py uses - never a blind
    # ascending/descending pixel guess. best_detections is already
    # ROI-filtered above when roi_usable, so every ordered position here
    # came from inside the calibrated ROI.
    ordered_vehicle_types: Optional[list] = None
    if roi_usable:
        if best_detections:
            ranked = sorted(
                best_detections,
                key=lambda det: distance_point_to_line(
                    (det["bbox"][0] + det["bbox"][2] / 2, det["bbox"][1] + det["bbox"][3]),
                    line_b[0], line_b[1],
                ),
            )
            ordered_vehicle_types = [det["label"] for det in ranked]
        else:
            queue_ordering_blocker = "no vehicle detections inside the calibrated ROI in any sampled frame"

    new_queue_result = build_queue_result(ordered_vehicle_types) if ordered_vehicle_types else {
        "available": False, "blocker": queue_ordering_blocker,
    }

    # ── Emergency candidate frame (parallel to, and independent from, the
    # normal densest-frame path above - normal outputs below are entirely
    # unaffected by whatever this block finds). Reuses the SAME line_b/
    # distance-to-line ordering and the SAME build_queue_result() (P1..P5+
    # grouping, representative lengths, 1.177 m gap, calibrated discharge
    # speeds, FIFO, and its own existing compute_emergency_priority()) - just
    # applied to the emergency-candidate frame's own ROI-filtered detections
    # instead of the normal densest frame's. emergency_detected is TRUE only
    # when a real ROI-filtered Ambulance detection existed in some sampled
    # frame AND that frame's own queue-position calculation succeeds.
    emergency_detected = False
    emergency_fields: dict = {}
    if emergency_best_detections:
        emergency_ranked = sorted(
            emergency_best_detections,
            key=lambda det: distance_point_to_line(
                (det["bbox"][0] + det["bbox"][2] / 2, det["bbox"][1] + det["bbox"][3]),
                line_b[0], line_b[1],
            ),
        )
        emergency_ordered_vehicle_types = [det["label"] for det in emergency_ranked]
        emergency_queue_result = build_queue_result(emergency_ordered_vehicle_types)
        if emergency_queue_result["available"] and emergency_queue_result["emergency_detected"]:
            emergency_detected = True
            emergency_fields = {
                "emergency_vehicle_type": emergency_queue_result["emergency_vehicle_type"],
                "emergency_queue_position": emergency_queue_result["emergency_queue_position"],
                "emergency_position_group": emergency_queue_result["emergency_position_group"],
                "emergency_clearance_time_s": emergency_queue_result["emergency_clearance_time_s"],
                "emergency_priority_green_s": emergency_queue_result["emergency_priority_green_s"],
                "emergency_priority": emergency_queue_result["emergency_priority"],
                "emergency_recommendation": emergency_queue_result["emergency_recommendation"],
            }
            print(f"🚨 Emergency candidate frame {emergency_best_frame_idx}: {emergency_best_count} ROI vehicles "
                  f"(Ambulance present) — emergency_clearance={emergency_fields['emergency_clearance_time_s']}s, "
                  f"emergency_priority_green={emergency_fields['emergency_priority_green_s']}s")
        else:
            print(f"⚠️  Emergency candidate frame {emergency_best_frame_idx} had an ROI Ambulance, but its own "
                  f"queue-position calculation was unavailable ({emergency_queue_result.get('blocker')}) - "
                  f"reporting emergency_detected=False.")

    if not new_queue_result["available"]:
        # No queue-position result available - the old count-based calculation
        # is no longer a valid substitute (superseded final research method),
        # so this is reported as a clear, non-fabricated calibration_required
        # error rather than inventing a green-time figure. Nothing is persisted
        # to MongoDB for an unsupported/unsupportable video.
        print(f"⛔ Queue-position calculation unavailable ({new_queue_result['blocker']}) - "
              f"no legacy fallback; reporting calibration_required.")
        raise HTTPException(
            status_code=422,
            detail=(
                "calibration_required: This video is not registered with the current TrafficEye "
                "fixed-camera calibration. Queue-position green-time analysis requires a compatible "
                f"ROI/reference-line calibration. (blocker: {new_queue_result['blocker']})"
            ),
        )

    queue_length_m = new_queue_result["ordered_queue"][-1]["cumulative_distance_m"]
    clearance_time_s = new_queue_result["queue_clearance_time_s"]
    suggested_green_s = new_queue_result["suggested_green_time_s"]
    print(f"📊 Queue-position calculation complete — {len(ordered_vehicle_types)}-vehicle ordered queue, "
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
        total_detected_vehicles=best_total_detected,
        vehicle_breakdown=best_breakdown,
        traffic_level=traffic["level"],
        queue_length_m=queue_length_m,
        clearance_time_s=clearance_time_s,
        suggested_green_s=suggested_green_s,
        emergency_detected=emergency_detected,
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
    # new_queue_result["available"] is guaranteed True here - the unavailable
    # case already returned via the calibration_required HTTPException above.
    saved["queue_calculation_method"] = "queue_position_based"
    saved["ordered_queue"] = new_queue_result["ordered_queue"]
    saved["average_gap_m"] = new_queue_result["average_gap_m"]
    saved["safety_buffer_s"] = new_queue_result["safety_buffer_s"]
    # emergency_detected/emergency_* come from the separate emergency-
    # candidate-frame calculation above, NOT from new_queue_result (whose own
    # emergency fields reflect only the normal densest frame) - an Ambulance
    # seen in a different sampled frame than the normal densest one must
    # still be reported.
    saved["emergency_detected"] = emergency_detected
    if emergency_detected:
        saved.update(emergency_fields)

    # ── Visualization-only annotated video — NON-BLOCKING ─────────────────────
    # The real research result above is finished and already saved to MongoDB.
    # Rendering the annotated preview MP4 (fresh per-sampled-frame YOLO + ffmpeg
    # encode) is the slow part, so it is NOT awaited here: the output filename /
    # URL is reserved now and generate_annotated_recording() runs in a
    # background task AFTER this response is sent (see _render_annotated_video_bg).
    # The research calculation is not re-run, no second record is saved, and
    # generate_annotated_recording() + the hybrid display-label helper are
    # unchanged. The frontend keeps the raw upload visible and polls
    # GET /api/analysis-video-ready/{filename} until the finished file appears,
    # then swaps it in. Response-only, same as ordered_queue/emergency_* above -
    # never persisted to MongoDB.
    annotated_filename = f"{uuid.uuid4().hex}.mp4"
    annotated_path = PROCESSED_DIR / annotated_filename
    # emergency_detected is passed ONLY as a display hint for the annotated
    # overlay (lets a trimmed copy of the ambulance footage get the same
    # visualization-only "Ambulance" relabel the exact original file gets).
    # It never changes a box, a count, or any research value.
    #
    # Record a '<uuid>.mp4.job' marker BEFORE scheduling the task so that if an
    # `uvicorn --reload` restart kills the background render mid-encode, the
    # next startup can resume it (or fail it cleanly) instead of the frontend
    # polling a file that will never appear. Visualization plumbing only.
    _write_render_job(annotated_path, dest, emergency_detected)
    background_tasks.add_task(_render_annotated_video_bg, dest, annotated_path, emergency_detected)
    saved["annotated_video_url"] = f"/media/{annotated_filename}"
    saved["annotated_video_pending"] = True
    print(f"🎬 Annotated video: deferred to background task — /media/{annotated_filename}")

    return saved

@app.get("/api/analysis-video-ready/{filename}")
async def analysis_video_ready(filename: str):
    """
    Read-only readiness probe for the background-rendered annotated video.
    `ready` is True only when the reserved MP4 exists in PROCESSED_DIR with a
    non-zero size (the background worker renames it into place atomically on
    success). `failed` is True when a '<name>.failed' marker exists - the
    render gave up (unreadable source, ffmpeg failure, or an interrupted
    render that could not be resumed) - so the frontend can stop polling and
    show the raw upload immediately instead of waiting out its poll ceiling.
    No DB access, no schema change, no auth/session state touched.
    """
    safe_name = Path(filename).name
    if safe_name != filename or not safe_name.endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Invalid annotated-video filename.")
    target = PROCESSED_DIR / safe_name
    ready = target.is_file() and target.stat().st_size > 0
    failed = (PROCESSED_DIR / (safe_name + ".failed")).is_file() and not ready
    return {"filename": safe_name, "ready": ready, "failed": failed}

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

    # ── Emergency-detection reporting (READ ONLY) ─────────────────────────────
    # Dashboard reporting of the emergency_detected value the analysis pipeline
    # already persists per record (see create_analysis / AnalysisRecord). This
    # does NOT recompute emergency detection, create a collection, or change
    # any record - it only counts/reads what is already saved.
    #
    # {"emergency_detected": True} matches ONLY documents whose stored value is
    # the boolean true. Legacy records saved before this field existed (key
    # absent) and any record with an explicit null are therefore not counted,
    # exactly as required.
    emergency_detection_count = await analysis_collection.count_documents(
        {"emergency_detected": True}
    )
    latest_emergency = await analysis_collection.find_one(
        {"emergency_detected": True}, sort=[("created_at", -1)]
    )
    latest_emergency_date = latest_emergency.get("recorded_date") if latest_emergency else None
    latest_emergency_time = latest_emergency.get("recorded_time") if latest_emergency else None

    return {
        "total_locations": total_locations,
        "total_cases": total_cases,
        "high_traffic_count": high_traffic_count,
        "avg_clearance_time": avg_clearance_time,
        "emergency_detection_count": emergency_detection_count,
        "latest_emergency_date": latest_emergency_date,
        "latest_emergency_time": latest_emergency_time,
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
