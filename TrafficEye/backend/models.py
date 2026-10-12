"""
Pydantic models for TrafficEye analysis records stored in MongoDB.
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class AnalysisRecord(BaseModel):
    # Optional: the New Analysis UI no longer collects these (Stitch design
    # only asks for the video + date/time), and the old required-string
    # contract would reject a real submission with no location metadata.
    # None means "not provided" - never a fabricated placeholder like
    # "Unknown Junction" or 0.0 GPS. Older requests that still send real
    # values continue to work exactly as before.
    location_name: Optional[str] = None
    road: Optional[str] = None
    direction: Optional[str] = None
    gps_lat: Optional[float] = None
    gps_lng: Optional[float] = None
    recorded_date: str
    recorded_time: str
    notes: Optional[str] = None

    video_filename: str
    vehicle_count: int
    # Raw YOLO vehicle-class detections in the same selected/densest frame,
    # before the ROI queue filter. Display/stat only - never used in queue
    # length, clearance time, suggested green, or traffic level. Optional so
    # records saved before this field existed still deserialize/return fine
    # (the key is simply absent for them; the frontend must not fabricate it).
    total_detected_vehicles: Optional[int] = None
    vehicle_breakdown: dict[str, int]
    traffic_level: str

    queue_length_m: float
    clearance_time_s: float
    suggested_green_s: float

    # The real emergency_detected value already computed by the emergency-
    # candidate-frame path in main.py - persisted so History can show a real
    # DETECTED/NONE status instead of always "not available". Optional so
    # records saved before this field existed still deserialize/return fine
    # (the key is simply absent for them; the frontend must not guess it).
    emergency_detected: Optional[bool] = None

    status: str
    created_at: datetime


class Officer(BaseModel):
    username: str
    password_hash: str
    full_name: str
    badge_number: Optional[str] = None
    created_at: datetime


class OfficerRegister(BaseModel):
    username: str
    password: str
    full_name: str
    badge_number: Optional[str] = None
