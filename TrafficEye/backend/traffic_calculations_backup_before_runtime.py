"""
Traffic decision-support calculations for TrafficEye.

These functions turn a raw vehicle-type breakdown (from YOLO detection) into
physical queue length and traffic-signal timing recommendations for a
traffic officer.
"""

import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
AVG_PASSING_TIME_PATH = OUTPUT_DIR / "average_passing_time.csv"
BUFFER_CALIBRATION_PATH = OUTPUT_DIR / "buffer_calibration.csv"

DEFAULT_PASSING_TIME_S = 3.0   # fallback clearance time for a vehicle type with no observed data
DEFAULT_SAFETY_BUFFER_S = 3.0  # fallback safety buffer if buffer_calibration.csv is missing
START_DELAY_S = 5.0            # driver reaction time before the queue starts moving after green

# Reference-based representative vehicle lengths in meters.
# Used for queue-length estimation only.
# Final green-time clearance uses observed vehicle-type passing times from crossing logs.
# These are average category lengths, not exact per-vehicle measurements.
VEHICLE_LENGTHS_M = {
    "Motorcycle": 2.0,
    "Bicycle": 1.9,
    "Tuktuk": 2.5,
    "Car": 4.3,
    "Bus": 10.8,
    "Truck": 8.5,
    "Ambulance": 5.5,
}


def calculate_queue_length(vehicle_breakdown: dict) -> float:
    """
    Queue length (m) = Σ (count_per_type × avg_length_m_per_type)

    Approximates the physical length of the vehicle queue by summing,
    for each detected vehicle type, the number of vehicles times that
    type's average real-world length (see VEHICLE_LENGTHS_M).
    Vehicle types not found in VEHICLE_LENGTHS_M contribute 0.
    """
    total = 0.0
    for vehicle_type, count in vehicle_breakdown.items():
        total += count * VEHICLE_LENGTHS_M.get(vehicle_type, 0)
    return round(total, 1)


def load_observed_passing_times() -> dict:
    """
    Reads outputs/average_passing_time.csv (produced by calibration_analyzer.py
    from the crossing logs) and returns {vehicle_type: seconds}. Observed
    passing times are for clearance time - not to be confused with
    VEHICLE_LENGTHS_M, which is for queue-length estimation only.

    Picks the most robust statistic available per vehicle type, in priority
    order: trimmed_mean_gap_s > median_gap_s > avg_passing_time_s (the raw
    mean is more sensitive to remaining outliers than the other two - see
    calibration_analyzer.py's own recommendation).
    """
    times = {}
    if not AVG_PASSING_TIME_PATH.exists():
        return times
    with open(AVG_PASSING_TIME_PATH, newline="") as f:
        for row in csv.DictReader(f):
            vehicle_type = row["vehicle_type"]
            for column in ("trimmed_mean_gap_s", "median_gap_s", "avg_passing_time_s"):
                value = row.get(column)
                if value not in (None, ""):
                    times[vehicle_type] = float(value)
                    break
    return times


def load_safety_buffer_s() -> float:
    """
    Reads final_safety_buffer_s from outputs/buffer_calibration.csv (produced
    by calibration_analyzer.py). That figure is calculated numerically from
    observed inter-vehicle crossing gaps - P85 gap minus median gap, clamped
    to a sane range - it is not manually guessed from
    outputs/buffer_gap_scatterplot.png, which is visual evidence only.
    Falls back to DEFAULT_SAFETY_BUFFER_S if the calibration file is missing.
    """
    if not BUFFER_CALIBRATION_PATH.exists():
        return DEFAULT_SAFETY_BUFFER_S
    with open(BUFFER_CALIBRATION_PATH, newline="") as f:
        row = next(csv.DictReader(f))
    return float(row["final_safety_buffer_s"])


def calculate_clearance_time(vehicle_breakdown: dict, passing_times: dict | None = None) -> float:
    """
    Clearance time (s) = Σ (count_per_type × observed_passing_time_s_per_type)

    Estimates how long it takes for every queued vehicle to pass the stop
    line once the signal turns green, using each vehicle type's OWN observed
    passing time (see load_observed_passing_times()) rather than one fixed
    average applied to the total vehicle count. A vehicle type with no
    observed data falls back to DEFAULT_PASSING_TIME_S.
    """
    if passing_times is None:
        passing_times = load_observed_passing_times()
    total = 0.0
    for vehicle_type, count in vehicle_breakdown.items():
        total += count * passing_times.get(vehicle_type, DEFAULT_PASSING_TIME_S)
    return round(total, 1)


def calculate_suggested_green_time(
    clearance_time_s: float,
    start_delay_s: float = START_DELAY_S,
    safety_buffer_s: float | None = None,
) -> float:
    """
    Suggested green time (s) = START_DELAY_S + Clearance Time + SAFETY_BUFFER_S

    start_delay_s accounts for driver reaction time before the queue starts
    moving after the signal turns green. safety_buffer_s defaults to the
    calibrated value loaded from outputs/buffer_calibration.csv via
    load_safety_buffer_s() when not explicitly provided, and adds extra
    margin so the last vehicle in the queue fully clears the intersection
    before the signal changes again.
    """
    if safety_buffer_s is None:
        safety_buffer_s = load_safety_buffer_s()
    return round(start_delay_s + clearance_time_s + safety_buffer_s, 1)
