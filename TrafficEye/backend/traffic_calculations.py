"""
Traffic decision-support calculations for TrafficEye.

Queue-position based green-time calculation (final research methodology).

Single-lane / single-stream ordered queue -> cumulative distance -> lookup
speed -> raw clear time -> FIFO-adjusted clear time -> queue clearance ->
+ operational margin -> Suggested Green. Speed/length/gap/buffer values are
read directly from outputs/final_green_time_lookup_table.csv (Part 5's
output) - nothing here hardcodes 1.177/3.12 or any lookup value; changing
the calibration only requires regenerating that CSV, not this code.

The earlier count-based calculation (calculate_queue_length/
calculate_clearance_time/calculate_suggested_green_time, reading
average_passing_time.csv/buffer_calibration.csv) has been removed - it is
superseded by this method and is no longer used anywhere as a runtime
fallback. See traffic_calculations_backup_before_legacy_removal.py for the
prior version if historical reference is ever needed.
"""

import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

# Representative vehicle lengths in meters - used by the cumulative-distance
# calculation below (compute_ordered_queue_details).
VEHICLE_LENGTHS_M = {
    "Motorcycle": 2.0,
    "Bicycle": 1.9,
    "Tuktuk": 2.5,
    "Car": 4.3,
    "Bus": 10.8,
    "Truck": 8.5,
    "Ambulance": 5.5,
}

FINAL_LOOKUP_PATH = OUTPUT_DIR / "final_green_time_lookup_table.csv"


def position_group_for_queue_position(position: int) -> str:
    """1->P1, 2->P2, 3 or 4->P3-4, >=5->P5+ - same mapping used throughout calibration."""
    if position == 1:
        return "P1"
    if position == 2:
        return "P2"
    if position in (3, 4):
        return "P3-4"
    return "P5+"


def load_final_lookup_table() -> dict:
    """
    {(vehicle_type, position_group): {"recommended_speed_mps", "speed_source",
     "average_inter_vehicle_gap_m", "safety_buffer_s"}}, read directly from
    outputs/final_green_time_lookup_table.csv. Returns {} if the file is
    missing - callers must treat that as "calculation unavailable," never
    fabricate a value.
    """
    if not FINAL_LOOKUP_PATH.exists():
        return {}
    table = {}
    with open(FINAL_LOOKUP_PATH, newline="") as f:
        for row in csv.DictReader(f):
            table[(row["vehicle_type"], row["position_group"])] = {
                "recommended_speed_mps": float(row["recommended_speed_mps"]),
                "speed_source": row["speed_source"],
                "average_inter_vehicle_gap_m": float(row["average_inter_vehicle_gap_m"]),
                "safety_buffer_s": float(row["safety_buffer_s"]),
            }
    return table


def compute_ordered_queue_details(ordered_vehicle_types: list[str], lookup: dict, gap_m: float):
    """
    ordered_vehicle_types: vehicle_type labels in queue order, position 1
    first (closest to the reference line). lookup/gap_m come from
    load_final_lookup_table() (and one of its rows) - passed in rather than
    reloaded, so the whole queue uses one consistent snapshot of the table.

    For each vehicle i: D_i = cumulative representative length + (i-1)*gap_m;
    T_raw_i = D_i / V_i; T_fifo_i = T_raw_i for i=1, else max(T_raw_i, T_fifo_(i-1)).

    Returns (queue_details, errors). queue_details is [] if any vehicle can't
    be resolved (unknown type, missing lookup entry, non-positive speed) -
    errors explains why. Never invents a length or speed.
    """
    queue_details = []
    errors = []
    cumulative_length = 0.0
    prev_fifo_time = None

    for i, vehicle_type in enumerate(ordered_vehicle_types, start=1):
        position_group = position_group_for_queue_position(i)

        vehicle_length = VEHICLE_LENGTHS_M.get(vehicle_type)
        if vehicle_length is None:
            errors.append(f"position {i}: no representative length for vehicle_type '{vehicle_type}'")
            return [], errors

        cumulative_length += vehicle_length
        distance_m = round(cumulative_length + (i - 1) * gap_m, 4)

        entry = lookup.get((vehicle_type, position_group))
        if entry is None:
            errors.append(f"position {i} ({vehicle_type}/{position_group}): no entry in final_green_time_lookup_table.csv")
            return [], errors

        speed = entry["recommended_speed_mps"]
        if not (speed == speed) or speed <= 0:  # speed == speed is False only for NaN
            errors.append(f"position {i} ({vehicle_type}/{position_group}): recommended_speed_mps invalid ({speed})")
            return [], errors

        raw_time = round(distance_m / speed, 4)
        fifo_time = raw_time if prev_fifo_time is None else round(max(raw_time, prev_fifo_time), 4)
        prev_fifo_time = fifo_time

        queue_details.append({
            "vehicle_type": vehicle_type,
            "queue_position": i,
            "position_group": position_group,
            "vehicle_length_m": vehicle_length,
            "cumulative_distance_m": distance_m,
            "recommended_speed_mps": speed,
            "speed_source": entry["speed_source"],
            "raw_clear_time_s": raw_time,
            "fifo_adjusted_clear_time_s": fifo_time,
        })

    return queue_details, errors


def compute_queue_clearance_and_green(queue_details: list, safety_buffer_s: float) -> dict:
    """queue_clearance_time_s = max(fifo_adjusted_clear_time_s); suggested_green_time_s
    = queue_clearance_time_s + safety_buffer_s. No extra delay/margin is added."""
    fifo_times = [d["fifo_adjusted_clear_time_s"] for d in queue_details]
    queue_clearance_time_s = round(max(fifo_times), 4)
    suggested_green_time_s = round(queue_clearance_time_s + safety_buffer_s, 2)
    return {
        "queue_clearance_time_s": queue_clearance_time_s,
        "suggested_green_time_s": suggested_green_time_s,
    }


def compute_emergency_priority(queue_details: list, safety_buffer_s: float) -> dict:
    """If an Ambulance is present, uses its FIFO-adjusted clear time (the same
    normal calculation as every other vehicle) to derive an additional
    priority advisory. Does not alter or replace the normal queue result.
    Picks the frontmost Ambulance if more than one is present."""
    for d in queue_details:
        if d["vehicle_type"] == "Ambulance":
            clearance = d["fifo_adjusted_clear_time_s"]
            return {
                "emergency_detected": True,
                "emergency_vehicle_type": "Ambulance",
                "emergency_queue_position": d["queue_position"],
                "emergency_position_group": d["position_group"],
                "emergency_clearance_time_s": clearance,
                "emergency_priority_green_s": round(clearance + safety_buffer_s, 2),
                "emergency_priority": "HIGH",
                "emergency_recommendation": "Prioritize this approach until the emergency vehicle clears.",
            }
    return {"emergency_detected": False}


def build_queue_result(ordered_vehicle_types: list[str]) -> dict:
    """
    Top-level entry point: given vehicle_type labels in spatial queue order
    (position 1 = closest to the reference line), returns the full
    queue-position-based result, or {"available": False, "blocker": "..."}
    if it cannot be computed - callers must treat that as "no result
    available" (e.g. report calibration_required), never fabricate a value.
    """
    if not ordered_vehicle_types:
        return {"available": False, "blocker": "no ordered queue detections available"}

    lookup = load_final_lookup_table()
    if not lookup:
        return {"available": False, "blocker": "final_green_time_lookup_table.csv not found or empty"}

    gap_m = next(iter(lookup.values()))["average_inter_vehicle_gap_m"]
    safety_buffer_s = next(iter(lookup.values()))["safety_buffer_s"]

    queue_details, errors = compute_ordered_queue_details(ordered_vehicle_types, lookup, gap_m)
    if errors:
        return {"available": False, "blocker": "; ".join(errors)}

    result = {
        "available": True,
        "ordered_queue": queue_details,
        "average_gap_m": gap_m,
        "safety_buffer_s": safety_buffer_s,
    }
    result.update(compute_queue_clearance_and_green(queue_details, safety_buffer_s))
    result.update(compute_emergency_priority(queue_details, safety_buffer_s))
    return result
