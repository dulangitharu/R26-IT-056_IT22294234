"""
TrafficEye - Calibration event dataset builder.

Joins crossing_log_clip_*.csv (line_b crossing events) with
queue_position_log_clip_*.csv (per-frame spatial queue-position snapshots) to
build a per-vehicle calibration dataset that ties together vehicle type,
vehicle length, spatial queue position, inter-vehicle gap, and effective
queue discharge speed - the ingredients for the queue-position-based
green-time model.

Standalone script - does not import or modify main.py, traffic_calculations.py,
calibration_analyzer.py, or track_crossings.py. Does not touch ROI/line
calibration or model weights.

Run from the backend/ directory:
    python build_calibration_dataset.py
"""

import csv
import statistics
from collections import Counter, defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

EVENT_DATASET_PATH = OUTPUT_DIR / "calibration_event_dataset.csv"
SPEED_BY_TYPE_POSITION_PATH = OUTPUT_DIR / "calibrated_effective_speed_by_type_position.csv"
SUMMARY_PATH = OUTPUT_DIR / "calibration_dataset_summary.csv"

VEHICLE_LENGTHS_M = {
    "Motorcycle": 2.0,
    "Bicycle": 1.9,
    "Tuktuk": 2.5,
    "Car": 4.3,
    "Bus": 10.8,
    "Truck": 8.5,
    "Ambulance": 5.5,
}
AVERAGE_VEHICLE_LENGTH_M = sum(VEHICLE_LENGTHS_M.values()) / len(VEHICLE_LENGTHS_M)

AVERAGE_GAP_M = 1.177

POSITION_GROUP_ORDER = ["P1", "P2", "P3-4", "P5+"]

EVENT_FIELDNAMES = [
    "video_name", "vehicle_id", "vehicle_type", "first_queue_time_s", "first_position",
    "first_position_group", "last_pre_cross_time_s", "last_pre_cross_position", "last_position_group",
    "pass_time_seconds", "observed_clear_time_s", "vehicle_length_m", "average_gap_m",
    "distance_to_clear_m", "effective_discharge_speed_mps", "sample_quality",
]


def position_group_for(position):
    if position == 1:
        return "P1"
    if position == 2:
        return "P2"
    if position in (3, 4):
        return "P3-4"
    return "P5+"


def _position_group_sort_key(group):
    try:
        return POSITION_GROUP_ORDER.index(group)
    except ValueError:
        return len(POSITION_GROUP_ORDER)


def load_crossing_logs():
    """clip_key (from filename, e.g. 'clip_09') -> list of crossing rows."""
    per_clip = {}
    for path in sorted(OUTPUT_DIR.glob("crossing_log_clip_*.csv")):
        clip_key = path.stem.replace("crossing_log_", "")
        rows = []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    rows.append({
                        "vehicle_id": row["vehicle_id"],
                        "vehicle_type": row["vehicle_type"],
                        "pass_time_seconds": float(row["pass_time_seconds"]),
                    })
                except (KeyError, ValueError, TypeError):
                    continue
        if rows:
            per_clip[clip_key] = rows
    return per_clip


def load_queue_position_logs():
    """
    clip_key -> {
        "rows_by_vehicle": {vehicle_id: [rows sorted by time_s]},
        "rows_by_frame": {frame_number: [rows]},
        "video_name": the video_name field value seen in the file,
        "max_position": max spatial_queue_position seen in this clip,
    }
    """
    per_clip = {}
    for path in sorted(OUTPUT_DIR.glob("queue_position_log_clip_*.csv")):
        clip_key = path.stem.replace("queue_position_log_", "")
        rows_by_vehicle = defaultdict(list)
        rows_by_frame = defaultdict(list)
        video_name = None
        max_position = 0
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    parsed = {
                        "vehicle_id": row["vehicle_id"],
                        "vehicle_type": row["vehicle_type"],
                        "frame_number": int(row["frame_number"]),
                        "time_s": float(row["time_s"]),
                        "position": int(row["spatial_queue_position"]),
                        "position_group": row["position_group"],
                        "distance_to_line_b_px": float(row["distance_to_line_b_px"]),
                    }
                except (KeyError, ValueError, TypeError):
                    continue
                video_name = video_name or row.get("video_name")
                max_position = max(max_position, parsed["position"])
                rows_by_vehicle[parsed["vehicle_id"]].append(parsed)
                rows_by_frame[parsed["frame_number"]].append(parsed)

        for vid in rows_by_vehicle:
            rows_by_vehicle[vid].sort(key=lambda r: r["time_s"])

        if rows_by_vehicle:
            per_clip[clip_key] = {
                "rows_by_vehicle": dict(rows_by_vehicle),
                "rows_by_frame": dict(rows_by_frame),
                "video_name": video_name or clip_key,
                "max_position": max_position,
            }
    return per_clip


def context_average_length(frame_rows):
    """Average estimated vehicle length among all vehicles detected in this
    queue-position frame snapshot - used as a fallback when a front vehicle's
    exact type/length isn't available."""
    lengths = [VEHICLE_LENGTHS_M.get(r["vehicle_type"], AVERAGE_VEHICLE_LENGTH_M) for r in frame_rows]
    if not lengths:
        return AVERAGE_VEHICLE_LENGTH_M
    return sum(lengths) / len(lengths)


def compute_distance_to_clear(own_position, own_vehicle_type, frame_rows):
    """
    distance_to_clear_m = cumulative estimated vehicle lengths for positions
    1..own_position (inclusive) + (own_position - 1) * AVERAGE_GAP_M.

    Uses the actual detected front-vehicle types from the same queue-position
    frame snapshot when available; falls back to the average vehicle length
    observed in that same frame's queue context for any front rank whose
    detected row/type is missing.

    Returns (distance_to_clear_m, quality): quality is "EXACT" when every
    position's length came from an actually-detected vehicle in that frame,
    else "ESTIMATED".
    """
    own_length = VEHICLE_LENGTHS_M.get(own_vehicle_type, AVERAGE_VEHICLE_LENGTH_M)
    if own_position <= 1:
        return round(own_length, 3), "EXACT"

    front_by_position = {r["position"]: r for r in frame_rows if r["position"] < own_position}
    context_avg_length = context_average_length(frame_rows)

    total_length = own_length
    quality = "EXACT"
    for pos in range(1, own_position):
        row = front_by_position.get(pos)
        if row is not None and row["vehicle_type"] in VEHICLE_LENGTHS_M:
            total_length += VEHICLE_LENGTHS_M[row["vehicle_type"]]
        else:
            total_length += context_avg_length
            quality = "ESTIMATED"

    total_gap = (own_position - 1) * AVERAGE_GAP_M
    return round(total_length + total_gap, 3), quality


def build_event_dataset(crossing_logs, queue_logs):
    """
    For each crossed vehicle, matches it against its clip's queue-position
    log by vehicle_id, keeps only pre-crossing snapshots (time_s < pass_time_seconds),
    and derives observed_clear_time_s / distance_to_clear_m / effective_discharge_speed_mps
    from the FIRST such snapshot. Vehicles with no pre-crossing snapshot are
    reported as unmatched, not fabricated.
    """
    events = []
    total_crossed = 0
    matched = 0
    unmatched = []

    for clip_key, crossings in crossing_logs.items():
        clip_queue = queue_logs.get(clip_key)
        for c in crossings:
            total_crossed += 1
            vid = c["vehicle_id"]
            pass_time = c["pass_time_seconds"]

            if clip_queue is None:
                unmatched.append((clip_key, vid, c["vehicle_type"]))
                continue

            vehicle_rows = clip_queue["rows_by_vehicle"].get(vid, [])
            pre_cross = [r for r in vehicle_rows if r["time_s"] < pass_time]
            if not pre_cross:
                unmatched.append((clip_key, vid, c["vehicle_type"]))
                continue

            matched += 1
            first_row = pre_cross[0]
            last_row = pre_cross[-1]

            observed_clear_time_s = round(pass_time - first_row["time_s"], 3)
            if observed_clear_time_s <= 0:
                continue

            vehicle_type = c["vehicle_type"]
            vehicle_length_m = round(VEHICLE_LENGTHS_M.get(vehicle_type, AVERAGE_VEHICLE_LENGTH_M), 3)

            frame_rows = clip_queue["rows_by_frame"].get(first_row["frame_number"], [])
            distance_to_clear_m, quality = compute_distance_to_clear(
                first_row["position"], vehicle_type, frame_rows
            )

            effective_discharge_speed_mps = round(distance_to_clear_m / observed_clear_time_s, 4)

            events.append({
                "video_name": clip_queue["video_name"],
                "vehicle_id": vid,
                "vehicle_type": vehicle_type,
                "first_queue_time_s": first_row["time_s"],
                "first_position": first_row["position"],
                "first_position_group": first_row["position_group"],
                "last_pre_cross_time_s": last_row["time_s"],
                "last_pre_cross_position": last_row["position"],
                "last_position_group": last_row["position_group"],
                "pass_time_seconds": pass_time,
                "observed_clear_time_s": observed_clear_time_s,
                "vehicle_length_m": vehicle_length_m,
                "average_gap_m": AVERAGE_GAP_M,
                "distance_to_clear_m": distance_to_clear_m,
                "effective_discharge_speed_mps": effective_discharge_speed_mps,
                "sample_quality": quality,
            })

    return events, total_crossed, matched, unmatched


def write_event_dataset(events, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EVENT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(events)


def sample_quality_for_group(count):
    if count >= 20:
        return "GOOD"
    if count >= 10:
        return "ACCEPTABLE"
    if count >= 3:
        return "LOW"
    return "LIMITED"


def write_speed_by_type_position(events, path):
    groups = defaultdict(list)
    for e in events:
        groups[(e["vehicle_type"], e["first_position_group"])].append(e["effective_discharge_speed_mps"])

    rows = []
    for (vtype, pgroup), speeds in sorted(
        groups.items(), key=lambda kv: (kv[0][0], _position_group_sort_key(kv[0][1]))
    ):
        count = len(speeds)
        rows.append({
            "vehicle_type": vtype,
            "first_position_group": pgroup,
            "sample_count": count,
            "avg_effective_speed_mps": round(statistics.mean(speeds), 3),
            "median_effective_speed_mps": round(statistics.median(speeds), 3),
            "min_effective_speed_mps": round(min(speeds), 3),
            "max_effective_speed_mps": round(max(speeds), 3),
            "sample_quality": sample_quality_for_group(count),
        })

    fieldnames = [
        "vehicle_type", "first_position_group", "sample_count", "avg_effective_speed_mps",
        "median_effective_speed_mps", "min_effective_speed_mps", "max_effective_speed_mps", "sample_quality",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows, fieldnames


def write_summary(events, total_crossed, matched, unmatched, queue_logs, path):
    """
    Two-column metric/value CSV (the underlying breakdowns don't share one
    tabular row shape, so this is a flat key-value log rather than a fixed
    grid). "event_distance_quality_count" is the EXACT/ESTIMATED split from
    calibration_event_dataset.csv's own sample_quality column (i.e. how many
    rows had their full front-vehicle sequence actually detected vs. had to
    fall back to an average length) - distinct from the GOOD/ACCEPTABLE/LOW/
    LIMITED sample_quality in calibrated_effective_speed_by_type_position.csv,
    which grades aggregated group sample sizes instead.
    """
    rows = []

    rows.append(("total_crossed_vehicles", total_crossed))
    rows.append(("matched_vehicles", matched))
    rows.append(("unmatched_vehicles", len(unmatched)))
    rows.append(("total_valid_calibration_rows", len(events)))

    type_counts = Counter(e["vehicle_type"] for e in events)
    for vtype in sorted(type_counts):
        rows.append((f"vehicle_type_count:{vtype}", type_counts[vtype]))

    group_counts = Counter(e["first_position_group"] for e in events)
    for g in sorted(group_counts, key=_position_group_sort_key):
        rows.append((f"first_position_group_count:{g}", group_counts[g]))

    type_group_counts = Counter((e["vehicle_type"], e["first_position_group"]) for e in events)
    for (vtype, g) in sorted(type_group_counts, key=lambda kv: (kv[0], _position_group_sort_key(kv[1]))):
        rows.append((f"type_position_count:{vtype}|{g}", type_group_counts[(vtype, g)]))

    quality_counts = Counter(e["sample_quality"] for e in events)
    for q in sorted(quality_counts):
        rows.append((f"event_distance_quality_count:{q}", quality_counts[q]))

    for clip_key in sorted(queue_logs):
        rows.append((f"max_spatial_queue_position:{clip_key}", queue_logs[clip_key]["max_position"]))

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)

    return rows


def print_table(title, rows, fieldnames, first_col_width=22, col_width=15):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    if not rows:
        print("  (no rows)")
        return
    print(" ".join(f"{h:>{first_col_width}s}" if i == 0 else f"{h:>{col_width}s}" for i, h in enumerate(fieldnames)))
    for row in rows:
        print(" ".join(
            f"{row[h]:>{first_col_width}}" if i == 0 else f"{row[h]:>{col_width}}"
            for i, h in enumerate(fieldnames)
        ))


# ── EXPERIMENTAL: common discharge-start (t0) detection - DIAGNOSTIC ONLY ──
# Not wired into build_event_dataset() or any output file yet. Used only by
# diagnose_discharge_start.py (dry-run, terminal output only - writes nothing).
#
# Thresholds below are heuristics chosen by inspecting actual per-vehicle
# distance_to_line_b_px sequences in the existing queue_position_log_clip_*.csv
# files (see diagnose_discharge_start.py's printed evidence for the specific
# examples that motivated each value) - they are NOT traffic-engineering
# standards, and are intentionally kept as plain module-level constants so
# they're easy to find and adjust.
T0_STABLE_JITTER_PX = 3.0        # per-step |distance delta| at/under this = "still stationary" (bbox jitter)
T0_STABLE_MIN_STEPS = 2          # consecutive stable steps required before trusting "it was stopped"
T0_MOVEMENT_MIN_DROP_PX = 5.0    # per-step distance decrease beyond this = "really moving," not jitter
T0_MOVEMENT_MIN_STEPS = 2        # consecutive moving steps required before trusting "discharge started"
T0_EMPTY_QUEUE_GAP_S = 2.0       # a gap this large between consecutive logged snapshots (any vehicle,
                                  # any position) suggests the queue was fully empty throughout - a
                                  # possible boundary between two separate discharge events


def find_first_p1_track(rows_by_frame):
    """First vehicle_id ever seen holding spatial_queue_position==1 in this
    clip's queue_position_log, in frame order. Returns (vehicle_id, vehicle_type)
    or (None, None) if no P1 row exists at all."""
    for frame_number in sorted(rows_by_frame):
        for r in rows_by_frame[frame_number]:
            if r["position"] == 1:
                return r["vehicle_id"], r["vehicle_type"]
    return None, None


def detect_discharge_start(track_series):
    """
    track_series: list of dicts (one vehicle_id's own logged rows, sorted by
    time_s), each with at least time_s, frame_number, distance_to_line_b_px.

    Scans for >=T0_STABLE_MIN_STEPS consecutive near-zero deltas (stationary)
    immediately followed by >=T0_MOVEMENT_MIN_STEPS consecutive deltas that
    decrease by more than T0_MOVEMENT_MIN_DROP_PX (sustained movement toward
    line_b). t0 is defined as the timestamp of the FIRST sample showing the
    sustained movement (i.e. the first sample after the stable window).

    Returns a dict: t0_time_s, t0_frame, confidence, reason (None on success),
    transition_index (index into track_series, or None).
    """
    n = len(track_series)
    if n < T0_STABLE_MIN_STEPS + T0_MOVEMENT_MIN_STEPS + 1:
        return {
            "t0_time_s": None, "t0_frame": None, "confidence": "LOW", "transition_index": None,
            "reason": f"only {n} logged sample(s) for this vehicle - too few to establish a "
                      f"stable-then-movement pattern (need at least {T0_STABLE_MIN_STEPS + T0_MOVEMENT_MIN_STEPS + 1})",
        }

    deltas = [track_series[i]["distance_to_line_b_px"] - track_series[i - 1]["distance_to_line_b_px"]
              for i in range(1, n)]

    for i in range(T0_STABLE_MIN_STEPS - 1, len(deltas) - T0_MOVEMENT_MIN_STEPS):
        stable_window = deltas[i - T0_STABLE_MIN_STEPS + 1: i + 1]
        if not all(abs(d) <= T0_STABLE_JITTER_PX for d in stable_window):
            continue
        movement_window = deltas[i + 1: i + 1 + T0_MOVEMENT_MIN_STEPS]
        if all(d <= -T0_MOVEMENT_MIN_DROP_PX for d in movement_window):
            transition_index = i + 1
            confidence = "HIGH" if n >= 6 else "MEDIUM"
            return {
                "t0_time_s": track_series[transition_index]["time_s"],
                "t0_frame": track_series[transition_index]["frame_number"],
                "confidence": confidence, "reason": None, "transition_index": transition_index,
            }

    return {
        "t0_time_s": None, "t0_frame": None, "confidence": "LOW", "transition_index": None,
        "reason": "no stable-then-sustained-movement transition found in this vehicle's logged samples "
                  "(either already moving at first log, or motion was too gradual/irregular for the thresholds)",
    }


def detect_possible_multiple_events(rows_by_vehicle):
    """
    Lightweight heuristic only - does NOT attempt real cycle separation.
    Flags a clip for manual review if there's a gap longer than
    T0_EMPTY_QUEUE_GAP_S between consecutive logged snapshots (any vehicle,
    any position) with queued rows both before and after the gap - i.e. a
    stretch where the queue appears to have been fully empty for a while,
    which could mean a second discharge event started later in the clip.
    """
    all_times = sorted({r["time_s"] for rows in rows_by_vehicle.values() for r in rows})
    if len(all_times) < 2:
        return False
    return any(all_times[i] - all_times[i - 1] > T0_EMPTY_QUEUE_GAP_S for i in range(1, len(all_times)))


if __name__ == "__main__":
    crossing_logs = load_crossing_logs()
    queue_logs = load_queue_position_logs()

    if not crossing_logs:
        print(f"No crossing_log_clip_*.csv files found in {OUTPUT_DIR}.")
        raise SystemExit(1)
    if not queue_logs:
        print(f"No queue_position_log_clip_*.csv files found in {OUTPUT_DIR}.")
        raise SystemExit(1)

    print(f"Loaded crossing logs for {len(crossing_logs)} clip(s): {', '.join(sorted(crossing_logs))}")
    print(f"Loaded queue-position logs for {len(queue_logs)} clip(s): {', '.join(sorted(queue_logs))}")

    events, total_crossed, matched, unmatched = build_event_dataset(crossing_logs, queue_logs)

    write_event_dataset(events, EVENT_DATASET_PATH)
    speed_rows, speed_fields = write_speed_by_type_position(events, SPEED_BY_TYPE_POSITION_PATH)
    write_summary(events, total_crossed, matched, unmatched, queue_logs, SUMMARY_PATH)

    print("\n" + "=" * 70)
    print("CALIBRATION EVENT DATASET SUMMARY")
    print("=" * 70)
    print(f"Total crossed vehicles:         {total_crossed}")
    print(f"Matched vehicles:               {matched}")
    print(f"Unmatched vehicles:             {len(unmatched)}")
    if unmatched:
        preview = ", ".join(f"{ck}:{vid}({vt})" for ck, vid, vt in unmatched[:15])
        more = "" if len(unmatched) <= 15 else f" ... (+{len(unmatched) - 15} more)"
        print(f"  Unmatched sample: {preview}{more}")
    print(f"Total valid calibration rows:   {len(events)}")

    type_counts = Counter(e["vehicle_type"] for e in events)
    print("\nCount by vehicle_type:")
    for vtype, count in type_counts.most_common():
        print(f"  {vtype:12s} {count}")

    group_counts = Counter(e["first_position_group"] for e in events)
    print("\nCount by first_position_group:")
    for g in sorted(group_counts, key=_position_group_sort_key):
        print(f"  {g:6s} {group_counts[g]}")

    max_pos_overall = max((q["max_position"] for q in queue_logs.values()), default=0)
    print(f"\nMaximum spatial_queue_position across all clips: {max_pos_overall}")
    print("Per-clip breakdown:")
    for clip_key in sorted(queue_logs):
        print(f"  {clip_key:10s} max_position={queue_logs[clip_key]['max_position']}")

    print(f"\nEvent dataset saved to:        {EVENT_DATASET_PATH}")
    print(f"Speed-by-type table saved to:  {SPEED_BY_TYPE_POSITION_PATH}")
    print(f"Summary saved to:              {SUMMARY_PATH}")

    print_table("calibrated_effective_speed_by_type_position.csv", speed_rows, speed_fields)
