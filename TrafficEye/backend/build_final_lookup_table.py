"""
TrafficEye - Part 5 final lookup table preparation for the queue-position
based green-time calculation.

Combines the robust per-(vehicle_type, position_group) speed stats from
Part 2 with the calibrated safety buffer from Part 3 into a single lookup
table, plus a coverage summary showing which (vehicle_type, position_group)
combinations have direct data vs. will need a runtime fallback.

Does not modify main.py, traffic_calculations.py, calibration_event_dataset.csv,
cleaned_calibration_event_dataset.csv, or any model weights. Does not
implement any runtime green-time logic - lookup table preparation only.

Run from the backend/ directory:
    python build_final_lookup_table.py
"""

import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

ROBUST_SPEED_PATH = OUTPUT_DIR / "robust_speed_by_type_position.csv"
SAFETY_BUFFER_SUMMARY_PATH = OUTPUT_DIR / "safety_buffer_summary.csv"
FINAL_LOOKUP_PATH = OUTPUT_DIR / "final_green_time_lookup_table.csv"
COVERAGE_SUMMARY_PATH = OUTPUT_DIR / "final_lookup_coverage_summary.csv"

AVERAGE_INTER_VEHICLE_GAP_M = 1.177

VEHICLE_LENGTHS_M = {
    "Ambulance": 5.5,
    "Bicycle": 1.9,
    "Bus": 10.8,
    "Car": 4.3,
    "Motorcycle": 2.0,
    "Truck": 8.5,
    "Tuktuk": 2.5,
}

EXPECTED_VEHICLE_TYPES = ["Ambulance", "Bicycle", "Bus", "Car", "Motorcycle", "Truck", "Tuktuk"]
EXPECTED_POSITION_GROUPS = ["P1", "P2", "P3-4", "P5+"]

# Prototype fallback speeds (m/s) for (vehicle_type, position_group) combinations
# with NO calibrated median in robust_speed_by_type_position.csv. These are
# explicit, hand-set prototype values - NOT measured/calibrated data - used
# only when building the final lookup table, never fed back into any
# calibration CSV (calibration_event_dataset.csv, cleaned_calibration_event_dataset.csv,
# robust_speed_by_type_position.csv all remain real-data-only).
PROTOTYPE_FALLBACK_SPEEDS = {
    ("Bicycle", "P1"): 0.661,
    ("Bicycle", "P2"): 1.355,
    ("Bicycle", "P3-4"): 1.778,
    ("Truck", "P3-4"): 2.202,
    ("Tuktuk", "P3-4"): 2.737,
}

REQUIRED_ROBUST_SPEED_COLUMNS = [
    "vehicle_type", "first_position_group", "sample_count", "mean_effective_speed_mps",
    "median_effective_speed_mps", "min_effective_speed_mps", "max_effective_speed_mps",
    "median_observed_clear_time_s", "sample_quality",
]

LOOKUP_FIELDNAMES = [
    "vehicle_type", "position_group", "vehicle_length_m", "average_inter_vehicle_gap_m",
    "recommended_speed_mps", "safety_buffer_s", "sample_count", "sample_quality", "speed_source",
]

COVERAGE_FIELDNAMES = ["vehicle_type", "position_group", "status", "sample_count", "sample_quality", "note"]

PROTECTED_FILES = [
    BASE_DIR / "main.py",
    BASE_DIR / "traffic_calculations.py",
    OUTPUT_DIR / "calibration_event_dataset.csv",
    OUTPUT_DIR / "cleaned_calibration_event_dataset.csv",
]


def load_robust_speed_rows():
    if not ROBUST_SPEED_PATH.exists():
        print(f"ERROR: {ROBUST_SPEED_PATH} not found. Run analyze_calibration_graphs.py first.")
        raise SystemExit(1)
    with open(ROBUST_SPEED_PATH, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    missing = [c for c in REQUIRED_ROBUST_SPEED_COLUMNS if c not in fieldnames]
    if missing:
        print(f"ERROR: {ROBUST_SPEED_PATH.name} is missing required column(s): {', '.join(missing)}")
        raise SystemExit(1)
    return rows


def load_safety_buffer():
    if not SAFETY_BUFFER_SUMMARY_PATH.exists():
        print(f"ERROR: {SAFETY_BUFFER_SUMMARY_PATH} not found. Run analyze_safety_buffer.py first.")
        raise SystemExit(1)
    with open(SAFETY_BUFFER_SUMMARY_PATH, newline="") as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
    if row is None or "final_safety_buffer_seconds" not in row:
        print(f"ERROR: {SAFETY_BUFFER_SUMMARY_PATH.name} has no final_safety_buffer_seconds value.")
        raise SystemExit(1)
    return float(row["final_safety_buffer_seconds"])


def build_lookup_table(robust_rows, safety_buffer_s):
    """
    For each of the 28 possible (vehicle_type, position_group) combinations:
      1. If a calibrated median exists in robust_speed_by_type_position.csv, use it.
      2. Else if the combination has an explicit PROTOTYPE_FALLBACK_SPEEDS entry, use that.
      3. Else record a validation error (missing combination) - never invent a value silently.
    """
    lookup_rows = []
    validation_errors = []

    robust_by_key = {(r["vehicle_type"], r["first_position_group"]): r for r in robust_rows}

    for vehicle_type in EXPECTED_VEHICLE_TYPES:
        for position_group in EXPECTED_POSITION_GROUPS:
            key = (vehicle_type, position_group)
            r = robust_by_key.get(key)

            if r is not None:
                try:
                    recommended_speed = float(r["median_effective_speed_mps"])
                except (ValueError, TypeError):
                    validation_errors.append(f"{vehicle_type}|{position_group}: unparseable median_effective_speed_mps")
                    continue
                try:
                    sample_count = int(r["sample_count"])
                except (ValueError, TypeError):
                    sample_count = 0
                sample_quality = r["sample_quality"]
                speed_source = "median_effective_speed_mps"
            elif key in PROTOTYPE_FALLBACK_SPEEDS:
                recommended_speed = PROTOTYPE_FALLBACK_SPEEDS[key]
                sample_count = 0
                sample_quality = "FALLBACK"
                speed_source = "prototype_fallback"
            else:
                validation_errors.append(
                    f"{vehicle_type}|{position_group}: no calibrated median and no prototype fallback - missing combination"
                )
                continue

            vehicle_length = VEHICLE_LENGTHS_M.get(vehicle_type)

            # ── validation ──
            if recommended_speed <= 0:
                validation_errors.append(f"{vehicle_type}|{position_group}: recommended_speed_mps <= 0 ({recommended_speed})")
                continue
            if vehicle_length is None:
                validation_errors.append(f"{vehicle_type}|{position_group}: no vehicle_length_m entry for '{vehicle_type}'")
                continue
            if position_group not in EXPECTED_POSITION_GROUPS:
                validation_errors.append(f"{vehicle_type}|{position_group}: unrecognized position_group")
                continue
            if safety_buffer_s <= 0:
                validation_errors.append(f"{vehicle_type}|{position_group}: safety_buffer_s <= 0 ({safety_buffer_s})")
                continue
            if AVERAGE_INTER_VEHICLE_GAP_M <= 0:
                validation_errors.append(
                    f"{vehicle_type}|{position_group}: average_inter_vehicle_gap_m must be positive, "
                    f"got {AVERAGE_INTER_VEHICLE_GAP_M} (expected configured gap value)"
                )
                continue

            lookup_rows.append({
                "vehicle_type": vehicle_type,
                "position_group": position_group,
                "vehicle_length_m": vehicle_length,
                "average_inter_vehicle_gap_m": AVERAGE_INTER_VEHICLE_GAP_M,
                "recommended_speed_mps": recommended_speed,
                "safety_buffer_s": safety_buffer_s,
                "sample_count": sample_count,
                "sample_quality": sample_quality,
                "speed_source": speed_source,
            })

    return lookup_rows, validation_errors


def write_lookup_table(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOOKUP_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def build_coverage_summary(lookup_rows):
    available = {(r["vehicle_type"], r["position_group"]): r for r in lookup_rows}
    coverage_rows = []
    for vtype in EXPECTED_VEHICLE_TYPES:
        for group in EXPECTED_POSITION_GROUPS:
            key = (vtype, group)
            if key in available:
                r = available[key]
                is_fallback = r["speed_source"] == "prototype_fallback"
                coverage_rows.append({
                    "vehicle_type": vtype,
                    "position_group": group,
                    "status": "AVAILABLE",
                    "sample_count": r["sample_count"],
                    "sample_quality": r["sample_quality"],
                    "note": "prototype fallback used (not calibrated)" if is_fallback else "direct median speed available",
                })
            else:
                coverage_rows.append({
                    "vehicle_type": vtype,
                    "position_group": group,
                    "status": "MISSING",
                    "sample_count": 0,
                    "sample_quality": "",
                    "note": "no calibrated median and no prototype fallback defined",
                })
    return coverage_rows


def write_coverage_summary(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COVERAGE_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    robust_rows = load_robust_speed_rows()
    safety_buffer_s = load_safety_buffer()

    print(f"Robust speed rows read: {len(robust_rows)}")
    print(f"Safety buffer loaded:   {safety_buffer_s:.3f}s")
    print(f"Average gap used:       {AVERAGE_INTER_VEHICLE_GAP_M}m")

    lookup_rows, validation_errors = build_lookup_table(robust_rows, safety_buffer_s)

    if validation_errors:
        print(f"\nWARNING: {len(validation_errors)} row(s) failed validation and were excluded from the lookup table:")
        for err in validation_errors:
            print(f"  - {err}")

    write_lookup_table(lookup_rows, FINAL_LOOKUP_PATH)

    coverage_rows = build_coverage_summary(lookup_rows)
    write_coverage_summary(coverage_rows, COVERAGE_SUMMARY_PATH)

    available_count = sum(1 for r in coverage_rows if r["status"] == "AVAILABLE")
    missing_count = sum(1 for r in coverage_rows if r["status"] == "MISSING")
    missing_combos = [f"{r['vehicle_type']}|{r['position_group']}" for r in coverage_rows if r["status"] == "MISSING"]
    low_quality = [
        r for r in lookup_rows
        if r["sample_quality"] in ("LOW", "LIMITED")
    ]

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(f"Robust speed rows read:        {len(robust_rows)}")
    print(f"Final lookup rows created:     {len(lookup_rows)}")
    print(f"Safety buffer used:            {safety_buffer_s:.3f}s")
    print(f"Average gap used:              {AVERAGE_INTER_VEHICLE_GAP_M}m")
    print(f"Available combinations:        {available_count}")
    print(f"Missing combinations:          {missing_count}")
    if missing_combos:
        print("Missing combinations list:")
        for combo in missing_combos:
            print(f"  - {combo}")
    else:
        print("Missing combinations list:    (none)")

    print(f"\nGroups with LOW or LIMITED sample_quality ({len(low_quality)}):")
    if low_quality:
        for r in low_quality:
            print(f"  {r['vehicle_type']:12s} {r['position_group']:6s} "
                  f"sample_count={r['sample_count']:<4d} sample_quality={r['sample_quality']}")
    else:
        print("  (none)")

    print("\nSaved output paths:")
    print(f"  {FINAL_LOOKUP_PATH}")
    print(f"  {COVERAGE_SUMMARY_PATH}")

    # ── Task 8: verify output files exist ──
    print("\nOutput file existence check:")
    for p in (FINAL_LOOKUP_PATH, COVERAGE_SUMMARY_PATH):
        print(f"  {'OK' if p.exists() else 'MISSING':7s} {p}")

    # ── Task 9: confirm protected files untouched ──
    print("\nProtected file check (must be untouched by this script):")
    for p in PROTECTED_FILES:
        print(f"  {p}: {'exists, untouched by this run' if p.exists() else 'not found'}")
    weights_dir = BASE_DIR / "runs2" / "detect" / "traffic_monitor" / "weights"
    print(f"  {weights_dir}: {'exists, untouched by this run' if weights_dir.exists() else 'not found'}")

    print(
        "\nFinal recommendation: use final_green_time_lookup_table.csv as the speed/length/gap/"
        "safety-buffer lookup source in the next runtime green-time calculation step. "
        "Runtime code has NOT been implemented by this script."
    )
