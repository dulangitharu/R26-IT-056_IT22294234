"""
TrafficEye - Calibration data verification (read-only).

Verifies backend/outputs/calibration_event_dataset.csv before it's used to
generate proof graphs or feed into runtime green-time logic. Checks required
columns, position-group mapping consistency, the distance/time speed
formula, and flags outlier/suspicious rows - then writes a cleaned dataset
containing only rows that pass every check.

Does not modify calibration_event_dataset.csv, main.py, traffic_calculations.py,
or any model weights. Only writes new verification output files.

Run from the backend/ directory:
    python verify_calibration_data.py
"""

import csv
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

INPUT_PATH = OUTPUT_DIR / "calibration_event_dataset.csv"
POSITION_MISMATCH_PATH = OUTPUT_DIR / "position_group_mismatch_rows.csv"
SPEED_MISMATCH_PATH = OUTPUT_DIR / "speed_formula_mismatch_rows.csv"
OUTLIERS_PATH = OUTPUT_DIR / "removed_calibration_outliers.csv"
CLEANED_PATH = OUTPUT_DIR / "cleaned_calibration_event_dataset.csv"

REQUIRED_COLUMNS = [
    "video_name", "vehicle_id", "vehicle_type", "first_position", "first_position_group",
    "distance_to_clear_m", "observed_clear_time_s", "effective_discharge_speed_mps", "sample_quality",
]

SPEED_TOLERANCE = 0.01
MIN_CLEAR_TIME_S = 0.5
MAX_PLAUSIBLE_SPEED_MPS = 15


def expected_position_group(first_position):
    if first_position == 1:
        return "P1"
    if first_position == 2:
        return "P2"
    if first_position in (3, 4):
        return "P3-4"
    if first_position >= 5:
        return "P5+"
    return None  # first_position < 1 - not a valid queue position at all


def load_rows():
    if not INPUT_PATH.exists():
        print(f"ERROR: {INPUT_PATH} not found.")
        raise SystemExit(1)
    with open(INPUT_PATH, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return fieldnames, rows


def check_required_columns(fieldnames):
    missing = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
    return missing


def count_by(rows, key_fn):
    counts = defaultdict(int)
    for r in rows:
        counts[key_fn(r)] += 1
    return counts


def check_position_group_mapping(rows):
    mismatches = []
    for r in rows:
        try:
            first_position = int(r["first_position"])
        except (KeyError, ValueError, TypeError):
            mismatches.append(r)
            continue
        expected = expected_position_group(first_position)
        actual = r.get("first_position_group")
        if expected is None or actual != expected:
            mismatches.append(r)
    return mismatches


def check_speed_formula(rows):
    mismatches = []
    for r in rows:
        try:
            distance = float(r["distance_to_clear_m"])
            clear_time = float(r["observed_clear_time_s"])
            reported_speed = float(r["effective_discharge_speed_mps"])
        except (KeyError, ValueError, TypeError):
            mismatches.append(r)
            continue
        if clear_time == 0:
            # Division by zero - can't verify the formula here; this row is
            # already caught by the outlier check (observed_clear_time_s <= 0).
            continue
        calculated_speed = distance / clear_time
        if abs(calculated_speed - reported_speed) > SPEED_TOLERANCE:
            mismatches.append(r)
    return mismatches


def check_outliers(rows):
    outliers = []
    for r in rows:
        try:
            clear_time = float(r["observed_clear_time_s"])
            speed = float(r["effective_discharge_speed_mps"])
        except (KeyError, ValueError, TypeError):
            outliers.append({**r, "outlier_reason": "unparseable_numeric_field"})
            continue

        reasons = []
        if clear_time <= 0:
            reasons.append("observed_clear_time_s<=0")
        if 0 < clear_time < MIN_CLEAR_TIME_S:
            reasons.append(f"observed_clear_time_s<{MIN_CLEAR_TIME_S}")
        if speed <= 0:
            reasons.append("effective_discharge_speed_mps<=0")
        if speed > MAX_PLAUSIBLE_SPEED_MPS:
            reasons.append(f"effective_discharge_speed_mps>{MAX_PLAUSIBLE_SPEED_MPS}")

        if reasons:
            outliers.append({**r, "outlier_reason": ";".join(reasons)})
    return outliers


def write_csv(rows, path, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in fieldnames})


def print_counts(title, counts):
    print(f"\n{title}")
    for key, count in sorted(counts.items(), key=lambda kv: (-kv[1], str(kv[0]))):
        print(f"  {str(key):30s} {count}")


if __name__ == "__main__":
    fieldnames, rows = load_rows()
    total_rows = len(rows)

    print(f"Loaded {INPUT_PATH}")
    print(f"Total row count: {total_rows}")

    # ── Task 7: required columns ──
    missing_columns = check_required_columns(fieldnames)
    if missing_columns:
        print(f"\nERROR: missing required column(s): {', '.join(missing_columns)}")
        print("Cannot proceed with row-level checks until these columns exist.")
        raise SystemExit(1)
    print("\nAll required columns present:")
    print(f"  {', '.join(REQUIRED_COLUMNS)}")

    # ── Task 3: sample_quality counts ──
    quality_counts = count_by(rows, lambda r: r.get("sample_quality", ""))
    exact_count = quality_counts.get("EXACT", 0)
    support_count = quality_counts.get("SUPPORT", 0)
    other_quality_count = total_rows - exact_count - support_count
    print_counts("Rows by sample_quality:", quality_counts)
    print(f"  -> EXACT: {exact_count}, SUPPORT: {support_count}, OTHER: {other_quality_count}")

    # ── Task 4: vehicle_type counts ──
    type_counts = count_by(rows, lambda r: r.get("vehicle_type", ""))
    print_counts("Rows by vehicle_type:", type_counts)

    # ── Task 5: first_position_group counts ──
    group_counts = count_by(rows, lambda r: r.get("first_position_group", ""))
    print_counts("Rows by first_position_group:", group_counts)

    # ── Task 6: vehicle_type + first_position_group counts ──
    type_group_counts = count_by(rows, lambda r: f"{r.get('vehicle_type', '')}|{r.get('first_position_group', '')}")
    print_counts("Rows by vehicle_type + first_position_group:", type_group_counts)

    # ── Task 8: position group mapping check ──
    position_mismatches = check_position_group_mapping(rows)
    write_csv(position_mismatches, POSITION_MISMATCH_PATH, fieldnames)
    print(f"\nPosition-group mapping mismatches: {len(position_mismatches)}")
    print(f"  Saved to: {POSITION_MISMATCH_PATH}")

    # ── Task 9: speed formula check ──
    speed_mismatches = check_speed_formula(rows)
    write_csv(speed_mismatches, SPEED_MISMATCH_PATH, fieldnames)
    print(f"\nSpeed formula mismatches (tolerance {SPEED_TOLERANCE}): {len(speed_mismatches)}")
    print(f"  Saved to: {SPEED_MISMATCH_PATH}")

    # ── Task 10: outlier / suspicious rows ──
    outliers = check_outliers(rows)
    outlier_fieldnames = fieldnames + ["outlier_reason"]
    write_csv(outliers, OUTLIERS_PATH, outlier_fieldnames)
    print(f"\nOutlier/suspicious rows: {len(outliers)}")
    print(f"  Saved to: {OUTLIERS_PATH}")

    # ── Task 11: cleaned dataset (excludes outliers + position/formula mismatches) ──
    bad_row_ids = set()
    for bad_set in (position_mismatches, speed_mismatches, outliers):
        for r in bad_set:
            bad_row_ids.add((r.get("video_name"), r.get("vehicle_id"), r.get("first_queue_time_s")))

    cleaned_rows = [
        r for r in rows
        if (r.get("video_name"), r.get("vehicle_id"), r.get("first_queue_time_s")) not in bad_row_ids
    ]
    write_csv(cleaned_rows, CLEANED_PATH, fieldnames)
    print(f"\nCleaned rows (pass all checks): {len(cleaned_rows)}")
    print(f"  Saved to: {CLEANED_PATH}")

    # ── Task 12: final summary ──
    ready = len(missing_columns) == 0 and len(cleaned_rows) > 0
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Total rows:                      {total_rows}")
    print(f"EXACT rows:                      {exact_count}")
    print(f"SUPPORT rows:                    {support_count}")
    print(f"OTHER sample_quality rows:       {other_quality_count}")
    print(f"Position-group mismatch count:   {len(position_mismatches)}")
    print(f"Speed formula mismatch count:    {len(speed_mismatches)}")
    print(f"Outlier/suspicious row count:    {len(outliers)}")
    print(f"Cleaned row count:               {len(cleaned_rows)}")
    print(f"\nReady for Part 2 graphs: {'YES' if ready else 'NO'}")
    if position_mismatches or speed_mismatches or outliers:
        print(
            f"  NOTE: {len(position_mismatches)} position mismatches, {len(speed_mismatches)} formula "
            f"mismatches, and {len(outliers)} outliers were excluded from the cleaned dataset - "
            f"use cleaned_calibration_event_dataset.csv, not the raw file, for graphing."
        )
