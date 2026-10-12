"""
TrafficEye - Module 4: Calibration analyzer.

Reads every outputs/crossing_log_clip_*.csv, sorts each clip's crossings by
pass_time_seconds, and computes the headway (gap) between each crossing and
the one immediately before it in that clip. Gaps larger than
LARGE_GAP_THRESHOLD_S are treated as a break in traffic flow (not genuine
vehicle-to-vehicle spacing) and excluded from the passing-time averages.

The resulting per-vehicle-type passing-time statistics feed directly into
the clearance-time calculation in traffic_calculations.py. This module also
derives a numeric safety-buffer figure (outputs/buffer_calibration.csv) from
the same crossing logs' inter-vehicle gaps, which feeds the suggested
green-time calculation's SAFETY_BUFFER_S.

Run from the backend/ directory:
    python calibration_analyzer.py
"""

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless - just saving a PNG, no display needed
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
LARGE_GAP_THRESHOLD_S = 10.0
SMALL_GAP_THRESHOLD_S = 0.5   # gaps at/below this are treated as simultaneous/side-by-side crossings, not sequential passing time
TRIM_PROPORTION = 0.1         # trimmed mean drops this fraction of values off each end

REMOVED_LARGE_GAPS_PATH = OUTPUT_DIR / "removed_large_gaps.csv"
REMOVED_SMALL_GAPS_PATH = OUTPUT_DIR / "removed_small_gaps.csv"
AVG_PASSING_TIME_PATH = OUTPUT_DIR / "average_passing_time.csv"
BUFFER_GAP_ANALYSIS_PATH = OUTPUT_DIR / "buffer_gap_analysis.csv"
BUFFER_CALIBRATION_PATH = OUTPUT_DIR / "buffer_calibration.csv"
BUFFER_GAP_SCATTERPLOT_PATH = OUTPUT_DIR / "buffer_gap_scatterplot.png"

SAFETY_BUFFER_MIN_S = 3.0     # final_safety_buffer_s is clamped to this range regardless
SAFETY_BUFFER_MAX_S = 8.0     # of what the raw P85-minus-median calculation produces

IMPORTANT_TYPES = {"Car", "Motorcycle", "Tuktuk"}   # want 20+ crossings each
SECONDARY_TYPES = {"Bus", "Truck"}                  # want 5-10+ crossings each


def load_crossing_logs():
    """Read every crossing_log_clip_*.csv, sorted by pass_time_seconds within each clip."""
    per_clip = {}
    for path in sorted(OUTPUT_DIR.glob("crossing_log_clip_*.csv")):
        clip_name = path.stem.replace("crossing_log_", "")
        rows = []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                rows.append({
                    "vehicle_id": int(row["vehicle_id"]),
                    "vehicle_type": row["vehicle_type"],
                    "frame_number": int(row["frame_number"]),
                    "pass_time_seconds": float(row["pass_time_seconds"]),
                })
        rows.sort(key=lambda r: r["pass_time_seconds"])
        per_clip[clip_name] = rows
    return per_clip


def compute_gaps(per_clip):
    """
    Returns (valid_gaps, removed_large, removed_small): all three lists of
    dict rows carrying clip/vehicle/gap info. A clip's first crossing has no
    previous crossing to gap against, so it contributes no gap row at all
    (it's the start of the clip, not a traffic break or a simultaneous
    crossing, so it doesn't belong in any of the three lists).
    """
    valid = []
    removed_large = []
    removed_small = []
    for clip_name, rows in per_clip.items():
        for i in range(1, len(rows)):
            prev_row = rows[i - 1]
            cur_row = rows[i]
            gap = round(cur_row["pass_time_seconds"] - prev_row["pass_time_seconds"], 3)
            record = {
                "clip": clip_name,
                "vehicle_id": cur_row["vehicle_id"],
                "vehicle_type": cur_row["vehicle_type"],
                "frame_number": cur_row["frame_number"],
                "pass_time_seconds": cur_row["pass_time_seconds"],
                "prev_pass_time_seconds": prev_row["pass_time_seconds"],
                "gap_seconds": gap,
            }
            if gap > LARGE_GAP_THRESHOLD_S:
                removed_large.append(record)
            elif gap <= SMALL_GAP_THRESHOLD_S:
                removed_small.append(record)
            else:
                valid.append(record)
    return valid, removed_large, removed_small


def write_gap_csv(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "clip", "vehicle_id", "vehicle_type", "frame_number",
            "pass_time_seconds", "prev_pass_time_seconds", "gap_seconds",
        ])
        writer.writeheader()
        writer.writerows(rows)


def trimmed_mean(values, proportion=TRIM_PROPORTION):
    """Mean after dropping `proportion` of values off each end (sorted) - less sensitive to outliers than a raw mean."""
    values = sorted(values)
    n = len(values)
    k = int(n * proportion)
    trimmed = values[k:n - k] if n - 2 * k > 0 else values
    return statistics.mean(trimmed)


def write_average_passing_time(valid):
    by_type = defaultdict(list)
    for r in valid:
        by_type[r["vehicle_type"]].append(r["gap_seconds"])

    rows = []
    with open(AVG_PASSING_TIME_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "vehicle_type", "count", "avg_passing_time_s", "median_gap_s",
            "trimmed_mean_gap_s", "min_gap_s", "max_gap_s",
        ])
        for vtype, gaps in sorted(by_type.items(), key=lambda kv: -len(kv[1])):
            row = [
                vtype,
                len(gaps),
                round(statistics.mean(gaps), 2),
                round(statistics.median(gaps), 2),
                round(trimmed_mean(gaps), 2),
                round(min(gaps), 2),
                round(max(gaps), 2),
            ]
            writer.writerow(row)
            rows.append(row)
    return by_type, rows


def print_sample_size_assessment(per_clip):
    all_crossings = [row for rows in per_clip.values() for row in rows]
    total = len(all_crossings)
    by_type_counts = defaultdict(int)
    for r in all_crossings:
        by_type_counts[r["vehicle_type"]] += 1

    print("\n" + "=" * 60)
    print("SAMPLE SIZE ASSESSMENT")
    print("=" * 60)
    print(f"Total crossings across all clips: {total}")
    if total >= 100:
        verdict = "VERY GOOD"
    elif total >= 80:
        verdict = "GOOD"
    elif total >= 50:
        verdict = "ACCEPTABLE for a prototype"
    else:
        verdict = "LOW - consider collecting more footage"
    print(f"  -> {verdict}")

    print("\nBreakdown by vehicle type:")
    seen_types = set(by_type_counts)
    for vtype in sorted(seen_types, key=lambda k: -by_type_counts[k]):
        count = by_type_counts[vtype]
        if vtype in IMPORTANT_TYPES:
            flag = "OK (20+)" if count >= 20 else "LOW - want 20+"
        elif vtype in SECONDARY_TYPES:
            flag = "OK (5-10+)" if count >= 5 else "LOW - want 5-10+"
        else:
            flag = ""
        print(f"  {vtype:12s} {count:4d}   {flag}")

    for vtype in sorted(IMPORTANT_TYPES - seen_types):
        print(f"  {vtype:12s} {0:4d}   LOW - want 20+")
    for vtype in sorted(SECONDARY_TYPES - seen_types):
        print(f"  {vtype:12s} {0:4d}   LOW - want 5-10+")


def compute_buffer_gap_rows(per_clip):
    """
    Full per-crossing audit trail for safety-buffer calibration - unlike
    compute_gaps() above (which silently drops each clip's first crossing),
    every row is kept here with an explicit gap_status: "first_crossing" (no
    previous crossing to gap against), "removed_small_gap" (<=SMALL_GAP_THRESHOLD_S,
    simultaneous/side-by-side), "removed_large_gap" (>LARGE_GAP_THRESHOLD_S,
    a traffic-flow break), or "valid".
    """
    rows = []
    for clip_name, crossings in per_clip.items():
        for i, cur in enumerate(crossings):
            if i == 0:
                gap, status = None, "first_crossing"
            else:
                gap = round(cur["pass_time_seconds"] - crossings[i - 1]["pass_time_seconds"], 3)
                if gap > LARGE_GAP_THRESHOLD_S:
                    status = "removed_large_gap"
                elif gap <= SMALL_GAP_THRESHOLD_S:
                    status = "removed_small_gap"
                else:
                    status = "valid"
            rows.append({
                "clip_id": clip_name,
                "sequence_number": i + 1,
                "vehicle_type": cur["vehicle_type"],
                "pass_time_seconds": cur["pass_time_seconds"],
                "gap_seconds": gap,
                "gap_status": status,
            })
    return rows


def write_buffer_gap_analysis(rows, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "clip_id", "sequence_number", "vehicle_type",
            "pass_time_seconds", "gap_seconds", "gap_status",
        ])
        writer.writeheader()
        for r in rows:
            row = dict(r)
            row["gap_seconds"] = "" if row["gap_seconds"] is None else row["gap_seconds"]
            writer.writerow(row)


def compute_buffer_calibration(rows):
    """
    Numeric safety-buffer derivation from valid gaps only: the buffer is not
    manually guessed from the scatterplot (that's visual evidence only) - it's
    calculated as P85 gap minus median gap, then clamped to
    [SAFETY_BUFFER_MIN_S, SAFETY_BUFFER_MAX_S] so an unusually sparse/dense
    sample can't push it to an unreasonable extreme.
    """
    valid_gaps = [r["gap_seconds"] for r in rows if r["gap_status"] == "valid"]
    counts = defaultdict(int)
    for r in rows:
        counts[r["gap_status"]] += 1

    median_gap = float(np.percentile(valid_gaps, 50))
    p85_gap = float(np.percentile(valid_gaps, 85))
    p90_gap = float(np.percentile(valid_gaps, 90))
    raw_buffer = p85_gap - median_gap
    final_buffer = min(max(raw_buffer, SAFETY_BUFFER_MIN_S), SAFETY_BUFFER_MAX_S)

    return {
        "median_gap_s": round(median_gap, 3),
        "p85_gap_s": round(p85_gap, 3),
        "p90_gap_s": round(p90_gap, 3),
        "raw_buffer_s": round(raw_buffer, 3),
        "final_safety_buffer_s": round(final_buffer, 3),
        "valid_gap_count": counts["valid"],
        "removed_small_gap_count": counts["removed_small_gap"],
        "removed_large_gap_count": counts["removed_large_gap"],
        "first_crossing_count": counts["first_crossing"],
    }


def write_buffer_calibration(calib, path):
    fieldnames = [
        "median_gap_s", "p85_gap_s", "p90_gap_s", "raw_buffer_s", "final_safety_buffer_s",
        "valid_gap_count", "removed_small_gap_count", "removed_large_gap_count", "first_crossing_count",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(calib)


def plot_buffer_gap_scatterplot(rows, calib, path):
    """Visual evidence only - the final_safety_buffer_s value itself comes from
    compute_buffer_calibration()'s numeric P85-minus-median calculation above,
    not from reading this chart."""
    valid_rows = [r for r in rows if r["gap_status"] == "valid"]
    x = list(range(1, len(valid_rows) + 1))
    y = [r["gap_seconds"] for r in valid_rows]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(x, y, s=18, alpha=0.6, color="#3b82f6", label="Valid crossing gap")

    ax.axhline(calib["median_gap_s"], color="#16a34a", linestyle="--", linewidth=1.5,
               label=f"Median = {calib['median_gap_s']:.2f}s")
    ax.axhline(calib["p85_gap_s"], color="#ea580c", linestyle="--", linewidth=1.5,
               label=f"P85 = {calib['p85_gap_s']:.2f}s")
    ax.axhline(calib["p90_gap_s"], color="#dc2626", linestyle="--", linewidth=1.5,
               label=f"P90 = {calib['p90_gap_s']:.2f}s")

    ax.set_title("Observed Vehicle Crossing Gaps for Safety Buffer Calibration")
    ax.set_xlabel("Crossing sequence number (valid gaps, combined across clips)")
    ax.set_ylabel("Gap seconds")
    ax.legend(loc="upper right")

    ax.annotate(
        f"Safety Buffer = P85 - Median, Final Buffer = {calib['final_safety_buffer_s']:.2f} seconds",
        xy=(0.5, -0.12), xycoords="axes fraction", ha="center", fontsize=9, color="#374151",
    )

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    per_clip = load_crossing_logs()
    if not per_clip:
        print(f"No crossing_log_clip_*.csv files found in {OUTPUT_DIR}. Run track_crossings.py first.")
        raise SystemExit(1)

    print(f"Loaded {len(per_clip)} clip(s): {', '.join(per_clip.keys())}")

    valid, removed_large, removed_small = compute_gaps(per_clip)
    write_gap_csv(removed_large, REMOVED_LARGE_GAPS_PATH)
    write_gap_csv(removed_small, REMOVED_SMALL_GAPS_PATH)
    by_type, rows = write_average_passing_time(valid)

    print(f"\nValid gaps used for averages:                 {len(valid)}")
    print(f"Large gaps (>{LARGE_GAP_THRESHOLD_S:.1f}s, traffic breaks) excluded:   {len(removed_large)}")
    print(f"Small gaps (<={SMALL_GAP_THRESHOLD_S:.1f}s, simultaneous/side-by-side) excluded: {len(removed_small)}")
    print(f"Large gaps saved to:         {REMOVED_LARGE_GAPS_PATH}")
    print(f"Small gaps saved to:         {REMOVED_SMALL_GAPS_PATH}")
    print(f"Average passing time saved: {AVG_PASSING_TIME_PATH}")

    print("\n" + "=" * 60)
    print("average_passing_time.csv")
    print("=" * 60)
    header = ["vehicle_type", "count", "avg_passing_time_s", "median_gap_s", "trimmed_mean_gap_s", "min_gap_s", "max_gap_s"]
    print(" ".join(f"{h:>18s}" if i == 0 else f"{h:>12s}" for i, h in enumerate(header)))
    for row in rows:
        print(" ".join(f"{v:>18}" if i == 0 else f"{v:>12}" for i, v in enumerate(row)))

    print(
        "\nRecommendation: use median_gap_s (or trimmed_mean_gap_s) as the passing-time\n"
        "figure per vehicle type, not avg_passing_time_s. The raw mean is still pulled\n"
        "around by a handful of longer gaps even after the >10s / <=0.5s filtering, while\n"
        "the median and trimmed mean are far more robust to that remaining skew."
    )

    print_sample_size_assessment(per_clip)

    # ── Safety-buffer calibration (from the same crossing logs, not average_passing_time.csv) ──
    buffer_rows = compute_buffer_gap_rows(per_clip)
    write_buffer_gap_analysis(buffer_rows, BUFFER_GAP_ANALYSIS_PATH)
    calib = compute_buffer_calibration(buffer_rows)
    write_buffer_calibration(calib, BUFFER_CALIBRATION_PATH)
    plot_buffer_gap_scatterplot(buffer_rows, calib, BUFFER_GAP_SCATTERPLOT_PATH)

    print("\n" + "=" * 60)
    print("buffer_calibration.csv")
    print("=" * 60)
    for k, v in calib.items():
        print(f"  {k:28s} {v}")

    print(f"\nBuffer gap analysis saved to: {BUFFER_GAP_ANALYSIS_PATH}")
    print(f"Buffer calibration saved to:  {BUFFER_CALIBRATION_PATH}")
    print(f"Scatterplot saved to:         {BUFFER_GAP_SCATTERPLOT_PATH}")
    print(
        f"\nfinal_safety_buffer_s = clamp(P85 - median, {SAFETY_BUFFER_MIN_S}, {SAFETY_BUFFER_MAX_S}) "
        f"= clamp({calib['p85_gap_s']:.2f} - {calib['median_gap_s']:.2f}, {SAFETY_BUFFER_MIN_S}, {SAFETY_BUFFER_MAX_S}) "
        f"= {calib['final_safety_buffer_s']:.2f}s"
    )
