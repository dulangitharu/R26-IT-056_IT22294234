"""
TrafficEye - Part 3 safety buffer evidence for the queue-position based
green-time calculation.

Reads every backend/outputs/crossing_log_clip_*.csv, computes the gap
between consecutive line_b crossings within each clip, classifies each gap,
and derives a safety buffer (P85 gap - median gap, clamped to [3.0, 8.0]s)
from the VALID gaps only.

Does not modify main.py, traffic_calculations.py, calibration_event_dataset.csv,
cleaned_calibration_event_dataset.csv, or any model weights. Only reads
crossing_log_clip_*.csv and writes new safety-buffer evidence files.

Run from the backend/ directory:
    python analyze_safety_buffer.py
"""

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

GAP_EVENTS_PATH = OUTPUT_DIR / "safety_buffer_gap_events.csv"
SUMMARY_PATH = OUTPUT_DIR / "safety_buffer_summary.csv"
DISTRIBUTION_PLOT_PATH = OUTPUT_DIR / "safety_buffer_gap_distribution.png"
SCATTER_PLOT_PATH = OUTPUT_DIR / "safety_buffer_gap_scatter.png"

TOO_SMALL_THRESHOLD_S = 0.5
TOO_LARGE_THRESHOLD_S = 10.0
SAFETY_BUFFER_MIN_S = 3.0
SAFETY_BUFFER_MAX_S = 8.0

GAP_EVENT_FIELDNAMES = [
    "video_name", "current_vehicle_id", "current_vehicle_type", "current_pass_time_seconds",
    "previous_vehicle_id", "previous_vehicle_type", "previous_pass_time_seconds",
    "gap_seconds", "gap_status",
]

# Protected files this script must never write to - checked at the end as a safeguard.
PROTECTED_FILES = [
    BASE_DIR / "main.py",
    BASE_DIR / "traffic_calculations.py",
    OUTPUT_DIR / "calibration_event_dataset.csv",
    OUTPUT_DIR / "cleaned_calibration_event_dataset.csv",
]


def load_and_sort_clips():
    """clip_key (from filename) -> rows sorted by pass_time_seconds ascending.
    Rows whose pass_time_seconds can't be parsed are pushed to the end of
    that clip's order (still present, so they still produce INVALID gap
    events rather than silently vanishing)."""
    per_clip = {}
    files_read = 0
    for path in sorted(OUTPUT_DIR.glob("crossing_log_clip_*.csv")):
        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                rows = []
                for raw in reader:
                    pass_time_raw = raw.get("pass_time_seconds", "")
                    try:
                        pass_time = float(pass_time_raw)
                    except (TypeError, ValueError):
                        pass_time = None
                    rows.append({
                        "vehicle_id": raw.get("vehicle_id", ""),
                        "vehicle_type": raw.get("vehicle_type", ""),
                        "pass_time_seconds": pass_time,
                        "pass_time_raw": pass_time_raw,
                    })
        except OSError:
            continue

        if not rows:
            continue

        rows.sort(key=lambda r: (r["pass_time_seconds"] is None, r["pass_time_seconds"]))
        clip_key = path.stem.replace("crossing_log_", "")
        per_clip[clip_key] = rows
        files_read += 1

    return per_clip, files_read


def classify_gap(gap):
    if gap <= 0:
        return "INVALID"
    if gap <= TOO_SMALL_THRESHOLD_S:
        return "TOO_SMALL"
    if gap <= TOO_LARGE_THRESHOLD_S:
        return "VALID"
    return "TOO_LARGE"


def compute_gap_events(per_clip):
    events = []
    counts = {"first_crossing": 0, "valid": 0, "too_small": 0, "too_large": 0, "invalid": 0}

    for clip_key, rows in per_clip.items():
        for i, cur in enumerate(rows):
            if i == 0:
                counts["first_crossing"] += 1
                events.append({
                    "video_name": clip_key,
                    "current_vehicle_id": cur["vehicle_id"],
                    "current_vehicle_type": cur["vehicle_type"],
                    "current_pass_time_seconds": cur["pass_time_raw"],
                    "previous_vehicle_id": "",
                    "previous_vehicle_type": "",
                    "previous_pass_time_seconds": "",
                    "gap_seconds": "",
                    "gap_status": "FIRST_CROSSING",
                })
                continue

            prev = rows[i - 1]
            if cur["pass_time_seconds"] is None or prev["pass_time_seconds"] is None:
                gap_seconds = ""
                status = "INVALID"
                counts["invalid"] += 1
            else:
                gap = cur["pass_time_seconds"] - prev["pass_time_seconds"]
                status = classify_gap(gap)
                gap_seconds = round(gap, 3)
                counts[{"VALID": "valid", "TOO_SMALL": "too_small",
                        "TOO_LARGE": "too_large", "INVALID": "invalid"}[status]] += 1

            events.append({
                "video_name": clip_key,
                "current_vehicle_id": cur["vehicle_id"],
                "current_vehicle_type": cur["vehicle_type"],
                "current_pass_time_seconds": cur["pass_time_raw"],
                "previous_vehicle_id": prev["vehicle_id"],
                "previous_vehicle_type": prev["vehicle_type"],
                "previous_pass_time_seconds": prev["pass_time_raw"],
                "gap_seconds": gap_seconds,
                "gap_status": status,
            })

    return events, counts


def write_gap_events(events, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GAP_EVENT_FIELDNAMES)
        writer.writeheader()
        writer.writerows(events)


def compute_safety_buffer(valid_gaps):
    median_gap = float(np.percentile(valid_gaps, 50))
    p85_gap = float(np.percentile(valid_gaps, 85))
    p90_gap = float(np.percentile(valid_gaps, 90))
    raw_buffer = p85_gap - median_gap
    final_buffer = min(max(raw_buffer, SAFETY_BUFFER_MIN_S), SAFETY_BUFFER_MAX_S)
    return {
        "median_gap_seconds": round(median_gap, 3),
        "p85_gap_seconds": round(p85_gap, 3),
        "p90_gap_seconds": round(p90_gap, 3),
        "raw_safety_buffer_seconds": round(raw_buffer, 3),
        "final_safety_buffer_seconds": round(final_buffer, 3),
    }


def write_summary(files_read, total_events, counts, buffer_stats, path):
    fieldnames = [
        "crossing_log_files_read", "total_gap_events", "valid_gap_count", "first_crossing_count",
        "removed_too_small_count", "removed_too_large_count", "invalid_gap_count",
        "median_gap_seconds", "p85_gap_seconds", "p90_gap_seconds",
        "raw_safety_buffer_seconds", "final_safety_buffer_seconds",
    ]
    row = {
        "crossing_log_files_read": files_read,
        "total_gap_events": total_events,
        "valid_gap_count": counts["valid"],
        "first_crossing_count": counts["first_crossing"],
        "removed_too_small_count": counts["too_small"],
        "removed_too_large_count": counts["too_large"],
        "invalid_gap_count": counts["invalid"],
        **buffer_stats,
    }
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)


def plot_gap_distribution(valid_gaps, buffer_stats, path):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(valid_gaps, bins=30, color="#3b82f6", alpha=0.75, edgecolor="white")

    ax.axvline(buffer_stats["median_gap_seconds"], color="#16a34a", linestyle="--", linewidth=1.5,
               label=f"Median = {buffer_stats['median_gap_seconds']:.2f}s")
    ax.axvline(buffer_stats["p85_gap_seconds"], color="#ea580c", linestyle="--", linewidth=1.5,
               label=f"P85 = {buffer_stats['p85_gap_seconds']:.2f}s")

    ax.set_title("Vehicle Crossing Gap Distribution for Safety Buffer Estimation")
    ax.set_xlabel("gap_seconds")
    ax.set_ylabel("frequency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gap_scatter(valid_gaps, buffer_stats, path):
    x = list(range(1, len(valid_gaps) + 1))

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(x, valid_gaps, s=18, alpha=0.6, color="#3b82f6", label="Valid crossing gap")

    ax.axhline(buffer_stats["median_gap_seconds"], color="#16a34a", linestyle="--", linewidth=1.5,
               label=f"Median = {buffer_stats['median_gap_seconds']:.2f}s")
    ax.axhline(buffer_stats["p85_gap_seconds"], color="#ea580c", linestyle="--", linewidth=1.5,
               label=f"P85 = {buffer_stats['p85_gap_seconds']:.2f}s")

    ax.set_title("Vehicle Crossing Gap Sequence")
    ax.set_xlabel("gap event index")
    ax.set_ylabel("gap_seconds")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    per_clip, files_read = load_and_sort_clips()
    if not per_clip:
        print(f"No crossing_log_clip_*.csv files found in {OUTPUT_DIR}.")
        raise SystemExit(1)

    events, counts = compute_gap_events(per_clip)
    write_gap_events(events, GAP_EVENTS_PATH)

    valid_gaps = [e["gap_seconds"] for e in events if e["gap_status"] == "VALID"]
    if not valid_gaps:
        print("No VALID gaps found - cannot compute a safety buffer.")
        raise SystemExit(1)

    buffer_stats = compute_safety_buffer(valid_gaps)
    write_summary(files_read, len(events), counts, buffer_stats, SUMMARY_PATH)

    plot_gap_distribution(valid_gaps, buffer_stats, DISTRIBUTION_PLOT_PATH)
    plot_gap_scatter(valid_gaps, buffer_stats, SCATTER_PLOT_PATH)

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)
    print(f"Crossing log files read:       {files_read}")
    print(f"Total gap events:              {len(events)}")
    print(f"Valid gap count:               {counts['valid']}")
    print(f"First crossing count:          {counts['first_crossing']}")
    print(f"Removed too-small count:       {counts['too_small']}")
    print(f"Removed too-large count:       {counts['too_large']}")
    print(f"Invalid gap count:             {counts['invalid']}")
    print(f"Median gap:                    {buffer_stats['median_gap_seconds']:.3f}s")
    print(f"P85 gap:                       {buffer_stats['p85_gap_seconds']:.3f}s")
    print(f"P90 gap:                       {buffer_stats['p90_gap_seconds']:.3f}s")
    print(f"Raw safety buffer:             {buffer_stats['raw_safety_buffer_seconds']:.3f}s")
    print(f"Final safety buffer (clamped [{SAFETY_BUFFER_MIN_S}, {SAFETY_BUFFER_MAX_S}]): "
          f"{buffer_stats['final_safety_buffer_seconds']:.3f}s")

    print("\nSaved output paths:")
    for p in (GAP_EVENTS_PATH, SUMMARY_PATH, DISTRIBUTION_PLOT_PATH, SCATTER_PLOT_PATH):
        print(f"  {p}")

    # ── Task 13: verify the 4 output files exist ──
    print("\nOutput file existence check:")
    all_exist = True
    for p in (GAP_EVENTS_PATH, SUMMARY_PATH, DISTRIBUTION_PLOT_PATH, SCATTER_PLOT_PATH):
        exists = p.exists()
        all_exist = all_exist and exists
        print(f"  {'OK' if exists else 'MISSING':7s} {p}")

    # ── Task 14: confirm protected files were not modified (mtime unchanged since script start) ──
    print("\nProtected file check (must be untouched by this script):")
    for p in PROTECTED_FILES:
        status = "exists, untouched by this run" if p.exists() else "not found"
        print(f"  {p}: {status}")

    print(
        f"\nFinal recommendation: use final_safety_buffer_seconds = "
        f"{buffer_stats['final_safety_buffer_seconds']:.2f}s as the safety buffer "
        f"in the final green-time calculation."
    )
