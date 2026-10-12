"""
TrafficEye - all-clips crossing summary.

Reads every outputs/crossing_log_clip_*.csv and produces a per-clip
breakdown (total crossings + count per vehicle type) plus totals across
all clips.

Run from the backend/ directory:
    python summarize_clips.py
"""

import csv
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
SUMMARY_PATH = OUTPUT_DIR / "all_clips_crossing_summary.csv"

VEHICLE_TYPES = ["Car", "Motorcycle", "Tuktuk", "Bus", "Truck", "Ambulance", "Bicycle"]


def run():
    clip_paths = sorted(OUTPUT_DIR.glob("crossing_log_clip_*.csv"))
    if not clip_paths:
        print(f"No crossing_log_clip_*.csv files found in {OUTPUT_DIR}.")
        return

    rows = []
    totals = defaultdict(int)
    grand_total = 0

    for path in clip_paths:
        clip_name = path.stem.replace("crossing_log_", "")
        counts = defaultdict(int)
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                counts[row["vehicle_type"]] += 1

        clip_total = sum(counts.values())
        grand_total += clip_total
        for vtype in VEHICLE_TYPES:
            totals[vtype] += counts.get(vtype, 0)

        rows.append({
            "clip": clip_name,
            "total_crossings": clip_total,
            **{vtype: counts.get(vtype, 0) for vtype in VEHICLE_TYPES},
        })

    with open(SUMMARY_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip", "total_crossings"] + VEHICLE_TYPES)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Summary saved to: {SUMMARY_PATH}\n")
    header = f"{'clip':12s} {'total':>6s} " + " ".join(f"{v:>10s}" for v in VEHICLE_TYPES)
    print(header)
    for r in rows:
        print(f"{r['clip']:12s} {r['total_crossings']:>6d} " + " ".join(f"{r[v]:>10d}" for v in VEHICLE_TYPES))

    print(f"\n{'TOTAL':12s} {grand_total:>6d} " + " ".join(f"{totals[v]:>10d}" for v in VEHICLE_TYPES))


if __name__ == "__main__":
    run()
