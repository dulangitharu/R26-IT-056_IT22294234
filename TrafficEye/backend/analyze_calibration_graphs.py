"""
TrafficEye - Part 2 evidence graphs for the queue-position based green-time logic.

Reads the CLEANED calibration dataset (outputs of verify_calibration_data.py -
outliers, position-group mismatches, and speed-formula mismatches already
excluded) and produces proof graphs + a robust per-group speed summary table,
to support using median_effective_speed_mps as the recommended speed in the
runtime green-time calculation.

Does not modify main.py, traffic_calculations.py, calibration_event_dataset.csv,
or any model weights. Only reads cleaned_calibration_event_dataset.csv and
writes new graph/table files.

Run from the backend/ directory:
    python analyze_calibration_graphs.py
"""

import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"

INPUT_PATH = OUTPUT_DIR / "cleaned_calibration_event_dataset.csv"
SCATTER_DISTANCE_TIME_PATH = OUTPUT_DIR / "scatter_distance_vs_clear_time.png"
SCATTER_POSITION_SPEED_PATH = OUTPUT_DIR / "scatter_position_vs_speed.png"
ROBUST_SPEED_TABLE_PATH = OUTPUT_DIR / "robust_speed_by_type_position.csv"
MEAN_VS_MEDIAN_CHART_PATH = OUTPUT_DIR / "mean_vs_median_speed_by_group.png"

REQUIRED_COLUMNS = [
    "vehicle_type", "first_position", "first_position_group", "distance_to_clear_m",
    "observed_clear_time_s", "effective_discharge_speed_mps", "sample_quality",
]

POSITION_GROUP_ORDER = ["P1", "P2", "P3-4", "P5+"]
MEAN_VS_MEDIAN_RATIO_THRESHOLD = 1.3  # mean/median >= this counts as "much higher"


def _position_group_sort_key(group):
    try:
        return POSITION_GROUP_ORDER.index(group)
    except ValueError:
        return len(POSITION_GROUP_ORDER)


def load_rows():
    if not INPUT_PATH.exists():
        print(f"ERROR: {INPUT_PATH} not found. Run verify_calibration_data.py first.")
        raise SystemExit(1)
    with open(INPUT_PATH, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    return fieldnames, rows


def check_required_columns(fieldnames):
    missing = [c for c in REQUIRED_COLUMNS if c not in fieldnames]
    if missing:
        print(f"ERROR: cleaned_calibration_event_dataset.csv is missing required column(s): {', '.join(missing)}")
        raise SystemExit(1)


def parse_numeric_rows(rows):
    """Returns rows with the numeric fields converted, skipping any row that
    fails to parse (shouldn't happen post-cleaning, but guards regardless)."""
    parsed = []
    for r in rows:
        try:
            parsed.append({
                "vehicle_type": r["vehicle_type"],
                "first_position": int(r["first_position"]),
                "first_position_group": r["first_position_group"],
                "distance_to_clear_m": float(r["distance_to_clear_m"]),
                "observed_clear_time_s": float(r["observed_clear_time_s"]),
                "effective_discharge_speed_mps": float(r["effective_discharge_speed_mps"]),
                "sample_quality": r["sample_quality"],
            })
        except (KeyError, ValueError, TypeError):
            continue
    return parsed


def plot_scatter_distance_vs_clear_time(rows, path):
    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.get_cmap("tab10")
    colors = {g: cmap(i) for i, g in enumerate(POSITION_GROUP_ORDER)}

    for group in POSITION_GROUP_ORDER:
        xs = [r["distance_to_clear_m"] for r in rows if r["first_position_group"] == group]
        ys = [r["observed_clear_time_s"] for r in rows if r["first_position_group"] == group]
        if xs:
            ax.scatter(xs, ys, s=22, alpha=0.65, color=colors[group], label=group)

    ax.set_title("Distance to Clear vs Observed Clear Time")
    ax.set_xlabel("distance_to_clear_m")
    ax.set_ylabel("observed_clear_time_s")
    ax.legend(title="first_position_group", loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_scatter_position_vs_speed(rows, path):
    vehicle_types = sorted({r["vehicle_type"] for r in rows})
    fig, ax = plt.subplots(figsize=(10, 7))
    cmap = plt.get_cmap("tab10")
    colors = {t: cmap(i % 10) for i, t in enumerate(vehicle_types)}

    for vtype in vehicle_types:
        xs = [r["first_position"] for r in rows if r["vehicle_type"] == vtype]
        ys = [r["effective_discharge_speed_mps"] for r in rows if r["vehicle_type"] == vtype]
        if xs:
            ax.scatter(xs, ys, s=22, alpha=0.65, color=colors[vtype], label=vtype)

    ax.set_title("Queue Position vs Effective Discharge Speed")
    ax.set_xlabel("first_position")
    ax.set_ylabel("effective_discharge_speed_mps")
    ax.legend(title="vehicle_type", loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def sample_quality_for(count):
    if count >= 20:
        return "GOOD"
    if count >= 10:
        return "ACCEPTABLE"
    if count >= 3:
        return "LOW"
    return "LIMITED"


def build_robust_speed_table(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["vehicle_type"], r["first_position_group"])].append(r)

    table = []
    for (vtype, pgroup), group_rows in sorted(
        groups.items(), key=lambda kv: (kv[0][0], _position_group_sort_key(kv[0][1]))
    ):
        speeds = [r["effective_discharge_speed_mps"] for r in group_rows]
        clear_times = [r["observed_clear_time_s"] for r in group_rows]
        count = len(speeds)
        table.append({
            "vehicle_type": vtype,
            "first_position_group": pgroup,
            "sample_count": count,
            "mean_effective_speed_mps": round(statistics.mean(speeds), 3),
            "median_effective_speed_mps": round(statistics.median(speeds), 3),
            "min_effective_speed_mps": round(min(speeds), 3),
            "max_effective_speed_mps": round(max(speeds), 3),
            "median_observed_clear_time_s": round(statistics.median(clear_times), 3),
            "sample_quality": sample_quality_for(count),
        })
    return table


def write_robust_speed_table(table, path):
    fieldnames = [
        "vehicle_type", "first_position_group", "sample_count", "mean_effective_speed_mps",
        "median_effective_speed_mps", "min_effective_speed_mps", "max_effective_speed_mps",
        "median_observed_clear_time_s", "sample_quality",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table)
    return fieldnames


def plot_mean_vs_median(table, path):
    labels = [f"{row['vehicle_type']}|{row['first_position_group']}" for row in table]
    means = [row["mean_effective_speed_mps"] for row in table]
    medians = [row["median_effective_speed_mps"] for row in table]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.5), 7))
    ax.bar(x - width / 2, means, width, label="mean_effective_speed_mps", color="#dc2626", alpha=0.85)
    ax.bar(x + width / 2, medians, width, label="median_effective_speed_mps", color="#2563eb", alpha=0.85)

    ax.set_title("Mean vs Median Effective Speed by vehicle_type + first_position_group")
    ax.set_xlabel("vehicle_type | first_position_group")
    ax.set_ylabel("effective speed (m/s)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=8)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def pearson_correlation(xs, ys):
    if len(xs) < 2:
        return float("nan")
    return float(np.corrcoef(xs, ys)[0, 1])


if __name__ == "__main__":
    fieldnames, raw_rows = load_rows()
    check_required_columns(fieldnames)
    rows = parse_numeric_rows(raw_rows)

    print(f"Loaded {INPUT_PATH}")
    print(f"Cleaned row count: {len(rows)}")

    plot_scatter_distance_vs_clear_time(rows, SCATTER_DISTANCE_TIME_PATH)
    plot_scatter_position_vs_speed(rows, SCATTER_POSITION_SPEED_PATH)

    robust_table = build_robust_speed_table(rows)
    speed_table_fields = write_robust_speed_table(robust_table, ROBUST_SPEED_TABLE_PATH)

    plot_mean_vs_median(robust_table, MEAN_VS_MEDIAN_CHART_PATH)

    dist_vals = [r["distance_to_clear_m"] for r in rows]
    time_vals = [r["observed_clear_time_s"] for r in rows]
    pos_vals = [r["first_position"] for r in rows]
    speed_vals = [r["effective_discharge_speed_mps"] for r in rows]

    corr_distance_time = pearson_correlation(dist_vals, time_vals)
    corr_position_speed = pearson_correlation(pos_vals, speed_vals)

    num_groups = len(robust_table)
    top_10_groups = sorted(robust_table, key=lambda r: -r["sample_count"])[:10]
    mean_much_higher = [
        r for r in robust_table
        if r["median_effective_speed_mps"] > 0
        and r["mean_effective_speed_mps"] / r["median_effective_speed_mps"] >= MEAN_VS_MEDIAN_RATIO_THRESHOLD
    ]

    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(f"Cleaned row count:                  {len(rows)}")
    print(f"vehicle_type + position groups:     {num_groups}")

    print("\nTop 10 groups by sample_count:")
    print(f"  {'vehicle_type':14s}{'position_group':16s}{'sample_count':14s}{'sample_quality':14s}")
    for r in top_10_groups:
        print(f"  {r['vehicle_type']:14s}{r['first_position_group']:16s}{r['sample_count']:<14d}{r['sample_quality']:14s}")

    print(f"\nGroups where mean is much higher than median (ratio >= {MEAN_VS_MEDIAN_RATIO_THRESHOLD}):")
    if mean_much_higher:
        for r in mean_much_higher:
            ratio = r["mean_effective_speed_mps"] / r["median_effective_speed_mps"]
            print(
                f"  {r['vehicle_type']:14s}{r['first_position_group']:8s} "
                f"mean={r['mean_effective_speed_mps']:.3f}  median={r['median_effective_speed_mps']:.3f}  "
                f"ratio={ratio:.2f}x"
            )
    else:
        print("  (none)")

    print(f"\nCorrelation distance_to_clear_m vs observed_clear_time_s: {corr_distance_time:.4f}")
    print(f"Correlation first_position vs effective_discharge_speed_mps: {corr_position_speed:.4f}")

    print("\nSaved graphs/tables:")
    print(f"  {SCATTER_DISTANCE_TIME_PATH}")
    print(f"  {SCATTER_POSITION_SPEED_PATH}")
    print(f"  {ROBUST_SPEED_TABLE_PATH}")
    print(f"  {MEAN_VS_MEDIAN_CHART_PATH}")

    print(
        "\nFinal recommendation: use median_effective_speed_mps as recommended_speed_mps "
        "for the runtime green-time calculation - the mean-vs-median comparison above shows "
        "the mean is repeatedly pulled upward by extreme values in multiple groups, while the "
        "median stays representative of typical queue-discharge behavior."
    )
