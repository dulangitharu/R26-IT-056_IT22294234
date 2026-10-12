"""
Prototype observed-clearance validation - Step 4 (metric script).

Reads outputs/final_validation/final_clearance_validation.csv, uses only rows
with measurement_status=VALID and a numeric observed_clearance_s, and reports
MAE/RMSE over predicted_clearance_s vs observed_clearance_s. Does not invent
observed values and does not report a metric if fewer than 3 valid cases
exist - the count of valid cases is always reported honestly.

This is a prototype observed-clearance validation (clips overlap the existing
calibration set), NOT an independent held-out validation.
"""
import csv
import math
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
VALIDATION_DIR = BASE_DIR / "outputs" / "final_validation"
INPUT_CSV = VALIDATION_DIR / "final_clearance_validation.csv"
SUMMARY_CSV = VALIDATION_DIR / "final_clearance_validation_summary.csv"


def main():
    if not INPUT_CSV.exists():
        raise SystemExit(f"{INPUT_CSV} not found - run generate_final_validation_snapshots.py first.")

    valid_cases = []
    with open(INPUT_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("measurement_status") != "VALID":
                continue
            observed_raw = (row.get("observed_clearance_s") or "").strip()
            if not observed_raw:
                continue
            try:
                observed = float(observed_raw)
                predicted = float(row["predicted_clearance_s"])
            except ValueError:
                continue
            error = round(abs(predicted - observed), 4)
            valid_cases.append({
                "clip_name": row["clip_name"],
                "predicted_clearance_s": predicted,
                "observed_clearance_s": observed,
                "absolute_error_s": error,
            })

    n = len(valid_cases)
    print(f"Valid observed cases: {n}")
    for c in valid_cases:
        print(f"  {c['clip_name']}: predicted={c['predicted_clearance_s']}s "
              f"observed={c['observed_clearance_s']}s abs_error={c['absolute_error_s']}s")

    if n == 0:
        mae_s = ""
        rmse_s = ""
        print("No valid observed_clearance_s values yet - MAE/RMSE not computed.")
    else:
        mae_s = round(sum(c["absolute_error_s"] for c in valid_cases) / n, 4)
        rmse_s = round(math.sqrt(sum(c["absolute_error_s"] ** 2 for c in valid_cases) / n), 4)
        print(f"MAE  = {mae_s}s")
        print(f"RMSE = {rmse_s}s")
        if n < 3:
            print(f"NOTE: only {n} valid case(s) - validation is NOT complete (3 were requested).")

    with open(SUMMARY_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["valid_cases", "mae_s", "rmse_s"])
        writer.writeheader()
        writer.writerow({"valid_cases": n, "mae_s": mae_s, "rmse_s": rmse_s})

    print(f"\nWrote {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
