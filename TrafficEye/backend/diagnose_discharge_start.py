"""
TrafficEye - Common discharge-start (t0) detection - DRY-RUN DIAGNOSTIC ONLY.

Reads the EXISTING crossing_log_clip_01-11.csv / queue_position_log_clip_01-11.csv
files and, for each clip, tries to identify one common discharge_start_time t0:
"the point at which the lead queued vehicle (P1) begins sustained forward
movement after a stationary or near-stationary period."

This script writes NOTHING to disk. It does not call YOLO/ByteTrack, does not
touch calibration_event_dataset.csv, cleaned_calibration_event_dataset.csv,
robust_speed_by_type_position.csv, or final_green_time_lookup_table.csv, and
does not modify main.py, traffic_calculations.py, vehicle lengths, gap value,
position grouping, effective-speed formula, outlier-cleaning, median
aggregation, or safety-buffer logic. clip_12 is untouched - it has no
crossing_log_clip_12.csv / queue_position_log_clip_12.csv, so it is naturally
excluded from load_crossing_logs()/load_queue_position_logs()'s glob pattern
(no special-casing needed).

Detection algorithm (plain English):
  1. Find the first vehicle_id that ever holds spatial_queue_position==1
     (P1) in this clip's queue_position_log, in time order.
  2. Look at THAT vehicle's own distance_to_line_b_px values over its
     logged frames (every 5th frame). Compute the frame-to-frame change
     ("delta") in that distance.
  3. Walk forward through the deltas looking for: >=T0_STABLE_MIN_STEPS
     consecutive deltas small enough to be bounding-box jitter rather than
     real motion (|delta| <= T0_STABLE_JITTER_PX), immediately followed by
     >=T0_MOVEMENT_MIN_STEPS consecutive deltas that decrease by more than
     T0_MOVEMENT_MIN_DROP_PX - i.e. a real, sustained move toward line_b.
  4. t0 = the timestamp of the first sample in that sustained-movement
     window (the first evidence of real discharge movement).
  5. The initial queue snapshot is read from the queue_position_log frame
     nearest to t0 (not a new frame - an existing logged row).
  6. Separately, flag (not fix) any clip where consecutive logged snapshots
     are more than T0_EMPTY_QUEUE_GAP_S apart with queued vehicles on both
     sides - a possible sign of a second, separate discharge event later in
     the same clip.

All thresholds are defined in build_calibration_dataset.py (module-level
constants: T0_STABLE_JITTER_PX, T0_STABLE_MIN_STEPS, T0_MOVEMENT_MIN_DROP_PX,
T0_MOVEMENT_MIN_STEPS, T0_EMPTY_QUEUE_GAP_S) - reused from there, not
duplicated, so there is exactly one place to adjust them.

Run from the backend/ directory:
    python diagnose_discharge_start.py
"""

from build_calibration_dataset import (
    T0_EMPTY_QUEUE_GAP_S,
    T0_MOVEMENT_MIN_DROP_PX,
    T0_MOVEMENT_MIN_STEPS,
    T0_STABLE_JITTER_PX,
    T0_STABLE_MIN_STEPS,
    detect_discharge_start,
    detect_possible_multiple_events,
    find_first_p1_track,
    load_crossing_logs,
    load_queue_position_logs,
)

SHOW_CONTEXT_SAMPLES = 4  # how many samples before/after the transition to print for verification


def nearest_frame_snapshot(rows_by_frame, target_time_s):
    """The queue_position_log frame (an EXISTING logged frame, not interpolated)
    whose time_s is closest to target_time_s, and all vehicles logged in it."""
    if not rows_by_frame:
        return None, []
    best_frame = min(rows_by_frame, key=lambda fn: abs(rows_by_frame[fn][0]["time_s"] - target_time_s))
    snapshot = sorted(rows_by_frame[best_frame], key=lambda r: r["position"])
    return best_frame, snapshot


def print_context_window(track_series, transition_index):
    lo = max(0, transition_index - SHOW_CONTEXT_SAMPLES)
    hi = min(len(track_series), transition_index + SHOW_CONTEXT_SAMPLES)
    print("    time_s   frame  distance_to_line_b_px")
    for i in range(lo, hi):
        r = track_series[i]
        marker = "  <-- t0 (first sustained-movement sample)" if i == transition_index else ""
        before_after = "before" if i < transition_index else ("at/after" if marker else "at/after")
        print(f"    {r['time_s']:>6.3f}  {r['frame_number']:>5d}  {r['distance_to_line_b_px']:>8.2f}   [{before_after}]{marker}")


if __name__ == "__main__":
    crossing_logs = load_crossing_logs()
    queue_logs = load_queue_position_logs()

    if not queue_logs:
        print("No queue_position_log_clip_*.csv files found. Nothing to diagnose.")
        raise SystemExit(1)

    print("=" * 78)
    print("DISCHARGE-START (t0) DETECTION - DRY RUN / DIAGNOSTIC ONLY")
    print("No files are written by this script.")
    print("=" * 78)
    print(f"\nConfigurable thresholds (defined in build_calibration_dataset.py):")
    print(f"  T0_STABLE_JITTER_PX      = {T0_STABLE_JITTER_PX}  (per-step px change treated as jitter/stationary)")
    print(f"  T0_STABLE_MIN_STEPS      = {T0_STABLE_MIN_STEPS}  (consecutive stable steps required)")
    print(f"  T0_MOVEMENT_MIN_DROP_PX  = {T0_MOVEMENT_MIN_DROP_PX}  (per-step px decrease required to count as real movement)")
    print(f"  T0_MOVEMENT_MIN_STEPS    = {T0_MOVEMENT_MIN_STEPS}  (consecutive moving steps required)")
    print(f"  T0_EMPTY_QUEUE_GAP_S     = {T0_EMPTY_QUEUE_GAP_S}  (gap between logged snapshots that flags a possible second event)")

    summary_rows = []
    manual_review_clips = []

    for clip_key in sorted(queue_logs):
        clip = queue_logs[clip_key]
        rows_by_frame = clip["rows_by_frame"]
        rows_by_vehicle = clip["rows_by_vehicle"]

        print("\n" + "-" * 78)
        print(f"CLIP: {clip_key}  (video_name field: {clip['video_name']})")
        print("-" * 78)

        p1_id, p1_type = find_first_p1_track(rows_by_frame)
        if p1_id is None:
            print("  No vehicle ever recorded at spatial_queue_position==1 in this clip's log.")
            summary_rows.append((clip_key, None, None, "N/A", 0, "LOW", "N/A"))
            manual_review_clips.append((clip_key, "no P1 row found at all"))
            continue

        track_series = rows_by_vehicle[p1_id]
        print(f"  P1 track_id: {p1_id}   vehicle_type: {p1_type}   logged samples: {len(track_series)}")

        result = detect_discharge_start(track_series)

        multi_event = detect_possible_multiple_events(rows_by_vehicle)

        if result["t0_time_s"] is None:
            print(f"  Candidate t0: COULD NOT BE RELIABLY INFERRED")
            print(f"  Reason: {result['reason']}")
            print(f"  Confidence: {result['confidence']}")
            summary_rows.append((clip_key, None, None, f"{p1_id}/{p1_type}", 0, result["confidence"],
                                  "YES" if multi_event else "no"))
            manual_review_clips.append((clip_key, result["reason"]))
            if multi_event:
                print(f"  WARNING: possible multiple discharge events detected in this clip "
                      f"(gap > {T0_EMPTY_QUEUE_GAP_S}s between logged snapshots) - not separated, just flagged.")
            continue

        t0_frame = result["t0_frame"]
        t0_time = result["t0_time_s"]
        confidence = result["confidence"]
        transition_index = result["transition_index"]

        print(f"  Candidate discharge_start_frame: {t0_frame}")
        print(f"  Candidate discharge_start_time_s: {t0_time:.3f}")
        print(f"  Confidence: {confidence}")

        print(f"\n  P1 distance_to_line_b_px around t0 (own logged samples):")
        print_context_window(track_series, transition_index)

        snapshot_frame, snapshot_rows = nearest_frame_snapshot(rows_by_frame, t0_time)
        print(f"\n  Nearest queue_position_log snapshot to t0: frame {snapshot_frame} "
              f"(time_s={rows_by_frame[snapshot_frame][0]['time_s']:.3f})")
        print(f"  Initial queue membership at that snapshot ({len(snapshot_rows)} vehicle(s)):")
        for r in snapshot_rows:
            print(f"    {r['position_group']:6s} (pos {r['position']})  track_id={r['vehicle_id']:<6s} type={r['vehicle_type']}")

        if multi_event:
            print(f"\n  WARNING: possible multiple discharge events detected in this clip "
                  f"(gap > {T0_EMPTY_QUEUE_GAP_S}s between logged snapshots) - not separated, just flagged for manual review.")

        summary_rows.append((clip_key, t0_frame, round(t0_time, 3), f"{p1_id}/{p1_type}",
                              len(snapshot_rows), confidence, "YES" if multi_event else "no"))

        if confidence != "HIGH" or multi_event:
            reason_bits = []
            if confidence != "HIGH":
                reason_bits.append(f"confidence={confidence}")
            if multi_event:
                reason_bits.append("possible multiple discharge events")
            manual_review_clips.append((clip_key, "; ".join(reason_bits)))

    print("\n" + "=" * 78)
    print("SUMMARY TABLE")
    print("=" * 78)
    header = ("Clip", "t0 frame", "t0 seconds", "P1 ID/type", "Initial queue size", "Confidence", "Multiple event?")
    print(f"{header[0]:<10s}{header[1]:<10s}{header[2]:<12s}{header[3]:<16s}{header[4]:<20s}{header[5]:<12s}{header[6]}")
    for clip_key, t0_frame, t0_time, p1_label, qsize, confidence, multi in summary_rows:
        t0_frame_s = str(t0_frame) if t0_frame is not None else "NONE"
        t0_time_s = f"{t0_time:.3f}" if t0_time is not None else "NONE"
        print(f"{clip_key:<10s}{t0_frame_s:<10s}{t0_time_s:<12s}{p1_label:<16s}{str(qsize):<20s}{confidence:<12s}{multi}")

    print("\n" + "=" * 78)
    print("CLIPS REQUIRING MANUAL / DEBUG-VIDEO VERIFICATION")
    print("=" * 78)
    if manual_review_clips:
        for clip_key, reason in manual_review_clips:
            print(f"  {clip_key}: {reason}  -> check outputs/debug_{clip_key}.mp4")
    else:
        print("  (none - all clips produced a HIGH-confidence single-event t0 candidate)")

    print(
        "\nNOTE: this is a dry run. No calibration CSVs were created, modified, or regenerated. "
        "Nothing was integrated into build_calibration_dataset.py's normal pipeline or into runtime code."
    )
