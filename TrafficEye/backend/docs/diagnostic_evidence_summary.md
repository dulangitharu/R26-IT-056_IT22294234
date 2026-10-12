# Diagnostic Evidence Summary

This document summarizes the diagnostic testing performed on the TrafficEye
vehicle detection/tracking pipeline before finalizing the calibrated dataset,
for panel/report reference.

## 1. Truck Threshold Test Summary

`CONF_THRESHOLD` values of 0.25 and 0.30 were tested (via a 3-clip pilot on
clip_05, clip_06, and clip_09, which had the most Truck activity) as an
attempt to improve Truck detection recall, after diagnostics showed the
model correctly labels trucks as "Truck" but often at confidence scores
below the default 0.40 threshold.

- **CONF_THRESHOLD=0.25** was rejected: Truck crossings did not improve
  reliably across the pilot clips (flat overall, and worse in one clip),
  while other vehicle classes changed abnormally (Bus crossings doubled,
  Car crossings rose ~18%, Tuktuk crossings dropped ~36%).
- **CONF_THRESHOLD=0.30** was also rejected: although Truck crossings
  increased slightly, Car/Bus/Tuktuk counts still changed abnormally
  (Bus again doubled, matching the 0.25 result almost exactly), and
  visual inspection of the debug videos confirmed track-ID
  fragmentation/double-counting - the same physical vehicle was seen
  assigned two different track IDs, with one incorrectly logged as a
  separate crossing.
- **Final decision:** the calibrated system uses the stable
  `CONF_THRESHOLD=0.40`.

**Report wording:**

> Truck confidence threshold tuning was tested using CONF_THRESHOLD values
> of 0.25 and 0.30. However, lower thresholds caused abnormal changes in
> other vehicle classes and track-ID fragmentation, while Truck detection
> did not improve reliably. Therefore, the final calibrated system uses the
> stable CONF_THRESHOLD=0.40. Truck low-confidence detection is documented
> as a model limitation.

## 2. Ambulance / clip_12 Summary

An additional clip (clip_12.mp4) containing a van-style ambulance was
tested against the trained model, including after a retraining attempt
intended to fix ambulance classification.

- The model did not reliably detect the van-style ambulance as
  "Ambulance" - it was consistently classified as "Car" instead, both
  before and after retraining, across every sample frame and confidence
  level tested.
- **Final decision:** clip_12 is excluded from the final calibrated
  dataset.

**Report wording:**

> An additional ambulance clip was tested, but the current model did not
> reliably classify the van-style ambulance as Ambulance. To avoid
> incorrect calibration results, clip_12 was excluded from the final
> calibrated dataset. Ambulance detection is documented as a model
> limitation.

## 3. Final Stable Setup

- Final calibration uses **clip_01.mp4 to clip_11.mp4 only**.
- Final `CONF_THRESHOLD` is **0.40**.
- Final numerical results are based on the stable CSV logs
  (`outputs/crossing_log_clip_01.csv` - `crossing_log_clip_11.csv`,
  `outputs/all_clips_crossing_summary.csv`,
  `outputs/average_passing_time.csv`).
- Debug videos (`outputs/debug_clip_01.mp4` - `debug_clip_11.mp4`) are
  visual references only, for verifying tracking/ROI/line behavior - not
  the source of truth for reported counts.
