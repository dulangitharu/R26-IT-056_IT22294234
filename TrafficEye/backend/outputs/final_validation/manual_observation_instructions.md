# Manual observed-clearance instructions (prototype observed-clearance validation)

This is a **prototype observed-clearance validation**, not an independent held-out
validation — clip_01, clip_09, and clip_11 were already part of the broader
calibration work for this camera.

For each clip below, open the annotated snapshot image, then play the
original video starting at `snapshot_time_s` and time how long it takes the
**last vehicle in the selected queue** (highest position number, marked in
the snapshot) to cross `line_b`. Do not include any vehicle that was not
already detected in the snapshot — a vehicle that joins the queue later is
NOT part of this measurement.

```
observed_clearance_s = (timestamp when that last queue vehicle crosses line_b)
                        - snapshot_time_s
```

If you cannot reliably follow the selected vehicle from the snapshot until it
crosses line_b (occlusion, ambiguity, camera cut, etc.), do not guess — mark
`measurement_status = NOT_RELIABLY_MEASURABLE` in
`final_clearance_validation.csv` and leave `observed_clearance_s` blank.

---

## clip_01.mp4
- Snapshot: `validation_clip_01_snapshot.jpg`
- Snapshot frame: 280 (fps ≈ 10.473) → snapshot_time_s = **26.736s**
- Selected queue size: 7
- Selected queue order (P1 = closest to line_b): Car, Car, Car, Motorcycle, Car, Car, Car
- Vehicle to watch: **P7 (Car)** — the last/rearmost vehicle in the snapshot
- line_b: green line drawn in the snapshot image (per `calibration/roi_config.json`, scaled to this clip's resolution)
- Watch from 26.736s onward in clip_01.mp4 until P7 (Car) crosses line_b.

## clip_09.mp4
- Snapshot: `validation_clip_09_snapshot.jpg`
- Snapshot frame: 1450 (fps ≈ 29.989) → snapshot_time_s = **48.351s**
- Selected queue size: 6
- Selected queue order (P1 = closest to line_b): Car, Car, Car, Motorcycle, Tuktuk, Motorcycle
- Vehicle to watch: **P6 (Motorcycle)** — the last/rearmost vehicle in the snapshot
- line_b: green line drawn in the snapshot image
- Watch from 48.351s onward in clip_09.mp4 until P6 (Motorcycle) crosses line_b.

## clip_11.mp4
- Snapshot: `validation_clip_11_snapshot.jpg`
- Snapshot frame: 1290 (fps ≈ 30.003) → snapshot_time_s = **42.995s**
- Selected queue size: 7
- Selected queue order (P1 = closest to line_b): Motorcycle, Car, Car, Car, Car, Car, Car
- Vehicle to watch: **P7 (Car)** — the last/rearmost vehicle in the snapshot
- line_b: green line drawn in the snapshot image
- Watch from 42.995s onward in clip_11.mp4 until P7 (Car) crosses line_b.

---

## After measuring

Fill in `observed_clearance_s` (and set `measurement_status` to `VALID`) for
each row in `final_clearance_validation.csv` that could be reliably measured,
then run:

```
python evaluate_final_clearance_validation.py
```
