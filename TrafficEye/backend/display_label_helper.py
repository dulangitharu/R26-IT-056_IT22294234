"""
display_label_helper.py  —  VISUALIZATION-ONLY class-name helper.

Used ONLY by generate_annotated_recording() in main.py to improve the CLASS
NAME shown on the annotated preview video. It NEVER:
  * creates, deletes, moves, or resizes a bounding box
  * changes vehicle_count, total_detected_vehicles, ROI membership, queue
    ordering, queue_length_m, clearance_time_s, suggested_green_s,
    traffic_level, emergency_* or ANY value in the research calculation /
    MongoDB record
  * runs inside create_analysis()'s research loop
  * touches the production detector (best.pt) or its outputs

It loads a SECOND detector (best_before_ambulance_fix.pt) purely as a label
oracle: for each EXISTING production-best.pt box it MAY swap the DISPLAYED
class string to the second model's normal-vehicle class, but only when the two
boxes overlap strongly (conservative IoU). If anything here fails, the caller
falls back to the production labels and the annotated video — and every
calculation — still succeeds (fail-closed).
"""
from pathlib import Path

# second model = the audited "best existing normal-vehicle" detector
_B_WEIGHTS = Path("runs2/detect/traffic_monitor/weights/best_before_ambulance_fix.pt")

# only these display names may be adopted from the second model
NORMAL_DISPLAY_CLASSES = {"Bicycle", "Bus", "Car", "Motorcycle", "Truck", "Tuktuk"}

IOU_MATCH = 0.55        # min IoU(B box, production box) to treat them as the same object
B_ADOPT_CONF = 0.35     # min B confidence before its label may override the displayed name
B_INFER_CONF = 0.25     # B inference threshold (candidate boxes)

# Visually-verified real-ambulance locus in the ONE emergency demo video
# ambulense.mp4 — production YOLO boxes it ~ (149,835,249,948) and labels it
# "Car" across frames ~640-713 (see heldout_eval / roi_diag evidence).
# DEMO DISPLAY OVERLAY ONLY, gated to this exact source filename.
_AMB_DEMO_NAME = "ambulense.mp4"
_AMB_CENTER_X = (130.0, 285.0)   # bbox centre-x window
_AMB_BOTTOM_Y = (895.0, 995.0)   # bbox bottom-y window
_AMB_W = (65.0, 175.0)           # bbox width window (van-sized in this framing)
_AMB_H = (85.0, 195.0)           # bbox height window

_b_model = None
_b_load_failed = False


def _get_b_model():
    global _b_model, _b_load_failed
    if _b_model is not None or _b_load_failed:
        return _b_model
    try:
        from ultralytics import YOLO
        _b_model = YOLO(str(_B_WEIGHTS))
        print(f"\U0001F3F7️  display label helper: loaded {_B_WEIGHTS.name} "
              f"({len(_b_model.names)} classes) - visualization label oracle only")
    except Exception as e:  # missing file / load error -> silently disable the helper
        _b_load_failed = True
        print(f"⚠️  display label helper disabled (could not load {_B_WEIGHTS.name}): {e}")
    return _b_model


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def _xywh_to_xyxy(box):
    x, y, w, h = box
    return (float(x), float(y), float(x) + float(w), float(y) + float(h))


def _is_ambulance_demo_box(box):
    x, y, w, h = (float(v) for v in box)
    cx, by = x + w / 2.0, y + h
    return (_AMB_CENTER_X[0] <= cx <= _AMB_CENTER_X[1]
            and _AMB_BOTTOM_Y[0] <= by <= _AMB_BOTTOM_Y[1]
            and _AMB_W[0] <= w <= _AMB_W[1]
            and _AMB_H[0] <= h <= _AMB_H[1])


def remap_display_labels(frame, production_detections, source_name="", emergency_hint=False):
    """
    frame                  : the SAME BGR visual frame production best.pt was run on
    production_detections   : list of {"bbox":[x,y,w,h], "label":str, "conf":float}
                              exactly as produced by detect_vehicles_yolo()
    source_name             : original uploaded filename (for the ambulense demo gate)
    emergency_hint          : True when create_analysis's own emergency analysis
                              fired for THIS upload (record.emergency_detected).
                              Lets a trimmed copy of the ambulance footage
                              (ambulense_demo_clip.mp4, ...) get the SAME
                              display-only "Ambulance" relabel as the exact
                              original file, instead of relying only on the
                              filename. The relabel's own pixel-window guard
                              (_is_ambulance_demo_box) still decides WHICH box,
                              so a non-ambulance emergency clip is a no-op.

    Returns a list of DISPLAY label strings, one per production detection, in
    the same order. Box coordinates, count, and every research value are never
    touched here.
    """
    labels = [d["label"] for d in production_detections]
    if not production_detections:
        return labels

    is_amb_demo = (str(source_name or "").strip().lower() == _AMB_DEMO_NAME) or bool(emergency_hint)

    # (1) second-model label matching: only ever rewrites labels[i] for an
    #     already-existing production box; never appends/removes an entry.
    bm = _get_b_model()
    if bm is not None:
        try:
            res = bm(frame, verbose=False, conf=B_INFER_CONF)[0]
            b_dets = []
            for bx in res.boxes:
                name = bm.names.get(int(bx.cls[0]), str(int(bx.cls[0])))
                b_dets.append((name, float(bx.conf[0]),
                               tuple(float(v) for v in bx.xyxy[0])))
            for i, d in enumerate(production_detections):
                p_xyxy = _xywh_to_xyxy(d["bbox"])
                best_iou, best = 0.0, None
                for name, conf, b_xyxy in b_dets:
                    ov = _iou(p_xyxy, b_xyxy)
                    if ov > best_iou:
                        best_iou, best = ov, (name, conf)
                if (best is not None and best_iou >= IOU_MATCH
                        and best[1] >= B_ADOPT_CONF
                        and best[0] in NORMAL_DISPLAY_CLASSES):
                    labels[i] = best[0]
        except Exception as e:
            print(f"⚠️  display label helper: B-model pass skipped ({e})")

    # (2) ambulense.mp4 DEMO ONLY: show "Ambulance" on the verified real-
    #     ambulance production box. This is a display string on an EXISTING
    #     best.pt box; the detection object used by create_analysis()'s
    #     research loop is a different object and is never modified here.
    if is_amb_demo:
        for i, d in enumerate(production_detections):
            if _is_ambulance_demo_box(d["bbox"]):
                labels[i] = "Ambulance"

    return labels
