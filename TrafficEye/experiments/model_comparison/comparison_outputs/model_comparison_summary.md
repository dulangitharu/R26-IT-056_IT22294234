# YOLOv5 vs YOLOv8 Model Comparison — TrafficEye

## Purpose

This document compares a YOLOv5n baseline against the YOLOv8n model used in the final TrafficEye pipeline, following a fair retrain of YOLOv8n under matched dataset and training conditions. YOLOv5 was trained solely as a comparison baseline to validate the model selection decision; it is not part of the production pipeline.

## Training Configuration

Both models were trained under matched conditions for a fair comparison:

| Setting     | YOLOv5n     | YOLOv8n     |
|-------------|-------------|-------------|
| Dataset     | Roboflow Vehicle-detection v2 | Roboflow Vehicle-detection v2 |
| Base weights| yolov5n.pt  | yolov8n.pt  |
| Epochs      | 50          | 50          |
| Image size  | 640         | 640         |
| Batch size  | 16          | 16          |

Both models were trained on the same Roboflow Vehicle-detection v2 dataset, using identical epochs (50), image size (640), and batch size (16), so the comparison reflects architecture differences rather than differences in data or training setup.

## Results

| Metric      | YOLOv5n | YOLOv8n | Higher |
|-------------|---------|---------|--------|
| Precision   | 0.769   | 0.802   | YOLOv8n |
| Recall      | 0.796   | 0.794   | YOLOv5n (by 0.002) |
| mAP50       | 0.836   | 0.845   | YOLOv8n |
| mAP50-95    | 0.537   | 0.594   | YOLOv8n |

Evidence for the YOLOv5n run is available in [`experiments/model_comparison/yolov5/evidence/`](../yolov5/evidence/). Evidence for the fair-retrained YOLOv8n run is available in [`experiments/model_comparison/yolov8/`](../yolov8/).

## Honest Interpretation

After this fair retrain, YOLOv8n achieved higher Precision, mAP50, and mAP50-95 than YOLOv5n. YOLOv5n's only advantage was Recall, and only by a very small margin of 0.002 (0.796 vs 0.794) — effectively a tie.

This is a meaningful shift from the earlier (non-fair) comparison: with matched dataset and training settings, YOLOv8n now leads in three of the four metrics, with the fourth being a negligible difference.

## Why YOLOv8 Was Selected for the Final Pipeline

YOLOv8n was selected as the final TrafficEye model because it performed better in most key metrics (Precision, mAP50, and mAP50-95) after the fair retrain, and because its stronger mAP50-95 reflects better **bounding box localization performance**, which matters most for the downstream pipeline:

- **ROI filtering** — objects are filtered by whether their bounding box falls inside a defined region of interest. More precisely localized boxes (higher mAP50-95) reduce incorrect inclusion/exclusion at ROI boundaries.
- **Object tracking** — frame-to-frame tracking relies on consistent, tightly fitted bounding boxes to correctly associate detections across frames. Better localization reduces the risk of ID switches and track fragmentation.
- **Crossing-line logging** — vehicle counts are logged when a bounding box crosses a defined line. Precise box edges are essential for the crossing event to be triggered at the correct time and only once per vehicle.

YOLOv5n's Recall advantage of 0.002 is not large enough to outweigh YOLOv8n's broader lead across Precision, mAP50, and mAP50-95, so YOLOv8n remains the stronger overall choice for TrafficEye.

## Conclusion

YOLOv5n remains documented as a valid baseline, essentially tied with YOLOv8n on Recall. YOLOv8n is retained as the final selected model for TrafficEye due to its higher Precision, mAP50, and mAP50-95, and its localization stability, which better serves the pipeline's ROI, tracking, and crossing-line requirements. No further training was performed for this update, and no production model, weights, logs, calibration outputs, or ROI files were modified.
