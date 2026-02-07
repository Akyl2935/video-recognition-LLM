# Video Object Detection using YOLO

This project performs object detection (and optional tracking) on video files using YOLOv8, writes annotated video output, and evaluates detections against ground-truth boxes using IoU-based metrics.

## Project Layout

```text
.
|-- data/
|   `-- samples/
|-- docs/
|   `-- REPORT_TEMPLATE.md
|-- examples/
|   `-- example_usage.py
|-- models/
|   `-- yolov8n.pt
|-- notebooks/
|   `-- video_object_detection.ipynb
|-- outputs/
|-- scripts/
|   |-- video_object_detection.py
|   `-- annotate_ground_truth.py
|-- src/
|   `-- video_recognition/
|       |-- detection.py
|       `-- annotation.py
|-- tests/
|   `-- test_evaluation.py
|-- requirements.txt
`-- README.md
```

## Features

- Object detection with YOLOv8
- Optional multi-object tracking (`--track`) with persistent track IDs
- Annotated output video generation
- Ground-truth annotation tool
- Evaluation metrics: IoU, precision, recall, F1
- Unit tests for evaluation and utility logic

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Run detection

```bash
python scripts/video_object_detection.py --input data/samples/personWalking.mp4 --output outputs/output_video.mp4
```

### Run detection with tracking

```bash
python scripts/video_object_detection.py --input data/samples/personWalking.mp4 --output outputs/output_video.mp4 --track
```

### Run detection with evaluation

```bash
python scripts/video_object_detection.py --input data/samples/personWalking.mp4 --output outputs/output_video.mp4 --ground-truth ground_truth.json
```

### Create ground-truth annotations

```bash
python scripts/annotate_ground_truth.py --video data/samples/personWalking.mp4 --output ground_truth.json --frame 0
```

### Run example script

```bash
python examples/example_usage.py
```

## CLI Options

Detection script (`scripts/video_object_detection.py`):

- `--input`: Input video path (required)
- `--output`: Output video path (default: `outputs/output_video.mp4`)
- `--model`: Model path (default: `models/yolov8n.pt`)
- `--confidence`: Confidence threshold in `[0,1]`
- `--class-id`: COCO class ID (default: `0`)
- `--ground-truth`: Ground-truth JSON path
- `--iou-threshold`: IoU threshold in `[0,1]`
- `--track`: Enable tracking mode
- `--tracker`: Ultralytics tracker config (default: `bytetrack.yaml`)

## Output Artifacts

Given `--output outputs/output_video.mp4`, sidecar files are written as:

- `outputs/output_video_detections.json`
- `outputs/output_video_evaluation_report.txt` (when ground truth is provided)

## Ground Truth Format

```json
{
  "0": [[x1, y1, x2, y2]],
  "10": [[x1, y1, x2, y2]]
}
```

## Tests

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## License

This project is provided as-is for educational and research purposes.
