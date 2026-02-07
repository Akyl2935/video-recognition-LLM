# Video Object Detection using YOLO

This project implements a video object detection system using YOLO (You Only Look Once) to detect objects (primarily people) in video files frame-by-frame. The system marks detected objects with bounding boxes and evaluates detection accuracy using Intersection over Union (IOU) metrics.

## Features

- **Object Detection**: Uses pre-trained YOLO models (YOLOv8) to detect objects in video frames
- **Object Tracking (Optional)**: Tracks objects across frames and assigns stable track IDs
- **Bounding Box Visualization**: Draws bounding boxes around detected objects in the output video
- **Ground Truth Comparison**: Compares detection results with manually annotated ground truth data
- **IOU Evaluation**: Calculates IOU, precision, recall, and F1 metrics
- **Annotation Tool**: Interactive tool for manually creating ground truth annotations

## Requirements

- Python 3.8 or higher
- See `requirements.txt` for all dependencies

## Installation

1. Clone or download this repository
2. Install required packages:

```bash
pip install -r requirements.txt
```

3. The YOLO model will be automatically downloaded on first use (for example, `yolov8n.pt`)

## Usage

### Option 1: Use the Python script

#### 1. Process a video with object detection

```bash
python video_object_detection.py --input input_video.mp4 --output output_video.mp4
```

#### 2. Process and evaluate against ground truth

```bash
python video_object_detection.py --input input_video.mp4 --output output_video.mp4 --ground-truth ground_truth.json
```

#### 3. Process with tracking enabled

```bash
python video_object_detection.py --input input_video.mp4 --output output_video.mp4 --class-id 0 --track
```

#### Command-line arguments

- `--input`: Path to input video file (required)
- `--output`: Path to output video file (default: `output_video.mp4`)
- `--model`: YOLO model file (default: `yolov8n.pt`)
- `--confidence`: Confidence threshold in [0, 1] (default: 0.25)
- `--class-id`: COCO class ID to detect (default: 0 = person). Use `2` for cars.
- `--ground-truth`: Path to ground truth JSON file (optional)
- `--iou-threshold`: IoU threshold in [0, 1] for matching (default: 0.5)
- `--track`: Enable multi-object tracking and include track IDs in output
- `--tracker`: Ultralytics tracker config (default: `bytetrack.yaml`)

### Option 2: Use the Jupyter notebook

1. Open `video_object_detection.ipynb` in Jupyter Notebook
2. Update the configuration section with your video file path
3. Run all cells sequentially

### Create ground truth annotations

If your video does not have annotations, create them with:

```bash
python annotate_ground_truth.py --video input_video.mp4 --output ground_truth.json --frame 0
```

Annotation controls:

- Click and drag: draw bounding boxes
- `s`: save annotations for current frame and exit
- `d`: delete last box
- `c`: clear all boxes
- `q`: quit without saving
- `n`: next frame
- `p`: previous frame

## Output files

After processing, these files are generated:

1. **Output Video** (`output_video.mp4`): Input video with detections drawn
2. **Detections JSON** (`output_video_detections.json`): Per-frame detection output (includes `track_id` when tracking is enabled)
3. **Evaluation Report** (`output_video_evaluation_report.txt`): Detailed evaluation metrics (if ground truth is provided)

Sidecar files are derived from the output filename stem, so they work for non-`.mp4` output names too.

## Ground truth format

Ground truth annotations should be JSON:

```json
{
  "0": [[x1, y1, x2, y2], [x1, y1, x2, y2]],
  "10": [[x1, y1, x2, y2]],
  "20": [[x1, y1, x2, y2]]
}
```

Where:

- Keys are frame numbers (as strings)
- Values are arrays of bounding boxes
- Each box is `[x1, y1, x2, y2]` (top-left and bottom-right)

## YOLO models

The project uses pre-trained models from Ultralytics. Common options:

- `yolov8n.pt` - Nano (fastest, smallest)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (most accurate)

## Detection classes

By default, the system detects people (class `0` in COCO). To detect other objects, pass `--class-id`.

Common COCO classes:

- 0: person
- 2: car
- 3: motorcycle
- 5: bus
- 7: truck

## Evaluation metrics

The system calculates:

- **IOU (Intersection over Union)**
- **Precision** = TP / (TP + FP)
- **Recall** = TP / (TP + FN)
- **F1 Score** = harmonic mean of precision and recall

## Example workflow

1. Prepare your video in the project directory
2. Create ground truth if needed:

```bash
python annotate_ground_truth.py --video input_video.mp4 --output ground_truth.json
```

3. Run detection and evaluation:

```bash
python video_object_detection.py --input input_video.mp4 --output output_video.mp4 --ground-truth ground_truth.json
```

4. Review output video and evaluation report

## Troubleshooting

### Video codec issues

If you encounter codec issues, try installing additional codecs or changing video format.

### CUDA/GPU support

YOLO automatically uses GPU if available. CPU-only mode is supported but slower.

### Memory issues

For very long videos, process in chunks or use a smaller model (`yolov8n.pt`).

## Project structure

```text
.
|-- video_object_detection.py      # Main detection script
|-- video_object_detection.ipynb   # Jupyter notebook version
|-- annotate_ground_truth.py       # Ground truth annotation tool
|-- requirements.txt               # Python dependencies
`-- README.md                      # This file
```

## References

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [YOLO Models](https://github.com/ultralytics/ultralytics#models)
- [Video Object Segmentation Datasets](https://github.com/xiaobai1217/Awesome-Video-Datasets)

## License

This project is provided as-is for educational and research purposes.
