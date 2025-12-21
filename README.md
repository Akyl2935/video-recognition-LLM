# Video Object Detection using YOLO

This project implements a video object detection system using YOLO (You Only Look Once) to detect and track objects (primarily people) in video files. The system marks detected objects with bounding boxes and evaluates detection accuracy using Intersection over Union (IOU) metrics.

## Features

- **Object Detection**: Uses pre-trained YOLO models (YOLOv8) to detect objects in video frames
- **Bounding Box Visualization**: Draws bounding boxes around detected objects in the output video
- **Ground Truth Comparison**: Compares detection results with manually annotated ground truth data
- **IOU Evaluation**: Calculates Intersection over Union (IOU) metrics for accuracy assessment
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

3. The YOLO model will be automatically downloaded on first use (e.g., `yolov8n.pt`)

## Usage

### Option 1: Using the Python Script

#### 1. Process a video with object detection:
```bash
python video_object_detection.py --input input_video.mp4 --output output_video.mp4
```

#### 2. With ground truth evaluation:
```bash
python video_object_detection.py --input input_video.mp4 --output output_video.mp4 --ground-truth ground_truth.json
```

#### Command-line Arguments:
- `--input`: Path to input video file (required)
- `--output`: Path to output video file (default: `output_video.mp4`)
- `--model`: YOLO model file (default: `yolov8n.pt`)
- `--confidence`: Confidence threshold (default: 0.25)
- `--class-id`: COCO class ID to detect (default: 0 = person). Use `2` for cars.
- `--ground-truth`: Path to ground truth JSON file (optional)
- `--iou-threshold`: IOU threshold for matching (default: 0.5)

### Option 2: Using the Jupyter Notebook

1. Open `video_object_detection.ipynb` in Jupyter Notebook
2. Update the configuration section with your video file path
3. Run all cells sequentially

### Creating Ground Truth Annotations

If your video doesn't have reference annotations, you can manually create them using the annotation tool:

```bash
python annotate_ground_truth.py --video input_video.mp4 --output ground_truth.json --frame 0
```

**Annotation Tool Controls:**
- **Click and drag**: Draw bounding boxes
- **'s'**: Save annotations for current frame
- **'d'**: Delete last box
- **'c'**: Clear all boxes
- **'q'**: Quit without saving
- **'n'**: Go to next frame
- **'p'**: Go to previous frame

**Note**: You should annotate at least 3 frames as specified in the requirements.

## Output Files

After processing, the following files will be generated:

1. **Output Video** (`output_video.mp4`): Input video with bounding boxes drawn around detected objects
2. **Detections JSON** (`output_video_detections.json`): All detection results in JSON format
3. **Evaluation Report** (`output_video_evaluation_report.txt`): Detailed evaluation metrics (if ground truth is provided)

## Ground Truth Format

Ground truth annotations should be in JSON format:
```json
{
  "0": [[x1, y1, x2, y2], [x1, y1, x2, y2], ...],
  "10": [[x1, y1, x2, y2], ...],
  "20": [[x1, y1, x2, y2], ...]
}
```

Where:
- Keys are frame numbers (as strings)
- Values are arrays of bounding boxes
- Each bounding box is `[x1, y1, x2, y2]` (top-left and bottom-right coordinates)

## YOLO Models

The project uses pre-trained YOLO models from Ultralytics. Available models:
- `yolov8n.pt` - Nano (fastest, smallest)
- `yolov8s.pt` - Small
- `yolov8m.pt` - Medium
- `yolov8l.pt` - Large
- `yolov8x.pt` - Extra Large (most accurate)

Models are automatically downloaded on first use. The default model (`yolov8n.pt`) is recommended for speed and is sufficient for most use cases.

## Detection Classes

By default, the system detects **people** (class 0 in COCO dataset). To detect other objects, pass the desired class via `--class-id`/`-class-id` when running `video_object_detection.py` (e.g., `--class-id 2` for cars). Common COCO classes include:
- 0: person
- 2: car
- 3: motorcycle
- 5: bus
- 7: truck
- etc.

## Evaluation Metrics

The system calculates the following metrics:

- **IOU (Intersection over Union)**: Measures overlap between detected and ground truth boxes
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1 Score**: Harmonic mean of precision and recall

## Example Workflow

1. **Prepare your video**: Place your input video file in the project directory

2. **Create ground truth** (if needed):
   ```bash
   python annotate_ground_truth.py --video input_video.mp4 --output ground_truth.json
   ```
   Annotate at least 3 frames manually.

3. **Run detection**:
   ```bash
   python video_object_detection.py --input input_video.mp4 --output output_video.mp4 --ground-truth ground_truth.json
   ```

4. **Review results**: Check the output video and evaluation report

## Troubleshooting

### Video codec issues
If you encounter codec issues, try installing additional codecs or using a different video format.

### CUDA/GPU support
YOLO will automatically use GPU if available. For CPU-only systems, processing may be slower but will still work.

### Memory issues
For very long videos, consider processing in chunks or using a smaller YOLO model (e.g., `yolov8n.pt`).

## Project Structure

```
.
├── video_object_detection.py      # Main detection script
├── video_object_detection.ipynb    # Jupyter notebook version
├── annotate_ground_truth.py        # Ground truth annotation tool
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## References

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [YOLO Models](https://github.com/ultralytics/ultralytics#models)
- [Video Object Segmentation Datasets](https://github.com/xiaobai1217/Awesome-Video-Datasets)

## License

This project is provided as-is for educational and research purposes.



