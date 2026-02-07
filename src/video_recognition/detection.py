"""
Video Object Detection using YOLO
This script processes a video file to detect objects (people, cars, etc.) and marks them with bounding boxes.
It also compares detection results with ground truth data using Intersection over Union (IOU).
"""

import cv2
import numpy as np
import json
from pathlib import Path
from typing import Any, List, Tuple, Dict


DEFAULT_VIDEO_FPS = 30.0


def build_output_artifact_path(output_video_path: str, artifact_suffix: str) -> str:
    """
    Build a sidecar artifact path from the output video path.

    Example:
      my/video.mov + "_detections.json" -> my/video_detections.json
    """
    video_path = Path(output_video_path)
    if video_path.suffix:
        base_path = video_path.with_suffix("")
    else:
        base_path = video_path
    return str(base_path.parent / f"{base_path.name}{artifact_suffix}")


def normalize_frame_map(frame_map: Dict[str, Any], source_name: str) -> Dict[int, Any]:
    """Convert JSON frame keys to integers with clear validation errors."""
    normalized = {}
    for frame_key, frame_value in frame_map.items():
        try:
            frame_number = int(frame_key)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid frame key '{frame_key}' in {source_name}. "
                "Expected integer-like frame numbers as JSON keys."
            ) from exc
        normalized[frame_number] = frame_value
    return normalized


def track_color(track_id: int) -> Tuple[int, int, int]:
    """Generate a stable BGR color for a track ID."""
    return (
        (37 * track_id) % 256,
        (17 * track_id + 89) % 256,
        (97 * track_id + 43) % 256,
    )


class VideoObjectDetector:
    """Class for detecting objects in video using YOLO and evaluating with IOU."""
    
    def __init__(
        self,
        model_name: str = 'models/yolov8n.pt',
        confidence_threshold: float = 0.25,
        detection_class: int = 0,
        enable_tracking: bool = False,
        tracker_config: str = 'bytetrack.yaml',
    ):
        """
        Initialize the detector with a YOLO model.
        
        Args:
            model_name: Name of the YOLO model file (e.g., 'yolov8n.pt')
            confidence_threshold: Minimum confidence for detections
        """
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError(
                f"confidence_threshold must be between 0 and 1, got {confidence_threshold}"
            )
        if detection_class < 0:
            raise ValueError(f"detection_class must be >= 0, got {detection_class}")
        if enable_tracking and not tracker_config:
            raise ValueError("tracker_config is required when tracking is enabled")

        # Keep ultralytics optional for utility usage (evaluation/report helpers).
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.detection_class = detection_class  # default 0 = person in COCO dataset
        self.enable_tracking = enable_tracking
        self.tracker_config = tracker_config
        # Store a readable label for annotations
        model_names = getattr(self.model, 'names', {}) or {}
        self.detection_label = model_names.get(self.detection_class, f"Class {self.detection_class}")
        
    def detect_objects_in_frame(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in a single frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of detections, each containing bbox, confidence, and class
        """
        if self.enable_tracking:
            results = self.model.track(
                frame,
                conf=self.confidence_threshold,
                persist=True,
                tracker=self.tracker_config,
                verbose=False,
            )
        else:
            results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            track_ids = None
            if self.enable_tracking and boxes.id is not None:
                track_ids = boxes.id.int().cpu().numpy().tolist()
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                track_id = int(track_ids[i]) if track_ids is not None else None
                
                # Filter by class (0 = person)
                if class_id == self.detection_class:
                    detections.append({
                        'bbox': box.tolist(),  # [x1, y1, x2, y2]
                        'confidence': confidence,
                        'class_id': class_id,
                        'track_id': track_id,
                    })
        
        return detections
    
    def draw_bounding_boxes(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        Draw bounding boxes on a frame.
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries
            
        Returns:
            Frame with bounding boxes drawn
        """
        annotated_frame = frame.copy()
        
        for det in detections:
            bbox = det['bbox']
            x1, y1, x2, y2 = map(int, bbox)
            confidence = det['confidence']
            track_id = det.get('track_id')
            color = (0, 255, 0)
            if track_id is not None:
                color = track_color(track_id)
            
            # Draw rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label with confidence
            label = f"{self.detection_label} {confidence:.2f}"
            if track_id is not None:
                label = f"{self.detection_label} #{track_id} {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            label_top = max(0, y1 - label_size[1] - 10)
            label_bottom = max(label_size[1] + 4, y1)
            cv2.rectangle(
                annotated_frame,
                (x1, label_top),
                (x1 + label_size[0], label_bottom),
                color,
                -1,
            )
            text_y = max(label_size[1] + 2, y1 - 5)
            cv2.putText(
                annotated_frame,
                label,
                (x1, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                2,
            )
        
        return annotated_frame
    
    def process_video(self, input_video_path: str, output_video_path: str, 
                     save_detections: bool = True) -> Dict:
        """
        Process entire video and save output with bounding boxes.
        
        Args:
            input_video_path: Path to input video
            output_video_path: Path to save output video
            save_detections: Whether to save detection data to JSON
            
        Returns:
            Dictionary with processing statistics
        """
        cap = cv2.VideoCapture(input_video_path)
        out = None

        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_video_path}")

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = DEFAULT_VIDEO_FPS
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output_path = Path(output_video_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))
        if not out.isOpened():
            raise ValueError(f"Could not create output video file: {output_video_path}")

        all_detections: Dict[str, List[Dict[str, Any]]] = {}
        frame_count = 0
        total_detections = 0
        track_ids_seen = set()

        print(f"Processing video: {input_video_path}")
        print(
            f"Total frames: {total_frames}, FPS: {fps:.2f}, "
            f"Resolution: {frame_width}x{frame_height}, Tracking: {self.enable_tracking}"
        )

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Detect objects
                detections = self.detect_objects_in_frame(frame)
                total_detections += len(detections)
                if self.enable_tracking:
                    for det in detections:
                        if det.get('track_id') is not None:
                            track_ids_seen.add(det['track_id'])

                # Save detections for this frame
                if save_detections:
                    all_detections[str(frame_count)] = detections

                # Draw bounding boxes
                annotated_frame = self.draw_bounding_boxes(frame, detections)

                # Write frame to output video
                out.write(annotated_frame)

                frame_count += 1
                if frame_count % 30 == 0:
                    print(f"Processed {frame_count}/{total_frames} frames...")
        finally:
            cap.release()
            if out is not None:
                out.release()

        # Save detections to JSON
        if save_detections:
            detections_path = build_output_artifact_path(output_video_path, '_detections.json')
            with open(detections_path, 'w', encoding='utf-8') as f:
                json.dump(all_detections, f, indent=2)
            print(f"Detections saved to: {detections_path}")

        stats = {
            'total_frames': frame_count,
            'total_detections': total_detections,
            'avg_detections_per_frame': total_detections / frame_count if frame_count > 0 else 0,
            'fps': fps,
            'resolution': (frame_width, frame_height),
            'tracking_enabled': self.enable_tracking,
            'unique_tracks': len(track_ids_seen),
        }

        print(f"\nProcessing complete!")
        print(f"Total frames: {frame_count}")
        print(f"Total detections: {total_detections}")
        print(f"Average detections per frame: {stats['avg_detections_per_frame']:.2f}")
        if self.enable_tracking:
            print(f"Unique tracks: {stats['unique_tracks']}")
        print(f"Output video saved to: {output_video_path}")

        return stats


def calculate_iou(boxA: List[float], boxB: List[float]) -> float:
    """
    Calculate Intersection over Union (IOU) between two bounding boxes.
    
    Args:
        boxA: Bounding box [x1, y1, x2, y2]
        boxB: Bounding box [x1, y1, x2, y2]
        
    Returns:
        IOU value between 0 and 1
    """
    # Determine coordinates of intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # Compute area of intersection
    inter_area = max(0, xB - xA) * max(0, yB - yA)
    
    # Compute area of both bounding boxes
    boxA_area = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxB_area = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    
    # Compute IOU
    union_area = boxA_area + boxB_area - inter_area
    if union_area == 0:
        return 0.0
    
    iou = inter_area / union_area
    return iou


def match_detections_to_ground_truth(detections: List[Dict], 
                                     ground_truth: List[List[float]], 
                                     iou_threshold: float = 0.5) -> Tuple[List[float], List[bool]]:
    """
    Match detected bounding boxes to ground truth boxes using IOU.
    
    Args:
        detections: List of detection dictionaries with 'bbox' key
        ground_truth: List of ground truth bounding boxes [x1, y1, x2, y2]
        iou_threshold: Minimum IOU to consider a match
        
    Returns:
        Tuple of (matched IOUs, match flags for each detection)
    """
    if not detections or not ground_truth:
        return [], [False] * len(detections)
    
    matched_ious = []
    match_flags = [False] * len(detections)
    used_gt = [False] * len(ground_truth)
    
    # Calculate IOU for all pairs
    iou_matrix = []
    for det in detections:
        det_bbox = det['bbox']
        ious = []
        for gt_bbox in ground_truth:
            iou = calculate_iou(det_bbox, gt_bbox)
            ious.append(iou)
        iou_matrix.append(ious)
    
    # Greedy matching: match highest IOU pairs first
    while True:
        max_iou = 0
        best_det_idx = -1
        best_gt_idx = -1
        
        for det_idx, ious in enumerate(iou_matrix):
            if match_flags[det_idx]:
                continue
            for gt_idx, iou in enumerate(ious):
                if used_gt[gt_idx]:
                    continue
                if iou > max_iou:
                    max_iou = iou
                    best_det_idx = det_idx
                    best_gt_idx = gt_idx
        
        if max_iou >= iou_threshold and best_det_idx >= 0:
            matched_ious.append(max_iou)
            match_flags[best_det_idx] = True
            used_gt[best_gt_idx] = True
        else:
            break
    
    return matched_ious, match_flags


def evaluate_detections(detections_path: str, ground_truth_path: str, 
                       iou_threshold: float = 0.5) -> Dict:
    """
    Evaluate detection results against ground truth data.
    
    Args:
        detections_path: Path to JSON file with detections
        ground_truth_path: Path to JSON file with ground truth annotations
        iou_threshold: Minimum IOU to consider a match
        
    Returns:
        Dictionary with evaluation metrics
    """
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(f"iou_threshold must be between 0 and 1, got {iou_threshold}")

    # Load detections
    with open(detections_path, 'r', encoding='utf-8') as f:
        detections = json.load(f)

    # Load ground truth
    with open(ground_truth_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)

    detection_by_frame = normalize_frame_map(detections, "detections file")
    ground_truth_by_frame = normalize_frame_map(ground_truth, "ground truth file")

    all_ious: List[float] = []
    frame_metrics: Dict[int, Dict[str, Any]] = {}
    total_true_positives = 0
    total_false_positives = 0
    total_false_negatives = 0

    # Evaluate on the union of frames so missing detections on GT frames count as false negatives.
    all_frames = sorted(set(detection_by_frame) | set(ground_truth_by_frame))
    for frame_num in all_frames:
        frame_detections = detection_by_frame.get(frame_num, [])
        gt_boxes = ground_truth_by_frame.get(frame_num, [])

        matched_ious, _ = match_detections_to_ground_truth(
            frame_detections, gt_boxes, iou_threshold
        )
        all_ious.extend(matched_ious)

        # Calculate metrics for this frame
        true_positives = len(matched_ious)
        false_positives = len(frame_detections) - true_positives
        false_negatives = len(gt_boxes) - true_positives

        total_true_positives += true_positives
        total_false_positives += false_positives
        total_false_negatives += false_negatives

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        frame_metrics[frame_num] = {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'avg_iou': float(np.mean(matched_ious)) if matched_ious else 0,
            'matched_ious': matched_ious,
            'detections': len(frame_detections),
            'ground_truth_boxes': len(gt_boxes),
        }

    overall_precision = (
        total_true_positives / (total_true_positives + total_false_positives)
        if (total_true_positives + total_false_positives) > 0
        else 0
    )
    overall_recall = (
        total_true_positives / (total_true_positives + total_false_negatives)
        if (total_true_positives + total_false_negatives) > 0
        else 0
    )
    overall_f1 = (
        2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)
        if (overall_precision + overall_recall) > 0
        else 0
    )

    # Overall metrics
    overall_metrics = {
        'mean_iou': float(np.mean(all_ious)) if all_ious else 0,
        'median_iou': float(np.median(all_ious)) if all_ious else 0,
        'min_iou': float(np.min(all_ious)) if all_ious else 0,
        'max_iou': float(np.max(all_ious)) if all_ious else 0,
        'total_matches': len(all_ious),
        'total_true_positives': total_true_positives,
        'total_false_positives': total_false_positives,
        'total_false_negatives': total_false_negatives,
        'overall_precision': overall_precision,
        'overall_recall': overall_recall,
        'overall_f1_score': overall_f1,
        'evaluated_frames': len(all_frames),
        'frames_with_ground_truth': sum(
            1 for frame_num in all_frames if len(ground_truth_by_frame.get(frame_num, [])) > 0
        ),
        'frame_metrics': frame_metrics
    }

    return overall_metrics


def create_evaluation_report(evaluation_results: Dict, output_path: str):
    """
    Create a text report with evaluation results.
    
    Args:
        evaluation_results: Dictionary with evaluation metrics
        output_path: Path to save the report
    """
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("VIDEO OBJECT DETECTION EVALUATION REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("OVERALL METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Mean IOU: {evaluation_results['mean_iou']:.4f}\n")
        f.write(f"Median IOU: {evaluation_results['median_iou']:.4f}\n")
        f.write(f"Min IOU: {evaluation_results['min_iou']:.4f}\n")
        f.write(f"Max IOU: {evaluation_results['max_iou']:.4f}\n")
        f.write(f"Total Matches: {evaluation_results['total_matches']}\n\n")
        f.write(f"Evaluated Frames: {evaluation_results['evaluated_frames']}\n")
        f.write(f"Frames with Ground Truth: {evaluation_results['frames_with_ground_truth']}\n")
        f.write(f"Total True Positives: {evaluation_results['total_true_positives']}\n")
        f.write(f"Total False Positives: {evaluation_results['total_false_positives']}\n")
        f.write(f"Total False Negatives: {evaluation_results['total_false_negatives']}\n")
        f.write(f"Overall Precision: {evaluation_results['overall_precision']:.4f}\n")
        f.write(f"Overall Recall: {evaluation_results['overall_recall']:.4f}\n")
        f.write(f"Overall F1 Score: {evaluation_results['overall_f1_score']:.4f}\n\n")
        
        f.write("PER-FRAME METRICS\n")
        f.write("-" * 80 + "\n")
        for frame_num, metrics in sorted(evaluation_results['frame_metrics'].items()):
            f.write(f"\nFrame {frame_num}:\n")
            f.write(f"  True Positives: {metrics['true_positives']}\n")
            f.write(f"  False Positives: {metrics['false_positives']}\n")
            f.write(f"  False Negatives: {metrics['false_negatives']}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall: {metrics['recall']:.4f}\n")
            f.write(f"  F1 Score: {metrics['f1_score']:.4f}\n")
            f.write(f"  Average IOU: {metrics['avg_iou']:.4f}\n")
            if metrics['matched_ious']:
                f.write(f"  Matched IOUs: {[f'{iou:.3f}' for iou in metrics['matched_ious']]}\n")
    
    print(f"Evaluation report saved to: {output_path}")


def main():
    """Main function to run the video object detection pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Video Object Detection using YOLO')
    parser.add_argument('--input', type=str, required=True, help='Input video file path')
    parser.add_argument('--output', type=str, default='outputs/output_video.mp4', help='Output video file path')
    parser.add_argument('--model', type=str, default='models/yolov8n.pt', help='YOLO model file')
    parser.add_argument('--confidence', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--ground-truth', type=str, default=None, help='Path to ground truth JSON file')
    parser.add_argument('--iou-threshold', type=float, default=0.5, help='IOU threshold for matching')
    parser.add_argument('--class-id', type=int, default=0,
                        help='COCO class ID to detect (e.g., 0=person, 2=car, 7=truck)')
    parser.add_argument(
        '--track',
        action='store_true',
        help='Enable multi-object tracking and include track IDs in output',
    )
    parser.add_argument(
        '--tracker',
        type=str,
        default='bytetrack.yaml',
        help='Ultralytics tracker config (used when --track is enabled)',
    )
    
    args = parser.parse_args()

    if not 0.0 <= args.confidence <= 1.0:
        parser.error(f"--confidence must be between 0 and 1, got {args.confidence}")
    if not 0.0 <= args.iou_threshold <= 1.0:
        parser.error(f"--iou-threshold must be between 0 and 1, got {args.iou_threshold}")
    if args.class_id < 0:
        parser.error(f"--class-id must be >= 0, got {args.class_id}")
    
    # Initialize detector
    detector = VideoObjectDetector(
        model_name=args.model,
        confidence_threshold=args.confidence,
        detection_class=args.class_id,
        enable_tracking=args.track,
        tracker_config=args.tracker,
    )
    
    # Process video
    stats = detector.process_video(args.input, args.output)
    
    # Evaluate if ground truth is provided
    if args.ground_truth:
        detections_path = build_output_artifact_path(args.output, '_detections.json')
        if Path(detections_path).exists():
            evaluation_results = evaluate_detections(
                detections_path, args.ground_truth, args.iou_threshold
            )
            report_path = build_output_artifact_path(args.output, '_evaluation_report.txt')
            create_evaluation_report(evaluation_results, report_path)
        else:
            print(f"Warning: Detections file not found: {detections_path}")


if __name__ == '__main__':
    main()
