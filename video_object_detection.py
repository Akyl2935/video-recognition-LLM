"""
Video Object Detection using YOLO
This script processes a video file to detect objects (people, cars, etc.) and marks them with bounding boxes.
It also compares detection results with ground truth data using Intersection over Union (IOU).
"""

import cv2
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict
from ultralytics import YOLO
import matplotlib.pyplot as plt
from collections import defaultdict


class VideoObjectDetector:
    """Class for detecting objects in video using YOLO and evaluating with IOU."""
    
    def __init__(
        self,
        model_name: str = 'yolov8n.pt',
        confidence_threshold: float = 0.25,
        detection_class: int = 0,
    ):
        """
        Initialize the detector with a YOLO model.
        
        Args:
            model_name: Name of the YOLO model file (e.g., 'yolov8n.pt')
            confidence_threshold: Minimum confidence for detections
        """
        self.model = YOLO(model_name)
        self.confidence_threshold = confidence_threshold
        self.detection_class = detection_class  # default 0 = person in COCO dataset
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
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            for i in range(len(boxes)):
                box = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
                confidence = float(boxes.conf[i].cpu().numpy())
                class_id = int(boxes.cls[i].cpu().numpy())
                
                # Filter by class (0 = person)
                if class_id == self.detection_class:
                    detections.append({
                        'bbox': box.tolist(),  # [x1, y1, x2, y2]
                        'confidence': confidence,
                        'class_id': class_id
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
            
            # Draw rectangle
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label with confidence
            label = f"{self.detection_label} {confidence:.2f}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
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
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_video_path}")
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
        
        all_detections = {}
        frame_count = 0
        total_detections = 0
        
        print(f"Processing video: {input_video_path}")
        print(f"Total frames: {total_frames}, FPS: {fps}, Resolution: {frame_width}x{frame_height}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect objects
            detections = self.detect_objects_in_frame(frame)
            total_detections += len(detections)
            
            # Save detections for this frame
            if save_detections:
                all_detections[frame_count] = detections
            
            # Draw bounding boxes
            annotated_frame = self.draw_bounding_boxes(frame, detections)
            
            # Write frame to output video
            out.write(annotated_frame)
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count}/{total_frames} frames...")
        
        cap.release()
        out.release()
        
        # Save detections to JSON
        if save_detections:
            detections_path = output_video_path.replace('.mp4', '_detections.json')
            with open(detections_path, 'w') as f:
                json.dump(all_detections, f, indent=2)
            print(f"Detections saved to: {detections_path}")
        
        stats = {
            'total_frames': frame_count,
            'total_detections': total_detections,
            'avg_detections_per_frame': total_detections / frame_count if frame_count > 0 else 0,
            'fps': fps,
            'resolution': (frame_width, frame_height)
        }
        
        print(f"\nProcessing complete!")
        print(f"Total frames: {frame_count}")
        print(f"Total detections: {total_detections}")
        print(f"Average detections per frame: {stats['avg_detections_per_frame']:.2f}")
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
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
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
    # Load detections
    with open(detections_path, 'r') as f:
        detections = json.load(f)
    
    # Load ground truth
    with open(ground_truth_path, 'r') as f:
        ground_truth = json.load(f)
    
    all_ious = []
    frame_metrics = {}
    
    for frame_num_str, frame_detections in detections.items():
        frame_num = int(frame_num_str)
        
        if str(frame_num) not in ground_truth:
            continue
        
        gt_boxes = ground_truth[str(frame_num)]
        matched_ious, match_flags = match_detections_to_ground_truth(
            frame_detections, gt_boxes, iou_threshold
        )
        
        all_ious.extend(matched_ious)
        
        # Calculate metrics for this frame
        true_positives = len(matched_ious)
        false_positives = sum(1 for flag in match_flags if not flag)
        false_negatives = len(gt_boxes) - true_positives
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        frame_metrics[frame_num] = {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'avg_iou': np.mean(matched_ious) if matched_ious else 0,
            'matched_ious': matched_ious
        }
    
    # Overall metrics
    overall_metrics = {
        'mean_iou': np.mean(all_ious) if all_ious else 0,
        'median_iou': np.median(all_ious) if all_ious else 0,
        'min_iou': np.min(all_ious) if all_ious else 0,
        'max_iou': np.max(all_ious) if all_ious else 0,
        'total_matches': len(all_ious),
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
    parser.add_argument('--output', type=str, default='output_video.mp4', help='Output video file path')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='YOLO model file')
    parser.add_argument('--confidence', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--ground-truth', type=str, default=None, help='Path to ground truth JSON file')
    parser.add_argument('--iou-threshold', type=float, default=0.5, help='IOU threshold for matching')
    parser.add_argument('--class-id', type=int, default=0,
                        help='COCO class ID to detect (e.g., 0=person, 2=car, 7=truck)')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = VideoObjectDetector(
        model_name=args.model,
        confidence_threshold=args.confidence,
        detection_class=args.class_id,
    )
    
    # Process video
    stats = detector.process_video(args.input, args.output)
    
    # Evaluate if ground truth is provided
    if args.ground_truth:
        detections_path = args.output.replace('.mp4', '_detections.json')
        if os.path.exists(detections_path):
            evaluation_results = evaluate_detections(
                detections_path, args.ground_truth, args.iou_threshold
            )
            report_path = args.output.replace('.mp4', '_evaluation_report.txt')
            create_evaluation_report(evaluation_results, report_path)
        else:
            print(f"Warning: Detections file not found: {detections_path}")


if __name__ == '__main__':
    main()

