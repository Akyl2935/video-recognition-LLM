"""
Example usage script for video object detection.
This demonstrates how to use the VideoObjectDetector class programmatically.
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.video_recognition import (
    VideoObjectDetector,
    build_output_artifact_path,
    evaluate_detections,
    create_evaluation_report,
)

def main():
    # Configuration
    input_video = "data/samples/personWalking.mp4"
    output_video = "outputs/output_video.mp4"
    model_name = "models/yolov8n.pt"  # Pre-trained YOLO model
    confidence_threshold = 0.25
    enable_tracking = False
    tracker_config = "bytetrack.yaml"
    ground_truth_file = "ground_truth.json"  # Optional
    
    # Check if input video exists
    if not os.path.exists(input_video):
        print(f"Error: Input video not found: {input_video}")
        print("Please provide a valid video file path.")
        return
    
    # Initialize detector
    print("Initializing YOLO detector...")
    detector = VideoObjectDetector(
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        enable_tracking=enable_tracking,
        tracker_config=tracker_config,
    )
    
    # Process video
    print(f"\nProcessing video: {input_video}")
    stats = detector.process_video(input_video, output_video, save_detections=True)
    
    # Print statistics
    print("\n" + "="*80)
    print("PROCESSING STATISTICS")
    print("="*80)
    print(f"Total frames: {stats['total_frames']}")
    print(f"Total detections: {stats['total_detections']}")
    print(f"Average detections per frame: {stats['avg_detections_per_frame']:.2f}")
    print(f"Video resolution: {stats['resolution'][0]}x{stats['resolution'][1]}")
    print(f"FPS: {stats['fps']}")
    if stats.get("tracking_enabled"):
        print(f"Unique tracks: {stats['unique_tracks']}")
    
    # Evaluate if ground truth is available
    detections_path = build_output_artifact_path(output_video, '_detections.json')
    if os.path.exists(ground_truth_file):
        if not os.path.exists(detections_path):
            print(f"\nDetections file not found: {detections_path}")
            print("Skipping evaluation.")
            return
        print(f"\nEvaluating with ground truth: {ground_truth_file}")
        evaluation_results = evaluate_detections(
            detections_path,
            ground_truth_file,
            iou_threshold=0.5
        )
        
        # Create evaluation report
        report_path = build_output_artifact_path(output_video, '_evaluation_report.txt')
        create_evaluation_report(evaluation_results, report_path)
        
        # Print summary
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        print(f"Mean IOU: {evaluation_results['mean_iou']:.4f}")
        print(f"Median IOU: {evaluation_results['median_iou']:.4f}")
        print(f"Overall Precision: {evaluation_results['overall_precision']:.4f}")
        print(f"Overall Recall: {evaluation_results['overall_recall']:.4f}")
        print(f"Overall F1: {evaluation_results['overall_f1_score']:.4f}")
        print(f"Total matches: {evaluation_results['total_matches']}")
        print(f"Detailed report saved to: {report_path}")
    else:
        print(f"\nGround truth file not found: {ground_truth_file}")
        print("To create ground truth annotations, run:")
        print(
            f"  python scripts/annotate_ground_truth.py --video {input_video} "
            f"--output {ground_truth_file}"
        )
    
    print(f"\nOutput video saved to: {output_video}")
    print("Processing complete!")

if __name__ == '__main__':
    main()
