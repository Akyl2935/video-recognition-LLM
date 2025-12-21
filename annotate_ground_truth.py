"""
Ground Truth Annotation Tool
This script helps manually annotate frames from a video to create ground truth data.
It allows selecting frames and drawing bounding boxes around objects.
"""

import cv2
import json
import numpy as np
from pathlib import Path


class BoundingBoxAnnotator:
    """Interactive tool for annotating bounding boxes on video frames."""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.annotations = {}
        self.current_frame_num = 0
        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.current_boxes = []
        self.temp_box = None
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for drawing bounding boxes."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.start_point = (x, y)
            self.temp_box = None
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                self.end_point = (x, y)
                self.temp_box = (self.start_point, self.end_point)
                
        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            if self.start_point and self.end_point:
                # Normalize coordinates
                x1 = min(self.start_point[0], self.end_point[0])
                y1 = min(self.start_point[1], self.end_point[1])
                x2 = max(self.start_point[0], self.end_point[0])
                y2 = max(self.start_point[1], self.end_point[1])
                
                # Only add if box has minimum size
                if abs(x2 - x1) > 10 and abs(y2 - y1) > 10:
                    self.current_boxes.append([x1, y1, x2, y2])
                self.temp_box = None
                self.start_point = None
                self.end_point = None
    
    def draw_boxes_on_frame(self, frame: np.ndarray) -> np.ndarray:
        """Draw all bounding boxes on the frame."""
        annotated_frame = frame.copy()
        
        # Draw existing boxes
        for i, box in enumerate(self.current_boxes):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Box {i+1}", (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw temporary box being drawn
        if self.temp_box:
            pt1, pt2 = self.temp_box
            cv2.rectangle(annotated_frame, pt1, pt2, (255, 0, 0), 2)
        
        return annotated_frame
    
    def annotate_frame(self, frame_num: int) -> bool:
        """
        Annotate a specific frame.
        
        Args:
            frame_num: Frame number to annotate
            
        Returns:
            True if annotation was saved, False if cancelled
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = self.cap.read()
        
        if not ret:
            print(f"Could not read frame {frame_num}")
            return False
        
        self.current_frame_num = frame_num
        self.current_boxes = []
        
        # Load existing annotations for this frame if any
        if str(frame_num) in self.annotations:
            self.current_boxes = self.annotations[str(frame_num)].copy()
        
        window_name = f"Annotate Frame {frame_num} - Click and drag to draw boxes"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        
        print(f"\nAnnotating frame {frame_num}")
        print("Instructions:")
        print("  - Click and drag to draw bounding boxes")
        print("  - Press 's' to save annotations for this frame")
        print("  - Press 'd' to delete last box")
        print("  - Press 'c' to clear all boxes")
        print("  - Press 'q' to quit without saving")
        print("  - Press 'n' to go to next frame")
        print("  - Press 'p' to go to previous frame")
        
        while True:
            annotated_frame = self.draw_boxes_on_frame(frame)
            
            # Add frame info
            info_text = f"Frame {frame_num}/{self.total_frames-1} | Boxes: {len(self.current_boxes)}"
            cv2.putText(annotated_frame, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow(window_name, annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('s'):
                # Save annotations
                if self.current_boxes:
                    self.annotations[str(frame_num)] = self.current_boxes.copy()
                    print(f"Saved {len(self.current_boxes)} boxes for frame {frame_num}")
                else:
                    print("No boxes to save")
                cv2.destroyWindow(window_name)
                return True
                
            elif key == ord('d'):
                # Delete last box
                if self.current_boxes:
                    self.current_boxes.pop()
                    print(f"Deleted last box. Remaining: {len(self.current_boxes)}")
                    
            elif key == ord('c'):
                # Clear all boxes
                self.current_boxes = []
                print("Cleared all boxes")
                
            elif key == ord('q'):
                # Quit without saving
                cv2.destroyWindow(window_name)
                return False
                
            elif key == ord('n'):
                # Next frame
                if frame_num < self.total_frames - 1:
                    cv2.destroyWindow(window_name)
                    return self.annotate_frame(frame_num + 1)
                else:
                    print("Already at last frame")
                    
            elif key == ord('p'):
                # Previous frame
                if frame_num > 0:
                    cv2.destroyWindow(window_name)
                    return self.annotate_frame(frame_num - 1)
                else:
                    print("Already at first frame")
        
        cv2.destroyWindow(window_name)
        return False
    
    def save_annotations(self, output_path: str):
        """Save annotations to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.annotations, f, indent=2)
        print(f"\nAnnotations saved to: {output_path}")
        print(f"Total frames annotated: {len(self.annotations)}")
    
    def load_annotations(self, input_path: str):
        """Load annotations from JSON file."""
        with open(input_path, 'r') as f:
            self.annotations = json.load(f)
        print(f"Loaded annotations from: {input_path}")
        print(f"Total frames annotated: {len(self.annotations)}")
    
    def release(self):
        """Release video capture."""
        self.cap.release()
        cv2.destroyAllWindows()


def main():
    """Main function for annotation tool."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ground Truth Annotation Tool')
    parser.add_argument('--video', type=str, required=True, help='Input video file path')
    parser.add_argument('--output', type=str, default='ground_truth.json', 
                       help='Output JSON file for annotations')
    parser.add_argument('--load', type=str, default=None, 
                       help='Load existing annotations from JSON file')
    parser.add_argument('--frame', type=int, default=0, 
                       help='Starting frame number (default: 0)')
    
    args = parser.parse_args()
    
    annotator = BoundingBoxAnnotator(args.video)
    
    if args.load:
        annotator.load_annotations(args.load)
    
    print(f"\nVideo: {args.video}")
    print(f"Total frames: {annotator.total_frames}")
    print(f"FPS: {annotator.fps}")
    
    # Annotate starting from specified frame
    annotator.annotate_frame(args.frame)
    
    # Save annotations
    annotator.save_annotations(args.output)
    annotator.release()


if __name__ == '__main__':
    main()

