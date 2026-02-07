"""Video recognition package."""

from .annotation import BoundingBoxAnnotator
from .detection import (
    VideoObjectDetector,
    build_output_artifact_path,
    calculate_iou,
    create_evaluation_report,
    evaluate_detections,
    match_detections_to_ground_truth,
    normalize_frame_map,
    track_color,
)

__all__ = [
    "BoundingBoxAnnotator",
    "VideoObjectDetector",
    "build_output_artifact_path",
    "calculate_iou",
    "create_evaluation_report",
    "evaluate_detections",
    "match_detections_to_ground_truth",
    "normalize_frame_map",
    "track_color",
]

