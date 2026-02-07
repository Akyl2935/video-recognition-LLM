import json
import tempfile
import unittest
from pathlib import Path

from video_object_detection import (
    build_output_artifact_path,
    calculate_iou,
    evaluate_detections,
    match_detections_to_ground_truth,
)


class EvaluationTests(unittest.TestCase):
    def test_calculate_iou_perfect_overlap(self):
        self.assertEqual(calculate_iou([0, 0, 10, 10], [0, 0, 10, 10]), 1.0)

    def test_calculate_iou_handles_invalid_box(self):
        self.assertEqual(calculate_iou([10, 10, 0, 0], [0, 0, 10, 10]), 0.0)

    def test_match_detections_to_ground_truth(self):
        detections = [
            {"bbox": [0, 0, 10, 10], "confidence": 0.9, "class_id": 0},
            {"bbox": [50, 50, 60, 60], "confidence": 0.8, "class_id": 0},
        ]
        ground_truth = [[0, 0, 10, 10]]
        matched_ious, flags = match_detections_to_ground_truth(detections, ground_truth, 0.5)

        self.assertEqual(len(matched_ious), 1)
        self.assertEqual(flags, [True, False])

    def test_evaluate_counts_false_negative_for_missing_detection_frame(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            detections_path = tmp_path / "detections.json"
            ground_truth_path = tmp_path / "ground_truth.json"

            # Detection exists only for frame 0.
            detections = {
                "0": [{"bbox": [0, 0, 10, 10], "confidence": 0.9, "class_id": 0}]
            }
            # Ground truth exists for frame 0 and frame 1.
            ground_truth = {
                "0": [[0, 0, 10, 10]],
                "1": [[20, 20, 30, 30]],
            }

            detections_path.write_text(json.dumps(detections), encoding="utf-8")
            ground_truth_path.write_text(json.dumps(ground_truth), encoding="utf-8")

            result = evaluate_detections(str(detections_path), str(ground_truth_path), 0.5)

            self.assertEqual(result["evaluated_frames"], 2)
            self.assertEqual(result["total_true_positives"], 1)
            self.assertEqual(result["total_false_positives"], 0)
            self.assertEqual(result["total_false_negatives"], 1)

    def test_build_output_artifact_path_for_non_mp4(self):
        self.assertEqual(
            build_output_artifact_path("results/video.MOV", "_detections.json"),
            str(Path("results") / "video_detections.json"),
        )


if __name__ == "__main__":
    unittest.main()

