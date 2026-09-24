import json
import math
import tempfile
import unittest
from pathlib import Path

from race_score_model import RaceScoreModel, course_features, time_to_hours


class RaceScoreModelTests(unittest.TestCase):
    def test_time_to_hours_supports_long_ultras(self):
        self.assertAlmostEqual(time_to_hours("44:24:06"), 44 + 24 / 60 + 6 / 3600)

    def test_course_features_use_km_effort(self):
        features = course_features(61, 3400)
        self.assertAlmostEqual(features[1], math.log(95))
        self.assertAlmostEqual(features[2], 34 / 95)

    def test_anchor_prediction_is_inverse_in_time(self):
        artifact = {
            "within_race_model": {"anchor_error_80": 2, "anchor_error_95": 4},
            "course_model": {
                "coefficients": [0, 1, 0],
                "loo_error_80": 20,
                "loo_error_95": 40,
            },
        }
        model = RaceScoreModel(artifact)
        result = model.estimate(
            distance_km=61,
            elevation_gain_m=3400,
            finish_time="06:00:00",
            anchor_time="05:00:00",
            anchor_score=900,
        )
        self.assertAlmostEqual(result.score, 750)
        self.assertEqual(result.confidence, "high")

    def test_multiple_anchors_are_robust_and_narrow_uncertainty(self):
        artifact = {
            "within_race_model": {"anchor_error_80": 2, "anchor_error_95": 4},
            "course_model": {
                "coefficients": [0, 1, 0],
                "loo_error_80": 20,
                "loo_error_95": 40,
            },
        }
        model = RaceScoreModel(artifact)
        anchors = [
            ("10:00:00", 800),
            ("08:00:00", 1000),
            ("20:00:00", 400),
            ("10:00:00", 1000),  # deliberately inconsistent outlier
        ]
        calibration = model.calibrate_anchors(anchors)
        result = model.estimate_with_anchors(
            finish_time="10:00:00", anchors=anchors
        )
        single = model.estimate(
            distance_km=1, elevation_gain_m=0, finish_time="10:00:00",
            anchor_time="10:00:00", anchor_score=800,
        )
        self.assertEqual(calibration.rejected_count, 1)
        self.assertAlmostEqual(result.score, 800)
        self.assertEqual(result.method, "multi_race_anchor_robust_median")
        self.assertLess(
            result.upper_80 - result.lower_80,
            single.upper_80 - single.lower_80,
        )

    def test_model_artifact_can_be_loaded(self):
        artifact = {
            "within_race_model": {"anchor_error_80": 2, "anchor_error_95": 4},
            "course_model": {
                "coefficients": [0, 1, 0],
                "loo_error_80": 20,
                "loo_error_95": 40,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(artifact), encoding="utf-8")
            self.assertIsInstance(RaceScoreModel.load(path), RaceScoreModel)


if __name__ == "__main__":
    unittest.main()
