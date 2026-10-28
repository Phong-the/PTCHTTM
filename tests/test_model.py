import unittest
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = joblib.load(ROOT / "models" / "performance_classifier.joblib")
FEATURES = [
    "attendance_pct", "weekly_study_hours", "assignment_completion_pct",
    "quiz_average", "midterm_score", "engagement_level"
]

class ModelSmokeTests(unittest.TestCase):
    def test_prediction_returns_known_class(self):
        sample = pd.DataFrame([{
            "attendance_pct": 92, "weekly_study_hours": 12,
            "assignment_completion_pct": 95, "quiz_average": 88,
            "midterm_score": 90, "engagement_level": 4.5
        }], columns=FEATURES)
        prediction = MODEL.predict(sample)[0]
        self.assertIn(prediction, ["Cần hỗ trợ", "Đạt yêu cầu", "Tốt"])

    def test_batch_prediction_shape(self):
        samples = pd.DataFrame([
            [95, 15, 95, 90, 92, 5],
            [40, 2, 20, 25, 30, 1],
            [75, 7, 70, 65, 66, 3],
        ], columns=FEATURES)
        predictions = MODEL.predict(samples)
        self.assertEqual(len(predictions), 3)

    def test_missing_values_are_supported(self):
        sample = pd.DataFrame([{
            "attendance_pct": 80, "weekly_study_hours": None,
            "assignment_completion_pct": 75, "quiz_average": 60,
            "midterm_score": 65, "engagement_level": 3
        }], columns=FEATURES)
        prediction = MODEL.predict(sample)[0]
        self.assertIn(prediction, ["Cần hỗ trợ", "Đạt yêu cầu", "Tốt"])

if __name__ == "__main__":
    unittest.main(verbosity=2)
