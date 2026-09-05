"""
Unit Tests for Model Training and Metrics Calculation
"""

import unittest
import numpy as np
from sklearn.linear_model import LinearRegression
from src.evaluate import calculate_metrics


class TestModelEvaluation(unittest.TestCase):

    def test_calculate_metrics_perfect_fit(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([100.0, 200.0, 300.0])
        metrics = calculate_metrics(y_true, y_pred)
        self.assertEqual(metrics["MAE"], 0.0)
        self.assertEqual(metrics["RMSE"], 0.0)
        self.assertEqual(metrics["R2_Score"], 1.0)
        self.assertEqual(metrics["MAPE_Percent"], 0.0)

    def test_calculate_metrics_known_difference(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 190.0])
        metrics = calculate_metrics(y_true, y_pred)
        self.assertEqual(metrics["MAE"], 10.0)
        self.assertEqual(metrics["RMSE"], 10.0)
        self.assertAlmostEqual(metrics["MAPE_Percent"], 7.5, delta=0.1)

    def test_model_training_fit(self):
        X = np.array([[1], [2], [3], [4]])
        y = np.array([2, 4, 6, 8])
        model = LinearRegression()
        model.fit(X, y)
        preds = model.predict([[5]])
        self.assertAlmostEqual(float(preds[0]), 10.0, delta=0.01)


if __name__ == "__main__":
    unittest.main()
