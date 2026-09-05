"""
Unit Tests for Prediction Module
"""

import os
import unittest
import pandas as pd
from src.feature_engineering import FeatureEngineer


class TestPrediction(unittest.TestCase):

    def setUp(self):
        self.fe = FeatureEngineer()
        self.sample_input = {
            "Brand": "Nike",
            "Category": "Running",
            "Gender": "Men",
            "Material": "Mesh",
            "Size": 9,
            "Rating": 4.5,
            "Reviews_Count": 300,
            "Stock_Quantity": 100,
            "Sales_Quantity": 50,
            "Competitor_Price": 5000.0,
            "Discount_Percentage": 20.0,
            "Original_Price": 6000.0,
            "Date": "2024-05-10"
        }

    def test_feature_engineering_on_single_input(self):
        df_input = pd.DataFrame([self.sample_input])
        df_feat = self.fe.transform(df_input)
        self.assertIn("Year", df_feat.columns)
        self.assertIn("Month", df_feat.columns)
        self.assertIn("Discount_Amount", df_feat.columns)
        self.assertIn("Brand_Tier", df_feat.columns)
        self.assertEqual(df_feat["Brand_Tier"].iloc[0], "Premium")


if __name__ == "__main__":
    unittest.main()
