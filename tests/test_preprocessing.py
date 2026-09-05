"""
Unit Tests for Data Preprocessing
"""

import unittest
import pandas as pd
import numpy as np
from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.preprocessor = DataPreprocessor(target_col="Price")
        self.fe = FeatureEngineer()
        self.sample_df = pd.DataFrame({
            "Product_ID": ["P1", "P2", "P3", "P1"],
            "Product_Name": ["Shoe A", "Shoe B", "Shoe C", "Shoe A"],
            "Brand": ["Nike", "Adidas", "Puma", "Nike"],
            "Category": ["Running", "Sneakers", "Casual", "Running"],
            "Rating": [4.5, np.nan, 4.0, 4.5],
            "Reviews_Count": [100, 200, np.nan, 100],
            "Original_Price": [5000.0, 4000.0, 3000.0, 5000.0],
            "Discount_Percentage": [10, 20, 15, 10],
            "Price": [4500.0, 3200.0, 2550.0, 4500.0]
        })

    def test_clean_data_removes_duplicates(self):
        cleaned = self.preprocessor.clean_data(self.sample_df)
        self.assertEqual(len(cleaned), 3, "Duplicate row should have been dropped")

    def test_clean_data_imputes_missing(self):
        cleaned = self.preprocessor.clean_data(self.sample_df)
        self.assertFalse(cleaned["Rating"].isna().any(), "Missing rating should be imputed")
        self.assertFalse(cleaned["Reviews_Count"].isna().any(), "Missing reviews count should be imputed")

    def test_feature_engineering_creates_discount_amount(self):
        featured = self.fe.transform(self.sample_df)
        self.assertIn("Discount_Amount", featured.columns)
        self.assertEqual(featured.loc[0, "Discount_Amount"], 500.0)

    def test_fit_transform_pipeline(self):
        featured = self.fe.transform(self.sample_df)
        cleaned = self.preprocessor.clean_data(featured)
        X, y = self.preprocessor.fit_transform(cleaned)
        self.assertIsNotNone(y)
        self.assertEqual(len(X), len(y))
        self.assertTrue(self.preprocessor.fitted)


if __name__ == "__main__":
    unittest.main()
