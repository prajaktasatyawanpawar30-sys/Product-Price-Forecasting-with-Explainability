"""
Prediction & Inference Module
Product Price Forecasting with Explainability

Loads trained model artifacts, preprocesses single or batch input records,
computes forecasted product price, and provides SHAP explanations.
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np

# Ensure root directory is on sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.feature_engineering import FeatureEngineer
from src.explainability import PriceExplainer


class PricePredictor:
    """Inference engine for product price forecasting with explainability."""

    def __init__(self, models_dir=None):
        self.models_dir = models_dir or os.path.join(PROJECT_ROOT, "models")
        self.model_path = os.path.join(self.models_dir, "best_model.pkl")
        self.preprocessor_path = os.path.join(self.models_dir, "preprocessing.pkl")
        self.features_path = os.path.join(self.models_dir, "feature_names.pkl")

        self.model = None
        self.preprocessor = None
        self.feature_names = None
        self.feature_engineer = FeatureEngineer()
        self.explainer = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads serialized model, preprocessor, and feature list."""
        if not (os.path.exists(self.model_path) and os.path.exists(self.preprocessor_path)):
            raise FileNotFoundError(
                f"Model artifacts not found in {self.models_dir}. "
                "Please run `python src/train_model.py` or `python run_pipeline.py` first."
            )

        self.model = joblib.load(self.model_path)
        self.preprocessor = joblib.load(self.preprocessor_path)
        if os.path.exists(self.features_path):
            self.feature_names = joblib.load(self.features_path)

        self.explainer = PriceExplainer(self.model, feature_names=self.feature_names)

    def predict_single(self, input_data: dict, generate_explanation=True, waterfall_path=None):
        """Predicts price for a single product dictionary and provides SHAP explanation."""
        # Convert dictionary to DataFrame
        df_input = pd.DataFrame([input_data])

        # Feature engineering
        df_featured = self.feature_engineer.transform(df_input)

        # Preprocessing & encoding
        X_proc = self.preprocessor.transform(df_featured)

        # Predict
        predicted_price = float(self.model.predict(X_proc)[0])
        predicted_price = max(0.0, round(predicted_price, 2))

        explanation_text = ""
        waterfall_exp = None

        if generate_explanation and self.explainer:
            explanation_text = self.explainer.generate_natural_language_explanation(
                X_proc, predicted_price, original_input=input_data
            )
            if waterfall_path:
                waterfall_exp = self.explainer.plot_waterfall(X_proc, save_path=waterfall_path)

        return {
            "predicted_price": predicted_price,
            "explanation": explanation_text,
            "features_used": X_proc.to_dict(orient="records")[0]
        }

    def predict_batch(self, df_inputs: pd.DataFrame):
        """Predicts prices for a batch DataFrame of products."""
        df_featured = self.feature_engineer.transform(df_inputs)
        X_proc = self.preprocessor.transform(df_featured)
        predictions = self.model.predict(X_proc)

        df_result = df_inputs.copy()
        df_result["Forecasted_Price"] = np.round(predictions, 2)
        return df_result


def quick_demo():
    """Demonstrates single product inference and explanation."""
    sample_product = {
        "Brand": "Nike",
        "Category": "Running",
        "Gender": "Men",
        "Material": "Mesh",
        "Size": 9,
        "Rating": 4.6,
        "Reviews_Count": 350,
        "Stock_Quantity": 90,
        "Sales_Quantity": 45,
        "Competitor_Price": 5200.0,
        "Discount_Percentage": 15.0,
        "Original_Price": 5500.0,
        "Date": "2024-06-15"
    }

    try:
        predictor = PricePredictor()
        res = predictor.predict_single(sample_product)
        print("=== Prediction Result ===")
        print(f"Predicted Selling Price: ₹{res['predicted_price']:,.2f}")
        print("\n" + res["explanation"])
    except Exception as e:
        print(f"Demo prediction info: {e}")


if __name__ == "__main__":
    quick_demo()
