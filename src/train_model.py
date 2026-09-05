"""
Model Training & Comparison Pipeline
Product Price Forecasting with Explainability

Trains and compares multiple regression models (Linear Regression, Ridge,
Random Forest, Gradient Boosting, XGBoost), selects the best-performing model,
and persists model artifacts and evaluation figures.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# Add root directory to sys.path to allow imports from src
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.evaluate import calculate_metrics, plot_actual_vs_predicted, plot_residuals, plot_price_trend
from src.explainability import PriceExplainer

# Optional XGBoost import with graceful fallback
try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


class ModelTrainer:
    """Orchestrates model training, evaluation, comparison, and serialization."""

    def __init__(self, data_path=None):
        self.data_path = data_path or os.path.join(PROJECT_ROOT, "data", "raw", "product_prices.csv")
        self.models_dir = os.path.join(PROJECT_ROOT, "models")
        self.outputs_dir = os.path.join(PROJECT_ROOT, "outputs")
        self.figures_dir = os.path.join(self.outputs_dir, "figures")
        self.predictions_dir = os.path.join(self.outputs_dir, "predictions")

        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.predictions_dir, exist_ok=True)

        self.preprocessor = DataPreprocessor(target_col="Price")
        self.feature_engineer = FeatureEngineer()
        self.best_model = None
        self.best_model_name = None
        self.results = {}

    def get_candidate_models(self):
        """Returns dictionary of candidate regression models to evaluate."""
        models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression": Ridge(alpha=1.0, random_state=42),
            "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=5, random_state=42)
        }
        if HAS_XGBOOST:
            models["XGBoost"] = XGBRegressor(n_estimators=100, learning_rate=0.08, max_depth=5, random_state=42, objective="reg:squarederror")
        return models

    def train_and_evaluate(self, test_size=0.2):
        """Runs the complete training and comparison pipeline."""
        print("=" * 65)
        print("Starting Model Training Pipeline: Product Price Forecasting")
        print("=" * 65)

        # 1. Ingestion
        print(f"[*] Ingesting dataset from: {self.data_path}")
        df_raw = self.preprocessor.load_data(self.data_path)
        print(f"    Raw records loaded: {len(df_raw)} rows, {df_raw.shape[1]} columns")

        # 2. Cleaning
        df_clean = self.preprocessor.clean_data(df_raw)
        processed_csv = os.path.join(PROJECT_ROOT, "data", "processed", "cleaned_data.csv")
        os.makedirs(os.path.dirname(processed_csv), exist_ok=True)
        df_clean.to_csv(processed_csv, index=False)
        print(f"[OK] Saved cleaned data to: {processed_csv}")

        # 3. Feature Engineering
        print("[*] Applying Feature Engineering (domain indicators, discounts, temporal)...")
        df_featured = self.feature_engineer.transform(df_clean)

        # 4. Encoding & Scaling
        X, y = self.preprocessor.fit_transform(df_featured)
        feature_names = list(X.columns)
        print(f"    Total engineered features: {len(feature_names)}")

        # 5. Train/Test Split
        X_train, X_test, y_train, y_test = self.preprocessor.split_data(X, y, test_size=test_size)
        print(f"    Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # 6. Train & Compare Models
        models = self.get_candidate_models()
        best_r2 = -float("inf")
        best_preds = None

        print("\n--- Model Benchmark Comparison ---")
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            metrics = calculate_metrics(y_test, y_pred)
            self.results[name] = metrics
            print(f"[{name:^20}] R²: {metrics['R2_Score']:.4f} | MAE: ₹{metrics['MAE']:>7.2f} | RMSE: ₹{metrics['RMSE']:>7.2f} | MAPE: {metrics['MAPE_Percent']}%")

            if metrics["R2_Score"] > best_r2:
                best_r2 = metrics["R2_Score"]
                self.best_model = model
                self.best_model_name = name
                best_preds = y_pred

        print("=" * 65)
        print(f"[SUCCESS] Top Model: {self.best_model_name} (R² = {best_r2:.4f})")
        print("=" * 65)

        # 7. Save Serialized Artifacts
        best_model_path = os.path.join(self.models_dir, "best_model.pkl")
        preprocessor_path = os.path.join(self.models_dir, "preprocessing.pkl")
        features_path = os.path.join(self.models_dir, "feature_names.pkl")

        joblib.dump(self.best_model, best_model_path)
        joblib.dump(self.preprocessor, preprocessor_path)
        joblib.dump(feature_names, features_path)

        # Save metrics json
        metrics_summary_path = os.path.join(self.outputs_dir, "model_comparison_metrics.json")
        with open(metrics_summary_path, "w") as f:
            json.dump(self.results, f, indent=4)

        # 8. Save Predictions
        pred_df = pd.DataFrame({
            "Actual_Price": y_test,
            "Predicted_Price": np.round(best_preds, 2),
            "Absolute_Error": np.round(np.abs(y_test - best_preds), 2)
        })
        pred_csv = os.path.join(self.predictions_dir, "predictions.csv")
        pred_df.to_csv(pred_csv, index=False)
        print(f"[OK] Saved predictions to: {pred_csv}")

        # 9. Generate and Save Figures
        print("[*] Generating diagnostic figures...")
        fig_actual_vs_pred = os.path.join(self.figures_dir, "actual_vs_predicted.png")
        plot_actual_vs_predicted(y_test, best_preds, model_name=self.best_model_name, save_path=fig_actual_vs_pred)

        fig_residuals = os.path.join(self.figures_dir, "residuals_distribution.png")
        plot_residuals(y_test, best_preds, model_name=self.best_model_name, save_path=fig_residuals)

        fig_trend = os.path.join(self.figures_dir, "price_trend.png")
        plot_price_trend(df_clean, save_path=fig_trend)

        # 10. Generate SHAP Explanations
        print("[*] Generating SHAP explainability artifacts...")
        explainer = PriceExplainer(self.best_model, feature_names=feature_names)
        shap_summary_path = os.path.join(self.figures_dir, "shap_summary.png")
        sample_subset = X_test.iloc[: min(80, len(X_test))]
        explainer.plot_summary(sample_subset, save_path=shap_summary_path)

        print("\nAll pipeline tasks completed successfully.")
        return self.results


if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_and_evaluate()
