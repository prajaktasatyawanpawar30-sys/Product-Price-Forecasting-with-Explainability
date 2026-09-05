"""
Model Evaluation Module
Product Price Forecasting with Explainability

Calculates standard regression evaluation metrics (MAE, MSE, RMSE, R², MAPE)
and generates visual diagnostics (actual vs predicted, residuals, price trends).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_metrics(y_true, y_pred):
    """Calculates MAE, MSE, RMSE, R2, and MAPE."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # Avoid division by zero in MAPE
    mask = y_true != 0
    mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0

    return {
        "MAE": round(float(mae), 2),
        "MSE": round(float(mse), 2),
        "RMSE": round(float(rmse), 2),
        "R2_Score": round(float(r2), 4),
        "MAPE_Percent": round(float(mape), 2)
    }


def plot_actual_vs_predicted(y_true, y_pred, model_name="Best Model", save_path=None):
    """Generates and saves Actual vs Predicted comparison plot."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, color="#1f77b4", alpha=0.6, edgecolors="k", label="Predictions")

    # Perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Ideal (y = x)")

    plt.title(f"Actual vs Predicted Prices ({model_name})", fontsize=14, fontweight="bold")
    plt.xlabel("Actual Price (₹)", fontsize=12)
    plt.ylabel("Predicted Price (₹)", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[OK] Saved Actual vs Predicted plot to {save_path}")
    else:
        plt.show()


def plot_residuals(y_true, y_pred, model_name="Best Model", save_path=None):
    """Generates and saves residual distribution plot."""
    residuals = np.array(y_true) - np.array(y_pred)
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=25, color="#2ca02c", edgecolor="black", alpha=0.7)
    plt.axvline(0, color="red", linestyle="--", linewidth=2)
    plt.title(f"Residuals Distribution ({model_name})", fontsize=14, fontweight="bold")
    plt.xlabel("Residual (Actual - Predicted Price in ₹)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[OK] Saved Residuals plot to {save_path}")
    else:
        plt.show()


def plot_price_trend(df, date_col="Date", price_col="Price", save_path=None):
    """Plots historical price trends over time."""
    if date_col not in df.columns or price_col not in df.columns:
        return

    df_plot = df.copy()
    df_plot[date_col] = pd.to_datetime(df_plot[date_col], errors="coerce")
    df_plot = df_plot.dropna(subset=[date_col]).sort_values(by=date_col)

    # Monthly average price trend
    monthly_trend = df_plot.set_index(date_col).resample("ME")[price_col].mean().dropna()

    plt.figure(figsize=(10, 5))
    plt.plot(monthly_trend.index, monthly_trend.values, marker="o", color="#d62728", linewidth=2.5)
    plt.title("Historical Average Product Price Trend", fontsize=14, fontweight="bold")
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Average Price (₹)", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[OK] Saved Price Trend plot to {save_path}")
    else:
        plt.show()
