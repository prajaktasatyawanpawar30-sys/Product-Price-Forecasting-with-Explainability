"""
Explainability Module (SHAP - Explainable AI)
Product Price Forecasting with Explainability

Generates global and local model explanations using SHAP (SHapley Additive exPlanations)
and produces natural language summaries explaining the key pricing drivers.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import shap
except ImportError:
    shap = None


class PriceExplainer:
    """SHAP-based model explainer for price forecasting."""

    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self._init_explainer()

    def _init_explainer(self):
        """Initializes the SHAP TreeExplainer or fallback explainer."""
        if shap is None:
            print("[!] Warning: shap library is not installed. Explanations will use feature importances.")
            return

        try:
            # TreeExplainer is ideal for XGBoost, Random Forest, GradientBoosting
            self.explainer = shap.TreeExplainer(self.model)
        except Exception:
            try:
                self.explainer = shap.Explainer(self.model)
            except Exception as e:
                print(f"[!] Warning initializing SHAP explainer: {e}")
                self.explainer = None

    def explain_dataset(self, X_sample):
        """Computes SHAP values for a sample dataset."""
        if self.explainer is None or shap is None:
            return None
        shap_values = self.explainer(X_sample)
        return shap_values

    def plot_summary(self, X_sample, save_path=None, max_display=12):
        """Generates global feature importance summary plot."""
        if self.explainer is None or shap is None:
            # Fallback to feature importance bar chart if model has it
            if hasattr(self.model, "feature_importances_"):
                importances = self.model.feature_importances_
                names = self.feature_names or [f"Feature {i}" for i in range(len(importances))]
                sorted_idx = np.argsort(importances)[-max_display:]

                plt.figure(figsize=(9, 6))
                plt.barh(range(len(sorted_idx)), importances[sorted_idx], color="#2b5c8f")
                plt.yticks(range(len(sorted_idx)), [names[i] for i in sorted_idx])
                plt.title("Global Feature Importance (Gini / Gain)", fontsize=14, fontweight="bold")
                plt.xlabel("Importance", fontsize=12)
                plt.tight_layout()
                if save_path:
                    os.makedirs(os.path.dirname(save_path), exist_ok=True)
                    plt.savefig(save_path, dpi=300)
                    plt.close()
                else:
                    plt.show()
            return

        shap_values = self.explain_dataset(X_sample)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, max_display=max_display, show=False)
        plt.title("SHAP Global Feature Importance (Price Impact)", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[OK] Saved SHAP summary plot to {save_path}")
        else:
            plt.show()

    def plot_waterfall(self, single_row_df, save_path=None, max_display=10):
        """Generates local Waterfall plot for an individual product price prediction."""
        if self.explainer is None or shap is None:
            return None

        shap_values = self.explainer(single_row_df)
        explanation = shap_values[0]

        plt.figure(figsize=(9, 6))
        shap.plots.waterfall(explanation, max_display=max_display, show=False)
        plt.title("SHAP Price Breakdown (Local Explanation)", fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.close()
            print(f"[OK] Saved SHAP waterfall plot to {save_path}")
        else:
            plt.show()

        return explanation

    def generate_natural_language_explanation(self, single_row_df, predicted_price, original_input=None):
        """Translates SHAP contributions into human-readable plain English reasoning."""
        if self.explainer is None or shap is None:
            # Fallback basic explanation
            return (
                f"Predicted Price is ₹{predicted_price:,.2f}. "
                "The price is determined primarily by the product's original price tier and applied discount."
            )

        shap_values = self.explainer(single_row_df)
        exp = shap_values[0]

        base_price = float(exp.base_values)
        vals = exp.values
        names = self.feature_names or single_row_df.columns.tolist()

        # Pair features with their SHAP impact
        impacts = []
        for name, impact, orig_val in zip(names, vals, single_row_df.iloc[0]):
            display_val = original_input.get(name, orig_val) if original_input else orig_val
            impacts.append((name, float(impact), display_val))

        # Sort by absolute contribution
        positive_impacts = sorted([item for item in impacts if item[1] > 0], key=lambda x: x[1], reverse=True)
        negative_impacts = sorted([item for item in impacts if item[1] < 0], key=lambda x: x[1])

        lines = []
        diff = predicted_price - base_price
        direction = "higher" if diff >= 0 else "lower"
        lines.append(f"### 🔍 AI Explainability Summary:")
        lines.append(f"- **Baseline Average Market Price**: ₹{base_price:,.2f}")
        lines.append(f"- **Model Forecasted Price**: ₹{predicted_price:,.2f} ({direction} by ₹{abs(diff):,.2f})\n")

        if positive_impacts:
            lines.append("**Factors increasing the price:**")
            for name, impact, val in positive_impacts[:3]:
                readable_name = name.replace("_", " ")
                lines.append(f"  • **{readable_name}** ({val}): +₹{impact:,.2f}")

        if negative_impacts:
            lines.append("\n**Factors reducing the price:**")
            for name, impact, val in negative_impacts[:3]:
                readable_name = name.replace("_", " ")
                lines.append(f"  • **{readable_name}** ({val}): -₹{abs(impact):,.2f}")

        conclusion = (
            f"\n*Summary:* The model adjusted the market benchmark of ₹{base_price:,.2f} "
            f"to ₹{predicted_price:,.2f} based on feature attributes and current market demand."
        )
        lines.append(conclusion)

        return "\n".join(lines)


def get_feature_importance_df(model, feature_names):
    """Returns a sorted DataFrame of feature importances."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)
        return df
    return pd.DataFrame()
