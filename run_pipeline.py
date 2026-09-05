"""
Master Pipeline Runner: Product Price Forecasting with Explainability
Executes the full machine learning lifecycle from data ingestion to model serialization,
evaluation visual generation, and explainability testing.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.train_model import ModelTrainer
from src.predict import PricePredictor


def main():
    print("\n" + "=" * 70)
    print("👟 PRODUCT PRICE FORECASTING WITH EXPLAINABILITY PIPELINE")
    print("   Author: Prajakta Pawar | College Project")
    print("=" * 70)

    # 1. Verify Dataset
    raw_data = os.path.join(BASE_DIR, "data", "raw", "product_prices.csv")
    if not os.path.exists(raw_data):
        print(f"[*] Raw dataset not found at {raw_data}. Generating rich seed dataset...")
        from generate_data import generate_full_dataset
        generate_full_dataset(raw_data, num_records=500)

    # 2. Run Training & Evaluation Pipeline
    trainer = ModelTrainer(data_path=raw_data)
    results = trainer.train_and_evaluate(test_size=0.2)

    # 3. Test Live Inference with SHAP
    print("\n" + "-" * 70)
    print("[*] Testing Live Price Predictor & SHAP Explainability Engine...")
    print("-" * 70)

    sample_product = {
        "Brand": "Nike",
        "Category": "Running",
        "Gender": "Men",
        "Material": "Mesh",
        "Size": 9,
        "Rating": 4.6,
        "Reviews_Count": 380,
        "Stock_Quantity": 85,
        "Sales_Quantity": 52,
        "Competitor_Price": 5200.0,
        "Discount_Percentage": 15.0,
        "Original_Price": 5500.0,
        "Date": "2024-06-15"
    }

    try:
        predictor = PricePredictor()
        waterfall_output = os.path.join(BASE_DIR, "outputs", "figures", "sample_waterfall.png")
        pred_res = predictor.predict_single(sample_product, generate_explanation=True, waterfall_path=waterfall_output)

        print(f"\n[DEMO INFERENCE]")
        print(f"  Input Product   : {sample_product['Brand']} {sample_product['Category']} (MRP: ₹{sample_product['Original_Price']})")
        print(f"  Forecasted Price: ₹{pred_res['predicted_price']:,.2f}")
        print(f"\n{pred_res['explanation']}")
    except Exception as e:
        print(f"[!] Live inference notice: {e}")

    print("\n" + "=" * 70)
    print("🎉 Pipeline completed successfully!")
    print("   • Models saved in       : models/")
    print("   • Figures saved in      : outputs/figures/")
    print("   • Predictions saved in  : outputs/predictions/")
    print("   • Launch Streamlit app  : streamlit run app/app.py")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
