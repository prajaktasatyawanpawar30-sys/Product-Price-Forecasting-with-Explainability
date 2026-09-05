"""
Jupyter Notebook Generator
Product Price Forecasting with Explainability

Generates 7 formatted .ipynb notebooks matching the project syllabus and structure:
01_data_collection.ipynb
02_data_cleaning.ipynb
03_eda.ipynb
04_feature_engineering.ipynb
05_model_training.ipynb
06_model_evaluation.ipynb
07_shap_explainability.ipynb
"""

import os
import json

NOTEBOOKS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "notebooks")
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)


def create_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.12.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }


def md_cell(text):
    lines = [line + "\n" for line in text.split("\n")]
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": lines
    }


def code_cell(code):
    lines = [line + "\n" for line in code.split("\n")]
    if lines and lines[-1].endswith("\n"):
        lines[-1] = lines[-1][:-1]
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": lines
    }


# -------------------------------------------------------------
# 01_data_collection.ipynb
# -------------------------------------------------------------
nb01 = create_nb([
    md_cell("# 01 - Data Collection\n## Product Price Forecasting with Explainability\n**Project**: TY College Project\n**Author**: Prajakta Pawar\n\n### Objective:\nLoad raw product price dataset, inspect metadata, data types, and initial records."),
    code_cell("import os\nimport pandas as pd\nimport numpy as np\n\n# Define data paths\nraw_data_path = '../data/raw/product_prices.csv'\nprint(f'Checking dataset at: {raw_data_path}')\nassert os.path.exists(raw_data_path), 'Raw dataset not found!'"),
    code_cell("df = pd.read_csv(raw_data_path)\nprint(f'Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns')\ndf.head()"),
    code_cell("df.info()"),
    code_cell("print('Summary Statistics:')\ndf.describe()")
])

# -------------------------------------------------------------
# 02_data_cleaning.ipynb
# -------------------------------------------------------------
nb02 = create_nb([
    md_cell("# 02 - Data Cleaning & Preprocessing\n## Product Price Forecasting with Explainability\n\n### Objective:\nIdentify and handle missing values, remove duplicates, filter invalid records, and cast proper data types."),
    code_cell("import os\nimport pandas as pd\nimport numpy as np\n\ndf = pd.read_csv('../data/raw/product_prices.csv')\nprint('Initial shape:', df.shape)"),
    code_cell("# Check for missing values\nprint('Missing values per column:')\nprint(df.isnull().sum())"),
    code_cell("# Check and drop duplicates\ndupes = df.duplicated().sum()\nprint(f'Number of duplicate rows: {dupes}')\ndf = df.drop_duplicates().reset_index(drop=True)"),
    code_cell("# Convert Date and ensure positive target price\nif 'Date' in df.columns:\n    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')\n\n# Ensure price is valid\ndf = df[df['Price'] > 0]\nprint('Cleaned shape:', df.shape)"),
    code_cell("# Persist processed data\nos.makedirs('../data/processed', exist_ok=True)\ndf.to_csv('../data/processed/cleaned_data.csv', index=False)\nprint('Saved cleaned data to ../data/processed/cleaned_data.csv')")
])

# -------------------------------------------------------------
# 03_eda.ipynb
# -------------------------------------------------------------
nb03 = create_nb([
    md_cell("# 03 - Exploratory Data Analysis (EDA)\n## Product Price Forecasting with Explainability\n\n### Objective:\nExplore price distributions, brand variations, category patterns, and feature correlations."),
    code_cell("import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\nsns.set_theme(style='whitegrid')\ndf = pd.read_csv('../data/processed/cleaned_data.csv')\ndf.head()"),
    code_cell("# 1. Price Distribution\nplt.figure(figsize=(9, 4))\nsns.histplot(df['Price'], kde=True, color='#2563eb', bins=25)\nplt.title('Target Price Distribution (₹)', fontsize=13)\nplt.xlabel('Price (₹)')\nplt.ylabel('Frequency')\nplt.show()"),
    code_cell("# 2. Average Price by Brand\nplt.figure(figsize=(10, 5))\ndf.groupby('Brand')['Price'].mean().sort_values(ascending=False).plot(kind='bar', color='#10b981')\nplt.title('Mean Price by Footwear Brand', fontsize=13)\nplt.ylabel('Average Price (₹)')\nplt.xticks(rotation=45)\nplt.show()"),
    code_cell("# 3. Price by Category\nplt.figure(figsize=(10, 5))\nsns.boxplot(x='Category', y='Price', data=df, palette='Set2')\nplt.title('Price Distribution Across Product Categories', fontsize=13)\nplt.xticks(rotation=45)\nplt.show()"),
    code_cell("# 4. Numerical Correlation Heatmap\nplt.figure(figsize=(8, 6))\nnum_df = df.select_dtypes(include=[np.number])\nsns.heatmap(num_df.corr(), annot=True, fmt='.2f', cmap='coolwarm', cbar=True)\nplt.title('Feature Correlation Heatmap', fontsize=13)\nplt.show()")
])

# -------------------------------------------------------------
# 04_feature_engineering.ipynb
# -------------------------------------------------------------
nb04 = create_nb([
    md_cell("# 04 - Feature Engineering\n## Product Price Forecasting with Explainability\n\n### Objective:\nCreate high-signal domain features (discount amount, price ratio, brand tier, temporal calendar features)."),
    code_cell("import sys\nsys.path.append('..')\nimport pandas as pd\nimport numpy as np\nfrom src.feature_engineering import FeatureEngineer\n\ndf = pd.read_csv('../data/processed/cleaned_data.csv')\nfe = FeatureEngineer()\ndf_features = fe.transform(df)\nprint('Engineered columns:', df_features.columns.tolist())"),
    code_cell("df_features[['Product_Name', 'Original_Price', 'Discount_Percentage', 'Discount_Amount', 'Brand_Tier', 'Popularity_Score']].head(8)"),
    code_cell("print('Feature matrix shape:', df_features.shape)")
])

# -------------------------------------------------------------
# 05_model_training.ipynb
# -------------------------------------------------------------
nb05 = create_nb([
    md_cell("# 05 - Model Training\n## Product Price Forecasting with Explainability\n\n### Objective:\nTrain multiple regression models (Linear Regression, Ridge, Random Forest, Gradient Boosting, XGBoost)."),
    code_cell("import sys\nsys.path.append('..')\nfrom src.train_model import ModelTrainer\n\ntrainer = ModelTrainer()\nresults = trainer.train_and_evaluate()"),
    code_cell("import json\nprint(json.dumps(results, indent=2))")
])

# -------------------------------------------------------------
# 06_model_evaluation.ipynb
# -------------------------------------------------------------
nb06 = create_nb([
    md_cell("# 06 - Model Evaluation & Diagnostics\n## Product Price Forecasting with Explainability\n\n### Objective:\nCompare regression performance across MAE, MSE, RMSE, R², and MAPE, and inspect error distributions."),
    code_cell("import os\nimport json\nimport pandas as pd\nimport matplotlib.pyplot as plt\n\nwith open('../outputs/model_comparison_metrics.json') as f:\n    metrics = json.load(f)\n\ndf_metrics = pd.DataFrame(metrics).T\ndf_metrics"),
    code_cell("# Plot R² Comparison\nplt.figure(figsize=(8, 4))\ndf_metrics['R2_Score'].plot(kind='bar', color='#3b82f6')\nplt.title('R² Score Comparison Across Models', fontsize=13)\nplt.ylabel('R²')\nplt.ylim(0.8, 1.0)\nplt.xticks(rotation=30)\nplt.grid(True, linestyle='--', alpha=0.5)\nplt.show()"),
    code_cell("# Load and inspect predictions\npreds_df = pd.read_csv('../outputs/predictions/predictions.csv')\npreds_df.head(10)")
])

# -------------------------------------------------------------
# 07_shap_explainability.ipynb
# -------------------------------------------------------------
nb07 = create_nb([
    md_cell("# 07 - SHAP Explainability (XAI)\n## Product Price Forecasting with Explainability\n\n### Objective:\nGenerate global feature importance (Summary plots) and local price predictions (Waterfall & Force plots) using SHAP."),
    code_cell("import sys\nsys.path.append('..')\nimport joblib\nimport pandas as pd\nimport matplotlib.pyplot as plt\nimport shap\nfrom src.explainability import PriceExplainer\nfrom src.predict import PricePredictor\n\npredictor = PricePredictor()\nprint('Loaded model:', predictor.model)"),
    code_cell("# Sample single product test\nsample_product = {\n    'Brand': 'Nike',\n    'Category': 'Running',\n    'Gender': 'Men',\n    'Material': 'Mesh',\n    'Size': 9,\n    'Rating': 4.5,\n    'Reviews_Count': 320,\n    'Stock_Quantity': 120,\n    'Sales_Quantity': 45,\n    'Competitor_Price': 5200.0,\n    'Discount_Percentage': 15.0,\n    'Original_Price': 5500.0,\n    'Date': '2024-06-01'\n}\n\nres = predictor.predict_single(sample_product)\nprint(f'Forecasted Selling Price: ₹{res[\"predicted_price\"]}')\nprint(res['explanation'])"),
    code_cell("# Display global SHAP summary plot\nfrom IPython.display import Image\nImage('../outputs/figures/shap_summary.png')")
])


def write_all():
    notebooks = {
        "01_data_collection.ipynb": nb01,
        "02_data_cleaning.ipynb": nb02,
        "03_eda.ipynb": nb03,
        "04_feature_engineering.ipynb": nb04,
        "05_model_training.ipynb": nb05,
        "06_model_evaluation.ipynb": nb06,
        "07_shap_explainability.ipynb": nb07,
    }
    for filename, nb in notebooks.items():
        path = os.path.join(NOTEBOOKS_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=2)
        print(f"[OK] Generated {path}")

if __name__ == "__main__":
    write_all()
