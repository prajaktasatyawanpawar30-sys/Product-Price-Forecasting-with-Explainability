# 👟 Product Price Forecasting with Explainability

**Project Type**: TY (Third Year) / Final Year College Project  
**Author**: Prajakta Pawar  
**Domain**: Machine Learning, Retail Analytics & Explainable AI (XAI)  

---

## 📌 1. Project Overview & Problem Statement

In retail and e-commerce, setting the right product price is crucial for business profitability and customer satisfaction. Traditional pricing strategies rely on manual intuition or simple historical averages, failing to capture complex interactions among brand tier, materials, customer ratings, discounts, competitor prices, and market demand.

Furthermore, traditional Machine Learning systems operate as "black boxes" — providing price predictions without explaining *why* a particular price was forecasted. This lack of transparency causes reluctance among retailers and consumers.

**This project solves both problems by:**
1. Forecasting product selling prices using advanced ML regression algorithms (Linear Regression, Ridge, Random Forest, Gradient Boosting, XGBoost).
2. Integrating **Explainable AI (XAI)** using **SHAP (SHapley Additive exPlanations)** to provide transparent, interpretable reasons behind every price forecast.
3. Providing an interactive **Streamlit Web Application** for real-time price estimation, data exploration, and visual explanation.

---

## 📂 2. Directory Structure

```text
Product_Price_Forecasting/
│
├── data/
│   ├── raw/
│   │   └── product_prices.csv           <- Raw footwear/product dataset
│   └── processed/
│       └── cleaned_data.csv             <- Cleaned, validated dataset
│
├── notebooks/                           <- Step-by-step Jupyter Notebooks
│   ├── 01_data_collection.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_feature_engineering.ipynb
│   ├── 05_model_training.ipynb
│   ├── 06_model_evaluation.ipynb
│   └── 07_shap_explainability.ipynb
│
├── src/                                 <- Modular Python Source Code
│   ├── __init__.py
│   ├── preprocessing.py                 <- Data cleaning, encoding, and scaling
│   ├── feature_engineering.py           <- Domain ratios, discounts, and brand tiers
│   ├── train_model.py                   <- Multi-model training and benchmark evaluation
│   ├── predict.py                       <- Live inference and SHAP prediction engine
│   ├── evaluate.py                      <- MAE, RMSE, R², MAPE metrics and plots
│   └── explainability.py                <- SHAP explainer and plain-English translation
│
├── models/                              <- Serialized ML Artifacts
│   ├── best_model.pkl                   <- Best performing regression model
│   ├── preprocessing.pkl                <- Fitted encoders and scalers
│   └── feature_names.pkl                <- Feature alignment list
│
├── outputs/
│   ├── figures/                         <- Visual plots
│   │   ├── actual_vs_predicted.png
│   │   ├── residuals_distribution.png
│   │   ├── price_trend.png
│   │   └── shap_summary.png
│   └── predictions/
│       └── predictions.csv              <- Test set predictions
│
├── app/
│   └── app.py                           <- Streamlit Web Application
│
├── tests/                               <- Automated Unit Tests
│   ├── test_preprocessing.py
│   ├── test_prediction.py
│   └── test_model.py
│
├── run_pipeline.py                      <- One-click pipeline runner
├── generate_data.py                     <- Dataset generator script
├── requirements.txt                     <- Project dependencies
├── README.md                            <- Project documentation
├── .gitignore
└── LICENSE
```

---

## 🧠 3. Methodology & Machine Learning Architecture

```
Raw Data Ingestion (CSV / XLSX)
           │
           ▼
Data Cleaning & Imputation (src/preprocessing.py)
           │
           ▼
Feature Engineering (src/feature_engineering.py)
   ├── Brand Tiering & Material Multipliers
   ├── Discount Amount & Effective Discount Ratio
   ├── Competitor Price Gap & Ratio
   └── Temporal / Seasonality Indicators
           │
           ▼
Train / Test Split (80% Train, 20% Test)
           │
           ▼
Model Training & Benchmark Comparison (src/train_model.py)
   ├── Linear Regression
   ├── Ridge Regression
   ├── Random Forest Regressor
   ├── Gradient Boosting Regressor
   └── XGBoost Regressor
           │
           ▼
Evaluation Metrics (MAE, MSE, RMSE, R², MAPE) (src/evaluate.py)
           │
           ▼
Model Persistence (models/best_model.pkl)
           │
           ▼
Explainable AI (XAI) with SHAP (src/explainability.py)
   ├── Global Feature Importance (Beeswarm & Summary Plots)
   ├── Local Waterfall Breakdown per Product
   └── Rule-Based Natural Language Explanation
           │
           ▼
Interactive Web Interface (Streamlit app/app.py)
```

---

## 📊 4. Model Evaluation Metrics

Models are evaluated using standard industry metrics:
- **MAE (Mean Absolute Error)**: Average absolute magnitude of errors in rupees (₹).
- **RMSE (Root Mean Squared Error)**: Penalizes larger pricing errors.
- **R² Score (Coefficient of Determination)**: Proportion of price variance explained by the model (closer to 1.0 is better).
- **MAPE (Mean Absolute Percentage Error)**: Average relative percentage error.

---

## 🚀 5. Getting Started & Installation

### Step 1: Clone or Navigate to Project Directory
```bash
cd "Prajakta_Pawar 32 Product Price Forecasting with Explainability"
```

### Step 2: Create a Virtual Environment (Recommended)
```bash
# Windows
py -m venv venv
.\venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Complete Machine Learning Pipeline
Train the models, evaluate performance, and generate all output charts in one command:
```bash
python run_pipeline.py
```

### Step 5: Launch the Streamlit Web Application
```bash
streamlit run app/app.py
```
*Open your browser at `http://localhost:8501` to view and interact with the application.*

### Step 6: Run Automated Unit Tests
```bash
pytest tests/
# or using python's built-in unittest
python -m unittest discover tests
```

---

## 💡 6. College Viva / Exam FAQ Guide

**Q1. What is the role of Explainable AI (XAI) in this project?**  
> Traditional ML models are black boxes. In pricing, retailers and customers need to know *why* a price was forecasted. Using SHAP (Shapley Additive exPlanations), our system attributes the exact rupee contribution of each feature (e.g. brand premium, discount rate, rating) to the final price.

**Q2. How does SHAP calculate feature contributions?**  
> SHAP is based on cooperative game theory. It considers all possible subsets of features (coalitions) and calculates the marginal contribution of each feature to the prediction compared to the base average prediction.

**Q3. Why compare multiple regression models?**  
> Different models capture different patterns. Linear and Ridge models capture basic linear trends, while tree ensembles (Random Forest, Gradient Boosting, XGBoost) capture complex non-linear interactions between brand, category, and discounts. Comparing them ensures we select the model with the highest $R^2$ and lowest $RMSE$.

**Q4. How do you prevent data leakage?**  
> All feature scaling and categorical encoders are fitted strictly on the training set (`fit_transform`) and subsequently applied to test and inference sets (`transform`).

---

## 📜 7. License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
