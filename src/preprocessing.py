"""
Data Preprocessing Module
Product Price Forecasting with Explainability

Handles data ingestion, validation, cleaning, missing value imputation,
encoding, feature scaling, and train-test splitting.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


class DataPreprocessor:
    """End-to-end data preprocessor and serialization manager."""

    def __init__(self, target_col="Price"):
        self.target_col = target_col
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.numerical_cols = []
        self.categorical_cols = []
        self.feature_columns = []
        self.fitted = False

    def load_data(self, file_path):
        """Loads data from CSV or Excel file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at: {file_path}")

        if file_path.endswith(".xlsx") or file_path.endswith(".xls"):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)

        # Strip any accidental whitespace from column names
        df.columns = [c.strip() for c in df.columns]
        return df

    def clean_data(self, df):
        """Cleans missing, duplicate, and anomalous data records."""
        df_clean = df.copy()

        # Remove complete duplicate rows
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)

        # Validate target column
        if self.target_col in df_clean.columns:
            # Ensure price is numeric and non-negative
            df_clean[self.target_col] = pd.to_numeric(df_clean[self.target_col], errors="coerce")
            df_clean = df_clean.dropna(subset=[self.target_col])
            df_clean = df_clean[df_clean[self.target_col] > 0]

        # Convert Date if present
        if "Date" in df_clean.columns:
            df_clean["Date"] = pd.to_datetime(df_clean["Date"], errors="coerce")
            # If any date is NaT, forward fill or replace with median date
            if df_clean["Date"].isna().any():
                df_clean["Date"] = df_clean["Date"].bfill().ffill()

        # Handle missing numerical values with median
        num_candidates = df_clean.select_dtypes(include=[np.number]).columns
        for col in num_candidates:
            if df_clean[col].isna().any():
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())

        # Handle missing categorical values with mode
        cat_candidates = df_clean.select_dtypes(include=["object", "category"]).columns
        for col in cat_candidates:
            if df_clean[col].isna().any():
                mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else "Unknown"
                df_clean[col] = df_clean[col].fillna(mode_val)

        return df_clean

    def fit_transform(self, df, categorical_cols=None, numerical_cols=None):
        """Fits encoders and scalers on features and returns transformed DataFrame."""
        df_proc = df.copy()

        # Separate target if present
        y = None
        if self.target_col in df_proc.columns:
            y = df_proc[self.target_col].values
            X = df_proc.drop(columns=[self.target_col])
        else:
            X = df_proc

        # Drop identifiers not relevant for direct modeling
        drop_cols = [c for c in ["Product_ID", "Product_Name", "Date"] if c in X.columns]
        X = X.drop(columns=drop_cols)

        # Identify categorical and numerical columns
        if categorical_cols is None:
            self.categorical_cols = list(X.select_dtypes(include=["object", "category"]).columns)
        else:
            self.categorical_cols = categorical_cols

        if numerical_cols is None:
            self.numerical_cols = [c for c in X.select_dtypes(include=[np.number]).columns if c != self.target_col]
        else:
            self.numerical_cols = numerical_cols

        # Encode categoricals
        for col in self.categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le

        self.feature_columns = list(X.columns)

        # Fit Scaler
        if self.numerical_cols:
            self.scaler.fit(X[self.numerical_cols])

        self.fitted = True
        return X, y

    def transform(self, df):
        """Transforms unseen data using previously fitted encoders."""
        if not self.fitted:
            raise ValueError("DataPreprocessor must be fitted before calling transform().")

        df_proc = df.copy()
        drop_cols = [c for c in ["Product_ID", "Product_Name", "Date", self.target_col] if c in df_proc.columns]
        X = df_proc.drop(columns=drop_cols, errors="ignore")

        # Encode categorical columns with unseen-value handling
        for col in self.categorical_cols:
            if col in X.columns:
                le = self.label_encoders[col]
                # Map unseen categories to an existing class (or mode)
                known_classes = set(le.classes_)
                X[col] = X[col].astype(str).map(lambda val: val if val in known_classes else le.classes_[0])
                X[col] = le.transform(X[col])
            else:
                X[col] = 0

        # Ensure all training feature columns exist in the same order
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0

        X = X[self.feature_columns]
        return X

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Splits data into train and test sets."""
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def save(self, filepath):
        """Saves preprocessor object."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"[OK] Preprocessor saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """Loads preprocessor object."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor file not found at: {filepath}")
        return joblib.load(filepath)


def prepare_and_save_data(raw_path, processed_path, preprocessor_path):
    """Convenience pipeline function to clean raw data and persist processed dataset."""
    preprocessor = DataPreprocessor(target_col="Price")
    raw_df = preprocessor.load_data(raw_path)
    cleaned_df = preprocessor.clean_data(raw_df)

    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    cleaned_df.to_csv(processed_path, index=False)
    print(f"[OK] Cleaned dataset saved with shape {cleaned_df.shape} to {processed_path}")

    return cleaned_df
