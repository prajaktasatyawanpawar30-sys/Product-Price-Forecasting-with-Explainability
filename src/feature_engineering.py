"""
Feature Engineering Module
Product Price Forecasting with Explainability

Creates domain-specific features including discount metrics, competitor price differences,
brand tiering, demand ratios, and temporal/calendar indicators.
"""

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Creates derived features for price forecasting and explainability."""

    def __init__(self):
        self.brand_tiers = {
            "Asics": "Premium",
            "Nike": "Premium",
            "Under Armour": "Premium",
            "Adidas": "Premium",
            "Woodland": "Mid-High",
            "Skechers": "Mid-Range",
            "Puma": "Mid-Range",
            "Reebok": "Mid-Range",
            "Bata": "Value",
            "Campus": "Budget"
        }

    def add_temporal_features(self, df):
        """Extracts calendar components from Date column."""
        df_feat = df.copy()
        if "Date" in df_feat.columns:
            date_series = pd.to_datetime(df_feat["Date"], errors="coerce")
            df_feat["Year"] = date_series.dt.year.fillna(2024).astype(int)
            df_feat["Month"] = date_series.dt.month.fillna(1).astype(int)
            df_feat["Day"] = date_series.dt.day.fillna(1).astype(int)
            df_feat["DayOfWeek"] = date_series.dt.dayofweek.fillna(0).astype(int)
            df_feat["Quarter"] = date_series.dt.quarter.fillna(1).astype(int)
            df_feat["Is_Weekend"] = df_feat["DayOfWeek"].isin([5, 6]).astype(int)

            # Assign season
            month = df_feat["Month"]
            conditions = [
                month.isin([12, 1, 2]),
                month.isin([3, 4, 5]),
                month.isin([6, 7, 8, 9]),
                month.isin([10, 11])
            ]
            seasons = ["Winter", "Summer", "Monsoon", "Autumn"]
            df_feat["Season"] = np.select(conditions, seasons, default="All-Season")
        return df_feat

    def add_domain_features(self, df):
        """Constructs business and pricing indicators."""
        df_feat = df.copy()

        # 1. Discount Indicators
        if "Discount_Percentage" in df_feat.columns and "Original_Price" in df_feat.columns:
            df_feat["Discount_Amount"] = (
                df_feat["Original_Price"] * (df_feat["Discount_Percentage"] / 100.0)
            )
            df_feat["Effective_Discount_Ratio"] = df_feat["Discount_Percentage"] / 100.0

        # 2. Competitor Price Gap
        if "Original_Price" in df_feat.columns and "Competitor_Price" in df_feat.columns:
            df_feat["Competitor_Diff"] = df_feat["Original_Price"] - df_feat["Competitor_Price"]
            df_feat["Competitor_Ratio"] = df_feat["Original_Price"] / (df_feat["Competitor_Price"] + 1e-5)

        # 3. Product Popularity & Social Proof
        if "Rating" in df_feat.columns and "Reviews_Count" in df_feat.columns:
            df_feat["Popularity_Score"] = df_feat["Rating"] * np.log1p(df_feat["Reviews_Count"])

        # 4. Inventory Velocity / Demand Pressure
        if "Sales_Quantity" in df_feat.columns and "Stock_Quantity" in df_feat.columns:
            df_feat["Demand_Stock_Ratio"] = df_feat["Sales_Quantity"] / (df_feat["Stock_Quantity"] + 1.0)

        # 5. Brand Tier Categorization
        if "Brand" in df_feat.columns:
            df_feat["Brand_Tier"] = df_feat["Brand"].map(self.brand_tiers).fillna("Standard")

        return df_feat

    def transform(self, df):
        """Full feature engineering pipeline."""
        df_trans = self.add_temporal_features(df)
        df_trans = self.add_domain_features(df_trans)
        return df_trans


def run_feature_engineering(df):
    """Utility function to apply full feature transformations."""
    fe = FeatureEngineer()
    return fe.transform(df)
