"""Feature engineering for churn prediction."""

import pandas as pd
import numpy as np


NUMERIC_FEATURES = [
    "age", "tenure_months", "annual_claims", "er_visits", "rx_fills",
    "preventive_visits", "specialist_visits", "portal_logins_6m",
    "mobile_sessions_6m", "calls_to_service_6m", "grievances_filed",
    "total_cost_ytd", "member_oop_ytd", "monthly_premium", "satisfaction_score",
]

CATEGORICAL_FEATURES = ["gender", "region", "plan_type", "income_bracket"]

TARGET = "churned"


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["digital_engagement"] = df["portal_logins_6m"] + df["mobile_sessions_6m"]
    df["cost_burden"] = np.where(
        df["monthly_premium"] > 0,
        df["member_oop_ytd"] / (df["monthly_premium"] * 12),
        0,
    )
    df["service_friction"] = df["calls_to_service_6m"] + df["grievances_filed"] * 3
    df["utilization_total"] = (
        df["annual_claims"] + df["er_visits"] + df["rx_fills"]
        + df["preventive_visits"] + df["specialist_visits"]
    )
    df["engagement_ratio"] = np.where(
        df["utilization_total"] > 0,
        df["digital_engagement"] / df["utilization_total"],
        df["digital_engagement"],
    )
    df["is_new_member"] = (df["tenure_months"] <= 12).astype(int)
    df["has_grievance"] = (df["grievances_filed"] > 0).astype(int)
    return df


DERIVED_FEATURES = [
    "digital_engagement", "cost_burden", "service_friction",
    "utilization_total", "engagement_ratio", "is_new_member", "has_grievance",
]


def prepare_model_data(df: pd.DataFrame):
    df = add_derived_features(df)

    feature_cols = NUMERIC_FEATURES + DERIVED_FEATURES
    df_encoded = pd.get_dummies(df, columns=CATEGORICAL_FEATURES, drop_first=True)

    cat_dummies = [c for c in df_encoded.columns if any(c.startswith(f + "_") for f in CATEGORICAL_FEATURES)]
    all_features = feature_cols + cat_dummies

    X = df_encoded[all_features]
    y = df_encoded[TARGET]

    return X, y, all_features
