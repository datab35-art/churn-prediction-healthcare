"""Model training and evaluation for churn prediction."""

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_curve, roc_curve,
)
from sklearn.preprocessing import StandardScaler


def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        ),
    }


def cross_validate_models(X, y, models=None):
    models = models or get_models()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    for name, model in models.items():
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring="roc_auc")
        results[name] = {
            "mean_auc": scores.mean(),
            "std_auc": scores.std(),
            "scores": scores,
        }

    return results


def train_and_evaluate(X_train, X_test, y_train, y_test, model_name="Gradient Boosting"):
    models = get_models()
    model = models[model_name]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model.fit(X_train_s, y_train)

    y_pred = model.predict(X_test_s)
    y_prob = model.predict_proba(X_test_s)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    precision, recall, thresholds_pr = precision_recall_curve(y_test, y_prob)

    return {
        "model": model,
        "scaler": scaler,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "auc": auc,
        "report": report,
        "confusion_matrix": cm,
        "roc_curve": (fpr, tpr),
        "pr_curve": (precision, recall),
    }


def get_feature_importance(model, feature_names):
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        return pd.DataFrame()

    fi = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return fi


def assign_risk_tiers(probabilities):
    tiers = pd.cut(
        probabilities,
        bins=[0, 0.2, 0.5, 0.8, 1.0],
        labels=["Low", "Medium", "High", "Critical"],
    )
    return tiers
