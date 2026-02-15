"""
Model Governance: Bias Detection, Data Drift, Model Drift, Explainability.
Used by Streamlit governance tab.
"""

import pandas as pd
import numpy as np
import shap
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_difference,
    equalized_odds_difference,
)
from sklearn.metrics import mean_absolute_error, r2_score


# ═══════════════════════════════════════════════════════════
# 1. ALGORITHMIC BIAS DETECTION
# ═══════════════════════════════════════════════════════════

def compute_bias_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    sensitive_features: pd.DataFrame,
) -> dict:
    """
    Compute fairness metrics across sensitive groups.

    Args:
        y_true: Actual risk scores.
        y_pred: Predicted risk scores.
        sensitive_features: DataFrame with columns like PRIM_DRIVER_GENDER.

    Returns:
        Dict with group-level metrics and fairness summary.
    """
    results = {}

    for col in sensitive_features.columns:
        metric_frame = MetricFrame(
            metrics={
                "MAE": mean_absolute_error,
                "Mean Prediction": lambda y, p: np.mean(p),
                "R2": r2_score,
            },
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features[col],
        )

        results[col] = {
            "by_group": metric_frame.by_group.reset_index(),
            "overall": metric_frame.overall.to_dict(),
            "difference": metric_frame.difference().to_dict(),
        }

    return results


def create_age_groups(ages: pd.Series) -> pd.Series:
    """Bin ages into groups for bias analysis."""
    bins = [0, 25, 35, 50, 65, 120]
    labels = ["16-25", "26-35", "36-50", "51-65", "65+"]
    return pd.cut(ages, bins=bins, labels=labels)


# ═══════════════════════════════════════════════════════════
# 2. DATA DRIFT DETECTION
# ═══════════════════════════════════════════════════════════

def compute_drift_metrics(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_columns: list,
) -> pd.DataFrame:
    """
    Compare feature distributions between reference and current data.
    Uses simple statistical tests: mean shift, std shift, KS-like range check.

    Args:
        reference_df: Training data (baseline).
        current_df: New/current data to compare.
        feature_columns: List of numeric feature columns to check.

    Returns:
        DataFrame with drift metrics per feature.
    """
    drift_results = []

    for col in feature_columns:
        if col not in reference_df.columns or col not in current_df.columns:
            continue

        ref = pd.to_numeric(reference_df[col], errors="coerce").dropna()
        cur = pd.to_numeric(current_df[col], errors="coerce").dropna()

        if len(ref) == 0 or len(cur) == 0:
            continue

        ref_mean = ref.mean()
        cur_mean = cur.mean()
        ref_std = ref.std()
        cur_std = cur.std()

        # Mean shift as percentage of reference std
        if ref_std > 0:
            mean_shift = abs(cur_mean - ref_mean) / ref_std
        else:
            mean_shift = 0.0

        # Std ratio
        if ref_std > 0:
            std_ratio = cur_std / ref_std
        else:
            std_ratio = 1.0

        # Flag drift if mean shifted more than 0.5 std or std changed by 50%
        drifted = mean_shift > 0.5 or abs(std_ratio - 1.0) > 0.5

        drift_results.append({
            "Feature": col,
            "Ref Mean": round(ref_mean, 2),
            "Cur Mean": round(cur_mean, 2),
            "Mean Shift (std)": round(mean_shift, 3),
            "Ref Std": round(ref_std, 2),
            "Cur Std": round(cur_std, 2),
            "Std Ratio": round(std_ratio, 3),
            "Drift Detected": drifted,
        })

    return pd.DataFrame(drift_results)


# ═══════════════════════════════════════════════════════════
# 3. MODEL DRIFT (PREDICTION MONITORING)
# ═══════════════════════════════════════════════════════════

def compute_prediction_drift(
    reference_preds: np.ndarray,
    current_preds: np.ndarray,
) -> dict:
    """
    Compare prediction distributions between reference and current.

    Args:
        reference_preds: Predictions on training/reference data.
        current_preds: Predictions on new/current data.

    Returns:
        Dict with distribution stats and drift flag.
    """
    ref_mean = np.mean(reference_preds)
    cur_mean = np.mean(current_preds)
    ref_std = np.std(reference_preds)
    cur_std = np.std(current_preds)

    if ref_std > 0:
        mean_shift = abs(cur_mean - ref_mean) / ref_std
    else:
        mean_shift = 0.0

    return {
        "ref_mean": round(ref_mean, 2),
        "cur_mean": round(cur_mean, 2),
        "ref_std": round(ref_std, 2),
        "cur_std": round(cur_std, 2),
        "mean_shift_std": round(mean_shift, 3),
        "drift_detected": mean_shift > 0.5,
    }


# ═══════════════════════════════════════════════════════════
# 4. EXPLAINABILITY (SHAP)
# ═══════════════════════════════════════════════════════════

def compute_shap_values(model, X_sample: pd.DataFrame) -> tuple:
    """
    Compute SHAP values for model explainability.

    Args:
        model: Fitted sklearn Pipeline.
        X_sample: Sample of input features (raw, before preprocessing).

    Returns:
        Tuple of (shap_values, expected_value, feature_names).
    """
    # Get the preprocessor and model from pipeline
    preprocessor = model.named_steps["preprocessor"]
    gb_model = model.named_steps["model"]

    # Transform the sample
    X_transformed = preprocessor.transform(X_sample)
    feature_names = preprocessor.get_feature_names_out()

    # Create SHAP explainer on the tree model directly
    explainer = shap.TreeExplainer(gb_model)
    shap_values = explainer.shap_values(X_transformed)

    ev = explainer.expected_value
    if hasattr(ev, '__len__'):
        ev = float(ev[0])
    return shap_values, ev, feature_names
