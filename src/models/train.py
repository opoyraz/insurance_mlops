"""
Model training with MLflow experiment tracking.
Trains a Gradient Boosting Regressor inside an sklearn Pipeline.
"""

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from src.config import settings
from src.features.engineering import prepare_features, build_preprocessor


def evaluate_model(pipeline: Pipeline, X_test, y_test) -> dict:
    """Evaluate model on test set and return metrics."""
    y_pred = pipeline.predict(X_test)
    return {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": root_mean_squared_error(y_test, y_pred),
        "r2": r2_score(y_test, y_pred),
    }


def train(df: pd.DataFrame) -> tuple:
    """
    Train the model and log to MLflow.

    Steps:
        1. Prepare features (clean, encode, split X/y)
        2. Train/test split (80/20)
        3. Build preprocessor + model pipeline
        4. Cross-validate on training set
        5. Evaluate on test set
        6. Log everything to MLflow

    Args:
        df: Merged DataFrame with driver data and hotspot distances.

    Returns:
        Tuple of (fitted pipeline, metrics dict).
    """
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("insurance-risk-score")

    X, y = prepare_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor()
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    with mlflow.start_run():
        # Cross-validation on training set
        cv_scores = cross_val_score(
            pipeline, X_train, y_train,
            cv=5, scoring="neg_mean_absolute_error",
        )
        cv_mae = -cv_scores.mean()

        # Fit on full training set
        pipeline.fit(X_train, y_train)

        # Evaluate on test set
        metrics = evaluate_model(pipeline, X_test, y_test)
        metrics["cv_mae"] = cv_mae

        # Log to MLflow
        mlflow.log_params({
            "model": "GradientBoosting",
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.2,
            "test_size": 0.2,
            "n_features": X.shape[1],
            "n_samples": len(X),
        })
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(
            pipeline,
            "Gradient Boosting model",
            registered_model_name="insurance-risk-model",
        )

        print(f"  CV MAE:    {cv_mae:.2f}")
        print(f"  Test MAE:  {metrics['mae']:.2f}")
        print(f"  Test RMSE: {metrics['rmse']:.2f}")
        print(f"  Test R2:   {metrics['r2']:.3f}")

    return pipeline, metrics
