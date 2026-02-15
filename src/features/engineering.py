"""
Feature engineering and preprocessing pipeline.
Defines feature groups, transformations, and the sklearn ColumnTransformer.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# ── Feature Definitions ──

HOTSPOT_FEATURES = [
    "HOTSPOT1", "HOTSPOT2", "HOTSPOT3",
    "HOTSPOT4", "HOTSPOT5", "HOTSPOT6",
]

NUMERIC_FEATURES = [
    "NB_DRIVERS", "UNDER25", "OVER60", "MALE25",
    "PRIM_DRIVER_AGE", "MODEL_YEAR", "INITIAL_ODOMETER",
    "NB_CLAIMS", "CLAIM_TOTAL", "NB_PRIM_CLAIMS",
    "PRIM_CLAIM_TOTAL", "VEHICLE_AGE",
]

CATEGORICAL_FEATURES = ["PRIM_DRIVER_GENDER"]
BINARY_FEATURES = ["LOW_MILEAGE_USE"]

DROP_COLUMNS = [
    "ZIPCODE", "MAKE_MODEL", "HOUSEHOLD_ID",
    "POLICY_ID", "LAST_NAME", "LATITUDE", "LONGITUDE",
]


def prepare_features(df: pd.DataFrame):
    """
    Clean data and separate features (X) from target (y).

    Steps:
        1. Fill missing claims with 0
        2. Convert LOW_MILEAGE_USE to int
        3. Add VEHICLE_AGE derived feature
        4. Drop non-feature columns
        5. Split into X and y

    Returns:
        Tuple of (X, y) DataFrames.
    """
    df = df.copy()

    # Fill missing claims with 0
    claim_cols = [
        "NB_CLAIMS", "CLAIM_TOTAL",
        "NB_PRIM_CLAIMS", "PRIM_CLAIM_TOTAL",
    ]
    for col in claim_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    # Convert binary column
    df["LOW_MILEAGE_USE"] = df["LOW_MILEAGE_USE"].astype(int)

    # Derived feature
    df["VEHICLE_AGE"] = 2025 - df["MODEL_YEAR"]

    # Separate target from features
    y = df["RISK"]
    X = df.drop(columns=DROP_COLUMNS + ["RISK"], errors="ignore")

    print(f"Features prepared: {X.shape[1]} features, {len(X):,} samples")
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """
    Build the sklearn preprocessing pipeline.

    Applies:
        - StandardScaler to numeric and hotspot features
        - OneHotEncoder to categorical features
        - Passthrough for binary features

    Returns:
        Configured ColumnTransformer.
    """
    all_numeric = HOTSPOT_FEATURES + NUMERIC_FEATURES
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), all_numeric),
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("bin", "passthrough", BINARY_FEATURES),
        ],
        remainder="drop",
    )
