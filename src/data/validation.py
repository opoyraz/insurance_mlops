"""
Data validation for crash and driver DataFrames.
Checks required columns, types, and null thresholds.
"""

import pandas as pd


def validate_crash_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate crash data from Socrata.

    Checks:
        - Required columns exist
        - latitude/longitude are numeric and non-null
        - injuries_total and injuries_fatal are numeric

    Returns:
        Cleaned DataFrame with nulls dropped for critical columns.
    """
    required = ["latitude", "longitude", "injuries_total", "injuries_fatal"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required crash columns: {missing}")

    # Convert to numeric
    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows missing coordinates
    before = len(df)
    df = df.dropna(subset=["latitude", "longitude"])
    after = len(df)
    if before != after:
        print(f"  Dropped {before - after} rows missing coordinates")

    return df


def validate_driver_data(
    df: pd.DataFrame, require_hotspots: bool = False
) -> pd.DataFrame:
    """
    Validate driver data.

    Args:
        df: Driver DataFrame.
        require_hotspots: If True, also check for HOTSPOT1-6 columns.
            Set to False before clustering, True after merging distances.

    Returns:
        Validated DataFrame.
    """
    required = [
        "HOUSEHOLD_ID", "RISK", "PRIM_DRIVER_AGE",
        "PRIM_DRIVER_GENDER", "MODEL_YEAR",
    ]

    if require_hotspots:
        required += [
            "HOTSPOT1", "HOTSPOT2", "HOTSPOT3",
            "HOTSPOT4", "HOTSPOT5", "HOTSPOT6",
        ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required driver columns: {missing}")

    print(f"  Validation passed: {len(df):,} rows, {len(df.columns)} columns")
    return df
