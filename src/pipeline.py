"""
Production Training Pipeline
Chains all src/ modules: load → fetch → validate → cluster → train → save

Usage:
    python -m src.pipeline                  # uses cached crash data if available
    python -m src.pipeline --refresh        # force re-fetch crash data from Socrata

Local driver and household data is always read from data/raw/.
Crash data is fetched from Socrata API (or cached from previous run).
"""

import os
import sys
import time
import joblib
import pandas as pd
from pathlib import Path

from src.data.socrata_client import get_crash_data
from src.data.validation import validate_crash_data, validate_driver_data
from src.features.clustering import (
    filter_injury_crashes,
    compute_clusters,
    compute_hotspot_distances,
)
from src.models.train import train


def run_pipeline(
    months_back: int = 6,
    n_clusters: int = 6,
    save_dir: str = "data",
    refresh: bool = False,
):
    """
    Run the full training pipeline end-to-end.

    Steps:
        1. Load driver + household data from local CSVs
        2. Fetch crash data from Socrata (or load cached)
        3. Validate all data
        4. Cluster crash hotspots (KMeans)
        5. Compute household-to-cluster distances
        6. Merge driver data with hotspot distances
        7. Train model with MLflow tracking
        8. Save model and artifacts

    Args:
        months_back: Months of crash data to fetch from Socrata.
        n_clusters: Number of hotspot clusters (K for KMeans).
        save_dir: Base directory for saving outputs.
        refresh: If True, force re-fetch crash data from Socrata.
    """
    t_start = time.time()
    raw_dir = Path(save_dir) / "raw"
    processed_dir = Path(save_dir) / "processed"
    model_dir = Path(save_dir) / "model"
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # ── Step 1: Load Local Data ──
    print("=" * 60)
    print("STEP 1: Loading local driver and household data")
    print("=" * 60)

    driver_path = raw_dir / "drivers_raw.csv"
    hh_path = raw_dir / "household_coords.csv"

    if not driver_path.exists():
        raise FileNotFoundError(f"Driver data not found: {driver_path}")
    if not hh_path.exists():
        raise FileNotFoundError(f"Household data not found: {hh_path}")

    driver_df = pd.read_csv(driver_path)
    hh_df = pd.read_csv(hh_path)
    print(f"  Drivers:    {driver_df.shape}")
    print(f"  Households: {hh_df.shape}")

    # ── Step 2: Fetch or Load Crash Data ──
    print("\n" + "=" * 60)
    print("STEP 2: Crash data ingestion")
    print("=" * 60)

    crash_cache = processed_dir / "crashes_cleaned.csv"

    if not refresh and crash_cache.exists():
        print("  Cached crash data found — loading from CSV")
        print("  (Run with --refresh to re-fetch from Socrata)")
        crashes_df = pd.read_csv(crash_cache)
        print(f"  Crashes: {crashes_df.shape}")
    else:
        if refresh:
            print("  Refresh flag set — fetching fresh crash data")
        else:
            print("  No cached crash data — fetching from Socrata")
        crashes_df = get_crash_data(months_back=months_back)
        crashes_df.to_csv(crash_cache, index=False)
        print("  Saved crash data to cache")

    # ── Step 3: Validate ──
    print("\n" + "=" * 60)
    print("STEP 3: Validating data")
    print("=" * 60)

    crashes_df = validate_crash_data(crashes_df)
    driver_df = validate_driver_data(driver_df, require_hotspots=False)

    # ── Step 4: Cluster ──
    print("\n" + "=" * 60)
    print("STEP 4: Computing hotspot clusters")
    print("=" * 60)

    injury_df = filter_injury_crashes(crashes_df)
    cluster_summary = compute_clusters(injury_df, n_clusters=n_clusters)
    cluster_summary.to_csv(processed_dir / "cluster_summary.csv", index=False)

    # ── Step 5: Hotspot Distances ──
    print("\n" + "=" * 60)
    print("STEP 5: Computing hotspot distances (this takes a few minutes)")
    print("=" * 60)

    hh_df = compute_hotspot_distances(cluster_summary, hh_df)

    # ── Step 6: Merge ──
    print("\n" + "=" * 60)
    print("STEP 6: Merging driver data with hotspot distances")
    print("=" * 60)

    merged_df = driver_df.merge(hh_df, on="HOUSEHOLD_ID", how="inner")
    merged_df.to_csv(processed_dir / "training_data.csv", index=False)
    print(f"  Merged dataset: {merged_df.shape}")

    # Validate merged data (now with hotspots)
    merged_df = validate_driver_data(merged_df, require_hotspots=True)

    # ── Step 7: Train ──
    print("\n" + "=" * 60)
    print("STEP 7: Training model")
    print("=" * 60)

    pipeline, metrics = train(merged_df)

    # ── Step 8: Save ──
    print("\n" + "=" * 60)
    print("STEP 8: Saving model")
    print("=" * 60)

    model_path = model_dir / "model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"  Saved model to {model_path}")

    # ── Summary ──
    t_end = time.time()
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total time:  {(t_end - t_start) / 60:.1f} minutes")
    print(f"  Crash data:  {'Fresh fetch' if refresh else 'Cached'}")
    print(f"  Samples:     {len(merged_df):,}")
    print(f"  Clusters:    {n_clusters}")
    print(f"  MAE:         {metrics['mae']:.2f}")
    print(f"  RMSE:        {metrics['rmse']:.2f}")
    print(f"  R2:          {metrics['r2']:.3f}")
    print(f"  Model:       {model_path}")

    return pipeline, metrics


if __name__ == "__main__":
    refresh_flag = "--refresh" in sys.argv
    run_pipeline(refresh=refresh_flag)
