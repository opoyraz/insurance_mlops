"""
Crash hotspot detection using KMeans clustering.
Filters injury/fatal crashes, computes cluster centers,
and calculates geodesic distances from households to hotspots.
"""

import math
import pandas as pd
from sklearn.cluster import KMeans
from geopy.distance import geodesic


def filter_injury_crashes(crashes_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to crashes with injuries or fatalities.

    Returns:
        DataFrame with only injury/fatal crashes (excludes property-only).
    """
    killed = crashes_df[crashes_df["injuries_fatal"] > 0]
    injured = crashes_df[
        (crashes_df["injuries_total"] > 0)
        & (crashes_df["injuries_fatal"] == 0)
    ]
    result = pd.concat([killed, injured], ignore_index=True)
    print(f"  Injury/fatal crashes: {len(result):,}")
    return result


def compute_clusters(
    injury_df: pd.DataFrame, n_clusters: int = 6
) -> pd.DataFrame:
    """
    Run KMeans clustering on crash coordinates.

    Args:
        injury_df: DataFrame with latitude and longitude columns.
        n_clusters: Number of clusters (K).

    Returns:
        DataFrame with cluster centers: cluster, latitude, longitude, cnt.
    """
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(injury_df[["longitude", "latitude"]])

    results_df = injury_df[["latitude", "longitude"]].copy()
    results_df["cluster"] = km.labels_

    cluster_summary = (
        results_df.groupby("cluster")
        .agg(
            longitude=("longitude", "mean"),
            latitude=("latitude", "mean"),
            cnt=("cluster", "count"),
        )
        .reset_index()
        .sort_values("cnt", ascending=False)
    )
    print(f"  Computed {n_clusters} clusters")
    return cluster_summary


def compute_hotspot_distances(
    cluster_summary: pd.DataFrame, hh_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute geodesic distance (miles) from each household to each cluster center.

    Adds HOTSPOT1 through HOTSPOT{n_clusters} columns to hh_df.

    Args:
        cluster_summary: Cluster centers with latitude, longitude columns.
        hh_df: Household coordinates with LATITUDE, LONGITUDE columns.

    Returns:
        hh_df with HOTSPOT distance columns added.
    """
    cluster_coords = cluster_summary[["latitude", "longitude"]].values
    hh_coords = hh_df[["LATITUDE", "LONGITUDE"]].values

    for i in range(len(cluster_summary)):
        col = f"HOTSPOT{i + 1}"
        distances = [
            math.floor(0.5 + geodesic(hh, cluster_coords[i]).miles)
            for hh in hh_coords
        ]
        hh_df[col] = distances
        print(f"  Computed {col}")

    return hh_df
