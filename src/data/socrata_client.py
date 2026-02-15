"""
Fetch Chicago Traffic Crashes data from Socrata Open Data API."""
# Section 1 — Docstring and imports

import pandas as pd 
from sodapy import Socrata
from datetime import datetime, timedelta

from src.config import settings

def fetch_crash_data(
        months_back: int = 6,
        end_date: datetime | None = None,
        limit: int = 10000) -> pd.DataFrame:
    """  
    Fetch Chicago crash data from Socrata API with pagination.
    Args:
        months_back: Number of months of data to fetch.
        end_date: End date for the query. Defaults to today.
        limit: Records per API request (pagination size).
    
    """
    if end_date is None:
        end_date = datetime.now()

    start_date = end_date - timedelta(days=months_back * 30)
    start_str = start_date.strftime("%Y-%m-%dT00:00:00")
    end_str = end_date.strftime("%Y-%m-%dT23:59:59")

    client = Socrata(
        settings.socrata_domain,
        settings.socrata_app_token,
    )

    results = []
    offset = 0

    while True:
        batch = client.get(
            settings.socrata_dataset_id,
            where=f"crash_date >= '{start_str}' AND crash_date <= '{end_str}'",
            limit=limit,
            offset=offset 
        )
        results.extend(batch)

        if len(batch) < limit:
            break 
        offset += limit

    client.close()

    df = pd.DataFrame.from_records(results)
    print(f"Fetched {len(df):,} crash records from Socrata")
    return df

# Section 3 — The clean function

def clean_crash_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean raw crash data: select columns, convert types, drop nulls.

    Args:
        df: Raw crash DataFrame from Socrata.

    Returns:
        Cleaned DataFrame with injuries_fatal, injuries_total, latitude, longitude.
    """

    columns = ['injuries_fatal', 'injuries_total','latitude','longitude']
    crashes_df = df[columns].copy()

    crashes_df = crashes_df.astype({
        'injuries_fatal': float,
        'injuries_total': float,
        'latitude': float,
        'longitude': float,
    })

    crashes_df = crashes_df.dropna()
    crashes_df = crashes_df[
        (crashes_df["longitude"] !=0) & (crashes_df["latitude"] !=0)

    ]

    print(f"Cleaned crash data: {len(crashes_df):,} records")
    return crashes_df.reset_index(drop=True)

# Section 4 — Single entry point

def get_crash_data(months_back: int =6) -> pd.DataFrame:
    """Full pipeline: fetch + clean. Single entry point for the pipeline."""
    raw = fetch_crash_data(months_back=months_back)
    return clean_crash_data(raw)

