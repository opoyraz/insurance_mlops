"""
Centralized configuration loaded from .env file.
Uses pydantic-settings for validation and type safety.
"""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    # ---- Socrata ----
    socrata_app_token: str | None = Field(
        None, description="Socrata API token (optional, increases rate limit)"
    )
    socrata_domain: str = Field("data.cityofchicago.org")
    socrata_dataset_id: str = Field("85ca-t3if")

    # ---- MLflow ----
    mlflow_tracking_uri: str = Field(
        "mlruns", description="MLflow tracking URI"
    )

    # ---- API ----
    api_host: str = Field("0.0.0.0")
    api_port: int = Field(8000)

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Singleton instance — import this everywhere
settings = Settings()
