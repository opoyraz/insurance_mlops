"""
Pydantic models for API request and response validation.
"""

from pydantic import BaseModel, Field


class PredictionInput(BaseModel):
    """Input schema for risk prediction."""

    prim_driver_age: int = Field(..., ge=16, le=120, description="Primary driver age")
    prim_driver_gender: str = Field(..., pattern="^(M|F)$", description="M or F")
    model_year: int = Field(..., ge=1950, le=2026, description="Vehicle model year")
    initial_odometer: int = Field(..., ge=0, description="Current odometer reading")
    low_mileage_use: bool = Field(..., description="Low mileage vehicle")
    nb_drivers: int = Field(..., ge=1, le=10, description="Number of drivers in household")
    under25: int = Field(0, ge=0, description="Drivers under 25")
    over60: int = Field(0, ge=0, description="Drivers over 60")
    male25: int = Field(0, ge=0, description="Male drivers under 25")
    nb_claims: int = Field(0, ge=0, description="Total claims")
    claim_total: float = Field(0.0, ge=0, description="Total claim amount")
    nb_prim_claims: int = Field(0, ge=0, description="Primary driver claims")
    prim_claim_total: float = Field(0.0, ge=0, description="Primary driver claim total")
    hotspot1: int = Field(..., ge=0, description="Distance (miles) to hotspot 1")
    hotspot2: int = Field(..., ge=0, description="Distance (miles) to hotspot 2")
    hotspot3: int = Field(..., ge=0, description="Distance (miles) to hotspot 3")
    hotspot4: int = Field(..., ge=0, description="Distance (miles) to hotspot 4")
    hotspot5: int = Field(..., ge=0, description="Distance (miles) to hotspot 5")
    hotspot6: int = Field(..., ge=0, description="Distance (miles) to hotspot 6")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "prim_driver_age": 35,
                    "prim_driver_gender": "M",
                    "model_year": 2020,
                    "initial_odometer": 45000,
                    "low_mileage_use": False,
                    "nb_drivers": 2,
                    "under25": 0,
                    "over60": 0,
                    "male25": 0,
                    "nb_claims": 1,
                    "claim_total": 3500.0,
                    "nb_prim_claims": 1,
                    "prim_claim_total": 3500.0,
                    "hotspot1": 5,
                    "hotspot2": 8,
                    "hotspot3": 12,
                    "hotspot4": 15,
                    "hotspot5": 10,
                    "hotspot6": 12,
                }
            ]
        }
    }


class PredictionOutput(BaseModel):
    """Output schema for risk prediction."""

    risk_score: float = Field(..., description="Predicted risk score")
    risk_level: str = Field(..., description="LOW, MEDIUM, or HIGH")
