"""
Prediction logic: maps API input to model-compatible DataFrame.
"""

import pandas as pd


def classify_risk(score: float) -> str:
    """Classify risk score into LOW, MEDIUM, or HIGH."""
    if score < 20:
        return "LOW"
    elif score < 35:
        return "MEDIUM"
    return "HIGH"


def predict(input_data: dict, model) -> dict:
    """
    Run prediction using the trained pipeline.

    Args:
        input_data: Dictionary from PredictionInput.model_dump().
        model: Fitted sklearn Pipeline.

    Returns:
        Dict with risk_score and risk_level.
    """
    df = pd.DataFrame([{
        "HOTSPOT1": input_data["hotspot1"],
        "HOTSPOT2": input_data["hotspot2"],
        "HOTSPOT3": input_data["hotspot3"],
        "HOTSPOT4": input_data["hotspot4"],
        "HOTSPOT5": input_data["hotspot5"],
        "HOTSPOT6": input_data["hotspot6"],
        "NB_DRIVERS": input_data["nb_drivers"],
        "UNDER25": input_data["under25"],
        "OVER60": input_data["over60"],
        "MALE25": input_data["male25"],
        "PRIM_DRIVER_AGE": input_data["prim_driver_age"],
        "PRIM_DRIVER_GENDER": input_data["prim_driver_gender"],
        "MODEL_YEAR": input_data["model_year"],
        "INITIAL_ODOMETER": input_data["initial_odometer"],
        "NB_CLAIMS": input_data["nb_claims"],
        "CLAIM_TOTAL": input_data["claim_total"],
        "NB_PRIM_CLAIMS": input_data["nb_prim_claims"],
        "PRIM_CLAIM_TOTAL": input_data["prim_claim_total"],
        "LOW_MILEAGE_USE": int(input_data["low_mileage_use"]),
    }])

    # Derived feature
    df["VEHICLE_AGE"] = 2025 - df["MODEL_YEAR"]

    prediction = model.predict(df)[0]
    return {
        "risk_score": round(float(prediction), 1),
        "risk_level": classify_risk(prediction),
    }
