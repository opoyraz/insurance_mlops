"""
Streamlit Dashboard for Insurance Risk Prediction.
Tabs: Predict Risk | Model Dashboard | Hotspot Map | Batch Upload | Governance
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import folium
from streamlit_folium import st_folium
from pathlib import Path
import matplotlib.pyplot as plt
import shap

from src.governance import (
    compute_bias_metrics,
    create_age_groups,
    compute_drift_metrics,
    compute_prediction_drift,
    compute_shap_values,
)
from src.features.engineering import (
    HOTSPOT_FEATURES,
    NUMERIC_FEATURES,
    prepare_features,
)

# ── Paths ──
MODEL_PATH = Path("data/model/model.joblib")
CLUSTER_PATH = Path("data/processed/cluster_summary.csv")
TRAINING_PATH = Path("data/processed/training_data.csv")

# ── Page Config ──
st.set_page_config(
    page_title="Insurance Risk MLOps",
    page_icon="🛡️",
    layout="wide",
)


# ── Cached Loading ──
@st.cache_resource
def load_model():
    """Load trained model pipeline (cached in memory)."""
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    return None


@st.cache_data
def load_cluster_summary():
    """Load cluster summary CSV (cached)."""
    if CLUSTER_PATH.exists():
        return pd.read_csv(CLUSTER_PATH)
    return None


@st.cache_data
def load_training_data():
    """Load training data for dashboard stats (cached)."""
    if TRAINING_PATH.exists():
        return pd.read_csv(TRAINING_PATH)
    return None


# ── Helper Functions ──
def classify_risk(score: float) -> tuple:
    """Return (level, emoji) tuple."""
    if score < 20:
        return "LOW", "🟢"
    elif score < 35:
        return "MEDIUM", "🟡"
    return "HIGH", "🔴"


def prepare_input(data: dict) -> pd.DataFrame:
    """Convert form input to model-compatible DataFrame."""
    df = pd.DataFrame([{
        "HOTSPOT1": data["hotspot1"],
        "HOTSPOT2": data["hotspot2"],
        "HOTSPOT3": data["hotspot3"],
        "HOTSPOT4": data["hotspot4"],
        "HOTSPOT5": data["hotspot5"],
        "HOTSPOT6": data["hotspot6"],
        "NB_DRIVERS": data["nb_drivers"],
        "UNDER25": data["under25"],
        "OVER60": data["over60"],
        "MALE25": data["male25"],
        "PRIM_DRIVER_AGE": data["prim_driver_age"],
        "PRIM_DRIVER_GENDER": data["prim_driver_gender"],
        "MODEL_YEAR": data["model_year"],
        "INITIAL_ODOMETER": data["initial_odometer"],
        "NB_CLAIMS": data["nb_claims"],
        "CLAIM_TOTAL": data["claim_total"],
        "NB_PRIM_CLAIMS": data["nb_prim_claims"],
        "PRIM_CLAIM_TOTAL": data["prim_claim_total"],
        "LOW_MILEAGE_USE": int(data["low_mileage_use"]),
    }])
    df["VEHICLE_AGE"] = 2025 - df["MODEL_YEAR"]
    return df


def prepare_input_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare batch DataFrame for prediction."""
    df = df.copy()
    if "LOW_MILEAGE_USE" in df.columns:
        df["LOW_MILEAGE_USE"] = df["LOW_MILEAGE_USE"].astype(int)
    df["VEHICLE_AGE"] = 2025 - df["MODEL_YEAR"]
    return df


# ── Load Data ──
model = load_model()
cluster_df = load_cluster_summary()
training_df = load_training_data()

# ── App Title ──
st.title("🛡️ Insurance Risk Prediction")
st.caption("MLOps Portfolio Project — Gradient Boosting + KMeans Clustering")

if model is None:
    st.error("Model not found. Run `python -m src.pipeline` first.")
    st.stop()

# ── Tabs ──
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Predict Risk",
    "📊 Model Dashboard",
    "🗺️ Hotspot Map",
    "📁 Batch Upload",
    "🔍 Model Governance",
])

# ════════════════════════════════════════════════
# TAB 1: PREDICT RISK
# ════════════════════════════════════════════════
with tab1:
    st.header("Single Driver Prediction")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Driver Info")
        age = st.number_input("Driver Age", 16, 120, 35)
        gender = st.selectbox("Gender", ["M", "F"])
        nb_drivers = st.number_input("Drivers in Household", 1, 10, 2)
        under25 = st.number_input("Drivers Under 25", 0, 10, 0)
        over60 = st.number_input("Drivers Over 60", 0, 10, 0)
        male25 = st.number_input("Male Drivers Under 25", 0, 10, 0)

    with col2:
        st.subheader("Vehicle Info")
        model_year = st.number_input("Model Year", 1950, 2026, 2020)
        odometer = st.number_input("Odometer", 0, 500000, 45000)
        low_mileage = st.checkbox("Low Mileage Use")

        st.subheader("Claims History")
        nb_claims = st.number_input("Total Claims", 0, 50, 0)
        claim_total = st.number_input("Total Claim Amount ($)", 0.0, 500000.0, 0.0)
        nb_prim_claims = st.number_input("Primary Driver Claims", 0, 50, 0)
        prim_claim_total = st.number_input("Primary Claim Amount ($)", 0.0, 500000.0, 0.0)

    with col3:
        st.subheader("Hotspot Distances (miles)")
        h1 = st.number_input("Hotspot 1", 0, 100, 5)
        h2 = st.number_input("Hotspot 2", 0, 100, 8)
        h3 = st.number_input("Hotspot 3", 0, 100, 12)
        h4 = st.number_input("Hotspot 4", 0, 100, 15)
        h5 = st.number_input("Hotspot 5", 0, 100, 10)
        h6 = st.number_input("Hotspot 6", 0, 100, 12)

    if st.button("🔮 Predict Risk Score", type="primary", use_container_width=True):
        input_data = {
            "prim_driver_age": age, "prim_driver_gender": gender,
            "model_year": model_year, "initial_odometer": odometer,
            "low_mileage_use": low_mileage, "nb_drivers": nb_drivers,
            "under25": under25, "over60": over60, "male25": male25,
            "nb_claims": nb_claims, "claim_total": claim_total,
            "nb_prim_claims": nb_prim_claims, "prim_claim_total": prim_claim_total,
            "hotspot1": h1, "hotspot2": h2, "hotspot3": h3,
            "hotspot4": h4, "hotspot5": h5, "hotspot6": h6,
        }
        df = prepare_input(input_data)
        score = model.predict(df)[0]
        level, emoji = classify_risk(score)

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk Score", f"{score:.1f}")
        c2.metric("Risk Level", f"{emoji} {level}")
        c3.metric("Vehicle Age", f"{2025 - model_year} years")
        st.progress(min(score / 50, 1.0))

# ════════════════════════════════════════════════
# TAB 2: MODEL DASHBOARD
# ════════════════════════════════════════════════
with tab2:
    st.header("Model Performance Dashboard")

    # Model info
    gb = model.named_steps["model"]
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Hyperparameters")
        st.json({
            "model": "GradientBoostingRegressor",
            "n_estimators": gb.n_estimators,
            "max_depth": gb.max_depth,
            "learning_rate": gb.learning_rate,
        })

    with col2:
        st.subheader("Training Data Stats")
        if training_df is not None:
            st.metric("Samples", f"{len(training_df):,}")
            st.metric("Features", f"{len(training_df.columns) - 1}")
            if "RISK" in training_df.columns:
                st.metric("Risk Range", f"{training_df['RISK'].min():.0f} - {training_df['RISK'].max():.0f}")

    # Feature importance
    st.subheader("Feature Importance")
    preprocessor = model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    importances = gb.feature_importances_

    feat_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances,
    }).sort_values("Importance", ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2196F3" if "HOTSPOT" in f else "#FF9800" for f in feat_df["Feature"]]
    ax.barh(feat_df["Feature"], feat_df["Importance"], color=colors)
    ax.set_xlabel("Importance")
    ax.set_title("Top 15 Features (Blue = Hotspot, Orange = Driver/Vehicle)")
    st.pyplot(fig)
    plt.close()

    # Risk distribution
    if training_df is not None and "RISK" in training_df.columns:
        st.subheader("Risk Score Distribution")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(training_df["RISK"], bins=50, color="#2196F3", alpha=0.7, edgecolor="white")
        ax.axvline(training_df["RISK"].mean(), color="red", linestyle="--", label=f"Mean: {training_df['RISK'].mean():.1f}")
        ax.set_xlabel("Risk Score")
        ax.set_ylabel("Count")
        ax.legend()
        st.pyplot(fig)
        plt.close()

# ════════════════════════════════════════════════
# TAB 3: HOTSPOT MAP
# ════════════════════════════════════════════════
with tab3:
    st.header("🗺️ Chicago Crash Hotspot Map")

    if cluster_df is not None:
        # Create folium map centered on Chicago
        m = folium.Map(
            location=[41.8781, -87.6298],
            zoom_start=11,
            tiles="CartoDB positron",
        )

        # Color palette for clusters
        colors = ["red", "blue", "green", "orange", "purple", "brown"]

        for _, row in cluster_df.iterrows():
            idx = int(row["cluster"])
            color = colors[idx % len(colors)]
            radius = max(row["cnt"] / 100, 5)

            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(
                    f"<b>Cluster {idx + 1}</b><br>"
                    f"Crashes: {int(row['cnt']):,}<br>"
                    f"Lat: {row['latitude']:.4f}<br>"
                    f"Lon: {row['longitude']:.4f}",
                    max_width=200,
                ),
            ).add_to(m)

        # Display map
        st_folium(m, width=900, height=500)

        # Cluster summary table
        st.subheader("Cluster Summary")
        display_df = cluster_df.copy()
        display_df.columns = ["Cluster", "Longitude", "Latitude", "Crash Count"]
        display_df["Cluster"] = display_df["Cluster"].astype(int) + 1
        st.dataframe(
            display_df.sort_values("Crash Count", ascending=False),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.warning("Cluster data not found. Run the pipeline first.")

# ════════════════════════════════════════════════
# TAB 4: BATCH UPLOAD
# ════════════════════════════════════════════════
with tab4:
    st.header("Batch Prediction")

    # Template download
    template_cols = [
        "HOTSPOT1", "HOTSPOT2", "HOTSPOT3", "HOTSPOT4", "HOTSPOT5", "HOTSPOT6",
        "NB_DRIVERS", "UNDER25", "OVER60", "MALE25",
        "PRIM_DRIVER_AGE", "PRIM_DRIVER_GENDER", "MODEL_YEAR",
        "INITIAL_ODOMETER", "NB_CLAIMS", "CLAIM_TOTAL",
        "NB_PRIM_CLAIMS", "PRIM_CLAIM_TOTAL", "LOW_MILEAGE_USE",
    ]
    template_df = pd.DataFrame(columns=template_cols)
    st.download_button(
        "📥 Download CSV Template",
        template_df.to_csv(index=False),
        "prediction_template.csv",
        "text/csv",
    )

    # Upload
    uploaded = st.file_uploader("Upload CSV file", type=["csv"])
    if uploaded:
        batch_df = pd.read_csv(uploaded)
        st.write(f"Uploaded {len(batch_df)} rows")
        st.dataframe(batch_df.head(), use_container_width=True)

        if st.button("🚀 Run Batch Prediction", type="primary"):
            try:
                input_df = prepare_input_batch(batch_df)
                scores = model.predict(input_df)
                batch_df["RISK_SCORE"] = scores.round(1)
                batch_df["RISK_LEVEL"] = [classify_risk(s)[0] for s in scores]

                # Summary
                st.divider()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Average Risk", f"{scores.mean():.1f}")
                c2.metric("🟢 LOW", sum(batch_df["RISK_LEVEL"] == "LOW"))
                c3.metric("🟡 MEDIUM", sum(batch_df["RISK_LEVEL"] == "MEDIUM"))
                c4.metric("🔴 HIGH", sum(batch_df["RISK_LEVEL"] == "HIGH"))

                st.dataframe(batch_df, use_container_width=True)

                # Download results
                st.download_button(
                    "📥 Download Results",
                    batch_df.to_csv(index=False),
                    "predictions.csv",
                    "text/csv",
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ════════════════════════════════════════════════
# TAB 5: MODEL GOVERNANCE
# ════════════════════════════════════════════════
with tab5:
    st.header("🔍 Model Governance")

    if training_df is None:
        st.error("Training data not found. Run the pipeline first.")
        st.stop()

    gov_tab1, gov_tab2, gov_tab3, gov_tab4 = st.tabs([
        "⚖️ Bias Detection",
        "📈 Data Drift",
        "🔄 Model Drift",
        "🧠 Explainability",
    ])

    # ── Prepare data for governance ──
    X, y = prepare_features(training_df)
    y_pred_all = model.predict(X)

    # ───────────────────────────────────────────
    # GOV TAB 1: BIAS DETECTION
    # ───────────────────────────────────────────
    with gov_tab1:
        st.subheader("⚖️ Algorithmic Bias Detection")
        st.caption("Are predictions fair across demographic groups?")

        # Gender bias
        st.markdown("#### Bias by Gender")
        sensitive_gender = pd.DataFrame({
            "Gender": training_df["PRIM_DRIVER_GENDER"].values[:len(y_pred_all)]
        })
        gender_metrics = compute_bias_metrics(y, y_pred_all, sensitive_gender)

        gender_data = gender_metrics["Gender"]
        st.dataframe(
            gender_data["by_group"].round(3),
            use_container_width=True,
            hide_index=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "MAE Difference (M vs F)",
                f"{gender_data['difference']['MAE']:.3f}",
                help="Closer to 0 = fairer. Difference in Mean Absolute Error between groups.",
            )
        with col2:
            st.metric(
                "Mean Prediction Difference",
                f"{gender_data['difference']['Mean Prediction']:.3f}",
                help="Closer to 0 = fairer. Difference in average predicted risk between groups.",
            )

        # Gender prediction distribution
        fig, ax = plt.subplots(figsize=(10, 4))
        for gender in ["M", "F"]:
            mask = training_df["PRIM_DRIVER_GENDER"].values[:len(y_pred_all)] == gender
            ax.hist(y_pred_all[mask], bins=40, alpha=0.5, label=f"Gender: {gender}", edgecolor="white")
        ax.set_xlabel("Predicted Risk Score")
        ax.set_ylabel("Count")
        ax.set_title("Prediction Distribution by Gender")
        ax.legend()
        st.pyplot(fig)
        plt.close()

        # Age group bias
        st.markdown("#### Bias by Age Group")
        age_groups = create_age_groups(
            training_df["PRIM_DRIVER_AGE"].values[:len(y_pred_all)]
        )
        sensitive_age = pd.DataFrame({"Age Group": age_groups})
        age_metrics = compute_bias_metrics(y, y_pred_all, sensitive_age)

        age_data = age_metrics["Age Group"]
        st.dataframe(
            age_data["by_group"].round(3),
            use_container_width=True,
            hide_index=True,
        )

        fig, ax = plt.subplots(figsize=(10, 4))
        for group in ["16-25", "26-35", "36-50", "51-65", "65+"]:
            mask = age_groups == group
            if mask.sum() > 0:
                ax.hist(y_pred_all[mask], bins=40, alpha=0.4, label=group, edgecolor="white")
        ax.set_xlabel("Predicted Risk Score")
        ax.set_ylabel("Count")
        ax.set_title("Prediction Distribution by Age Group")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    # ───────────────────────────────────────────
    # GOV TAB 2: DATA DRIFT
    # ───────────────────────────────────────────
    with gov_tab2:
        st.subheader("📈 Data Drift Monitoring")
        st.caption("Has the input data distribution changed?")

        st.info(
            "Upload a new dataset to compare against training data. "
            "Or simulate drift by splitting training data into two halves."
        )

        drift_mode = st.radio(
            "Drift detection mode:",
            ["Simulate (split training data)", "Upload new data"],
            horizontal=True,
        )

        feature_cols = HOTSPOT_FEATURES + NUMERIC_FEATURES

        if drift_mode == "Simulate (split training data)":
            # Split training data in half to simulate drift
            mid = len(training_df) // 2
            ref_df = training_df.iloc[:mid]
            cur_df = training_df.iloc[mid:]
            st.write(f"Reference: first {len(ref_df):,} rows | Current: last {len(cur_df):,} rows")

            drift_df = compute_drift_metrics(ref_df, cur_df, feature_cols)
            if len(drift_df) > 0:
                # Color drift rows
                n_drifted = drift_df["Drift Detected"].sum()
                st.metric("Features with Drift", f"{n_drifted} / {len(drift_df)}")
                st.dataframe(
                    drift_df.style.apply(
                        lambda row: ["background-color: #FFEBEE"] * len(row)
                        if row["Drift Detected"] else [""] * len(row),
                        axis=1,
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                # Distribution comparison for drifted features
                drifted_features = drift_df[drift_df["Drift Detected"]]["Feature"].tolist()
                if drifted_features:
                    st.markdown("#### Drifted Feature Distributions")
                    for feat in drifted_features[:4]:  # Show top 4
                        fig, ax = plt.subplots(figsize=(10, 3))
                        ref_vals = pd.to_numeric(ref_df[feat], errors="coerce").dropna()
                        cur_vals = pd.to_numeric(cur_df[feat], errors="coerce").dropna()
                        ax.hist(ref_vals, bins=30, alpha=0.5, label="Reference", color="#2196F3", edgecolor="white")
                        ax.hist(cur_vals, bins=30, alpha=0.5, label="Current", color="#FF9800", edgecolor="white")
                        ax.set_title(f"{feat} Distribution")
                        ax.legend()
                        st.pyplot(fig)
                        plt.close()
            else:
                st.success("No drift detected in any features.")

        else:
            uploaded_drift = st.file_uploader("Upload CSV with new data", type=["csv"], key="drift_upload")
            if uploaded_drift:
                new_df = pd.read_csv(uploaded_drift)
                drift_df = compute_drift_metrics(training_df, new_df, feature_cols)
                if len(drift_df) > 0:
                    n_drifted = drift_df["Drift Detected"].sum()
                    st.metric("Features with Drift", f"{n_drifted} / {len(drift_df)}")
                    st.dataframe(drift_df, use_container_width=True, hide_index=True)

    # ───────────────────────────────────────────
    # GOV TAB 3: MODEL DRIFT
    # ───────────────────────────────────────────
    with gov_tab3:
        st.subheader("🔄 Model Drift Monitoring")
        st.caption("Are predictions shifting over time?")

        # Simulate by comparing predictions on two halves
        mid = len(X) // 2
        ref_preds = model.predict(X.iloc[:mid])
        cur_preds = model.predict(X.iloc[mid:])

        drift_info = compute_prediction_drift(ref_preds, cur_preds)

        col1, col2, col3 = st.columns(3)
        col1.metric("Reference Mean", f"{drift_info['ref_mean']:.2f}")
        col2.metric("Current Mean", f"{drift_info['cur_mean']:.2f}")
        col3.metric(
            "Mean Shift (std units)",
            f"{drift_info['mean_shift_std']:.3f}",
            delta="Drift!" if drift_info["drift_detected"] else "Stable",
            delta_color="inverse" if drift_info["drift_detected"] else "off",
        )

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.hist(ref_preds, bins=40, alpha=0.5, label="Reference (first half)", color="#2196F3", edgecolor="white")
        ax.hist(cur_preds, bins=40, alpha=0.5, label="Current (second half)", color="#FF9800", edgecolor="white")
        ax.set_xlabel("Predicted Risk Score")
        ax.set_ylabel("Count")
        ax.set_title("Prediction Distribution: Reference vs Current")
        ax.legend()
        st.pyplot(fig)
        plt.close()

        st.markdown("#### How to Interpret")
        st.write(
            "Mean Shift measures how far the current prediction average has moved "
            "from the reference, in units of reference standard deviation. "
            "A shift > 0.5 std suggests the model may be degrading and needs retraining."
        )

    # ───────────────────────────────────────────
    # GOV TAB 4: EXPLAINABILITY (SHAP)
    # ───────────────────────────────────────────
    with gov_tab4:
        st.subheader("🧠 Model Explainability (SHAP)")
        st.caption("Why did the model give this prediction?")

        # Use a sample for performance
        sample_size = min(500, len(X))
        X_sample = X.sample(n=sample_size, random_state=42)

        with st.spinner("Computing SHAP values (this may take a moment)..."):
            shap_values, expected_value, feature_names = compute_shap_values(model, X_sample)

        # Global feature importance (mean |SHAP|)
        st.markdown("#### Global Feature Importance")
        st.write("Which features matter most across all predictions?")

        mean_shap = np.abs(shap_values).mean(axis=0)
        shap_df = pd.DataFrame({
            "Feature": feature_names,
            "Mean |SHAP|": mean_shap,
        }).sort_values("Mean |SHAP|", ascending=True).tail(15)

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#2196F3" if "HOTSPOT" in f or "hotspot" in f else "#FF9800" for f in shap_df["Feature"]]
        ax.barh(shap_df["Feature"], shap_df["Mean |SHAP|"], color=colors)
        ax.set_xlabel("Mean |SHAP Value|")
        ax.set_title("Top 15 Features by SHAP Importance (Blue=Hotspot, Orange=Driver)")
        st.pyplot(fig)
        plt.close()

        # Single prediction explanation
        st.markdown("#### Individual Prediction Explanation")
        st.write("Select a row to see why the model made that prediction.")

        row_idx = st.slider("Select sample row", 0, sample_size - 1, 0)

        fig, ax = plt.subplots(figsize=(12, 5))
        sorted_idx = np.argsort(np.abs(shap_values[row_idx]))[::-1][:10]
        top_features = [feature_names[i] for i in sorted_idx]
        top_shap = [shap_values[row_idx][i] for i in sorted_idx]

        bar_colors = ["#4CAF50" if v > 0 else "#F44336" for v in top_shap]
        ax.barh(range(len(top_features)), top_shap, color=bar_colors)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features)
        ax.set_xlabel("SHAP Value (impact on prediction)")
        ax.set_title(f"Top 10 Feature Contributions for Row {row_idx}")
        ax.axvline(x=0, color="black", linewidth=0.5)
        ax.invert_yaxis()
        st.pyplot(fig)
        plt.close()

        prediction = model.predict(X_sample.iloc[[row_idx]])[0]
        level, emoji = classify_risk(prediction)
        st.write(f"**Predicted Risk Score:** {prediction:.1f} — {emoji} {level}")
        st.write(f"**Base Value (average prediction):** {expected_value:.1f}")
        st.write(
            "Green bars push the prediction higher (more risk). "
            "Red bars push it lower (less risk). "
            "The prediction equals the base value plus all SHAP contributions."
        )
