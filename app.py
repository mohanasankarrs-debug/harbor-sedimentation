
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from datetime import datetime

# Project paths
BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR
DATA_FILE = BASE_DIR / "chennai_port_simulated_sedimentation.csv"

st.set_page_config(
    page_title="Harbor Sedimentation Forecast",
    page_icon="🌊",
    layout="wide",
)


st.markdown("""
<style>
/* Dark ocean background */
.stApp {
    background: linear-gradient(135deg, #071827, #10384a);
    color: #e8f4f8;
}

/* Spacious dashboard */
.block-container {
    padding-top: 2rem;
    max-width: 1450px;
}

/* Professional prediction cards */
div[data-testid="stMetric"] {
    background: linear-gradient(145deg, #12354a, #0d293b);
    border: 1px solid #226078;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 6px 22px rgba(0, 0, 0, 0.18);
}

/* Teal prediction values */
div[data-testid="stMetricValue"] {
    color: #5de4d0;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #071522;
}

/* Rounded buttons */
.stButton > button {
    background: #19c6bd;
    color: #071827;
    border-radius: 10px;
    border: none;
    font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

st.title("🌊 AI-Based Harbor Sedimentation Prediction")
st.subheader("Machine Learning–Based Sedimentation Forecasting and Predictive Dredging Management")

# Load data and model
if not DATA_FILE.exists():
    st.error(f"Dataset not found: {DATA_FILE.name}")
    st.info("Check the CSV filename inside your project's data folder.")
    st.stop()

if not (OUT_DIR / "best_model.joblib").exists():
    st.error("Model not found. Run train.py first.")
    st.stop()

data = pd.read_csv(DATA_FILE)
data["date"] = pd.to_datetime(data["date"])

saved = joblib.load(OUT_DIR / "best_model.joblib")
model = saved["model"]
features = saved["features"]
target = saved["target"]

# Latest forecast
latest = data.copy()
latest["date"] = pd.to_datetime(latest["date"])
drivers = [
    col for col in latest.select_dtypes("number").columns
    if col != target
]

import numpy as np

latest["month_sin"] = np.sin(2 * np.pi * latest["date"].dt.month / 12)
latest["month_cos"] = np.cos(2 * np.pi * latest["date"].dt.month / 12)

for col in [target] + drivers:
    for lag in (1, 2, 3, 6, 12):
        latest[f"{col}_lag{lag}"] = latest[col].shift(lag)
    for window in (3, 6):
        latest[f"{col}_roll{window}"] = (
            latest[col].shift(1).rolling(window).mean()
        )

forecast_row = latest.iloc[[-1]][features]
forecast = float(model.predict(forecast_row)[0])

# Summary cards
c1, c2, c3 = st.columns(3)
c1.metric("Selected Model", type(model).__name__)
c2.metric("Latest Recorded Date", data["date"].max().strftime("%b %Y"))
c3.metric("Next-Step Forecast", f"{forecast:.3f}")

st.caption(
    f"Forecast target: {target}. Forecast horizon: "
    f"{saved['horizon']} time step(s). Units follow the simulated dataset."
)

# Dataset preview
st.header("📋 Simulated Dataset")
st.dataframe(data.tail(12), use_container_width=True)

# Forecast history graph
st.header("📈 Sedimentation History")
st.line_chart(data.set_index("date")[target])

# Model comparison
comparison_file = OUT_DIR / "model_comparison.csv"
if comparison_file.exists():
    st.header("🤖 Model Performance Comparison")
    comparison = pd.read_csv(comparison_file, index_col=0)
    st.dataframe(comparison.round(4), use_container_width=True)

# Existing evaluation plots
st.header("📊 Model Evaluation")

plot1, plot2 = st.columns(2)

with plot1:
    image_path = OUT_DIR / "pred_vs_obs.png"
    if image_path.exists():
        st.image(str(image_path), caption="Observed vs predicted on test data")

with plot2:
    image_path = OUT_DIR / "scatter.png"
    if image_path.exists():
        st.image(str(image_path), caption="Prediction scatter plot")

importance_path = OUT_DIR / "feature_importance.png"
if importance_path.exists():
    st.image(str(importance_path), caption="Feature importance")

# Dredging Process
st.divider()
st.header("🚢 Dredging Process")

st.write(
    "Enter your survey and operational data below to assess "
    "sedimentation and plan a dredging operation."
)

st.subheader("1. Enter Harbor Survey Data")

col1, col2 = st.columns(2)

with col1:
    area_m2 = st.number_input(
        "Area requiring dredging (m²)",
        min_value=0.0,
        value=10000.0,
        step=500.0
    )

    initial_depth = st.number_input(
        "Current water depth (m)",
        min_value=0.0,
        value=8.0,
        step=0.1
    )

    required_depth = st.number_input(
        "Required water depth (m)",
        min_value=0.0,
        value=10.0,
        step=0.1
    )

with col2:
    sediment_thickness = st.number_input(
        "Measured sediment thickness (m)",
        min_value=0.0,
        value=0.5,
        step=0.1
    )

    dredger_capacity = st.number_input(
        "Dredger capacity (m³/hour)",
        min_value=1.0,
        value=200.0,
        step=10.0
    )

    cost_per_m3 = st.number_input(
        "Dredging cost per m³ (₹)",
        min_value=0.0,
        value=500.0,
        step=50.0
    )

st.subheader("2. Dredging Assessment")

if st.button("Calculate Dredging Plan", type="primary"):

    if area_m2 <= 0:
        st.error("Area must be greater than zero.")

    elif initial_depth >= required_depth and sediment_thickness == 0:
        st.info(
            "The entered depth meets the target and no sediment "
            "thickness was entered. No dredging volume is estimated."
        )

    else:
        # Estimate volume from the user-entered sediment thickness.
        volume_from_survey = area_m2 * sediment_thickness

        # Estimate additional volume needed to reach target depth.
        depth_deficit = max(0.0, required_depth - initial_depth)
        volume_from_depth = area_m2 * depth_deficit

        # Use the larger estimate to avoid understating the plan.
        estimated_volume = max(
            volume_from_survey,
            volume_from_depth
        )

        estimated_hours = estimated_volume / dredger_capacity
        estimated_cost = estimated_volume * cost_per_m3
        
        st.session_state["last_dredging_plan"] = {
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "area_m2": area_m2,
            "current_depth_m": initial_depth,
            "required_depth_m": required_depth,
            "sediment_thickness_m": sediment_thickness,
            "estimated_volume_m3": estimated_volume,
            "dredger_capacity_m3_hour": dredger_capacity,
            "estimated_hours": estimated_hours,
            "cost_per_m3": cost_per_m3,
            "estimated_cost": estimated_cost,
            }

        st.subheader("3. Results")

        m1, m2, m3 = st.columns(3)

        m1.metric(
            "Estimated dredging volume",
            f"{estimated_volume:,.0f} m³"
        )

        m2.metric(
            "Estimated operating time",
            f"{estimated_hours:,.1f} hours"
        )

        m3.metric(
            "Estimated cost",
            f"₹{estimated_cost:,.0f}"
        )

        if depth_deficit > 0 or sediment_thickness > 0:
            st.warning(
                "Assessment: dredging may be required based on "
                "the values you entered. Verify with a qualified "
                "hydrographic survey before operational use."
            )
        else:
            st.success(
                "Assessment: no dredging volume is indicated "
                "by the entered values."
            )

        st.subheader("4. Dredging Work Summary")

        summary = {
            "Parameter": [
                "Area",
                "Current depth",
                "Required depth",
                "Depth deficit",
                "Measured sediment thickness",
                "Estimated dredging volume",
                "Dredger capacity",
                "Estimated operating time",
                "Cost per cubic metre",
                "Estimated total cost"
            ],
            "Value": [
                f"{area_m2:,.2f} m²",
                f"{initial_depth:.2f} m",
                f"{required_depth:.2f} m",
                f"{depth_deficit:.2f} m",
                f"{sediment_thickness:.2f} m",
                f"{estimated_volume:,.2f} m³",
                f"{dredger_capacity:,.2f} m³/hour",
                f"{estimated_hours:,.2f} hours",
                f"₹{cost_per_m3:,.2f}",
                f"₹{estimated_cost:,.2f}"
            ]
        }

        st.table(summary)

        import pandas as pd

        report_df = pd.DataFrame(summary)

        st.download_button(
            "Download Dredging Report (CSV)",
            data=report_df.to_csv(index=False).encode("utf-8"),
            file_name="dredging_plan.csv",
            mime="text/csv"
        )



# ==========================================
# DREDGING HISTORY - SAVE AND VIEW RECORDS
# ==========================================

from pathlib import Path
from datetime import datetime
import pandas as pd

HISTORY_FILE = Path(__file__).resolve().parent / "dredging_history.csv"

st.divider()
st.header("📚 Dredging History")

st.write(
    "Save a completed dredging plan to keep a history "
    "of your operations."
)

if "last_dredging_plan" not in st.session_state:
    st.session_state["last_dredging_plan"] = None

if st.button("Save Current Dredging Plan"):
    # Read the current calculator values from the page.
    # These variables are created in the calculator section.
    if "estimated_volume" in locals():
        st.session_state["last_dredging_plan"] = {
            "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "area_m2": area_m2,
            "current_depth_m": initial_depth,
            "required_depth_m": required_depth,
            "sediment_thickness_m": sediment_thickness,
            "estimated_volume_m3": estimated_volume,
            "dredger_capacity_m3_hour": dredger_capacity,
            "estimated_hours": estimated_hours,
            "cost_per_m3": cost_per_m3,
            "estimated_cost": estimated_cost,
        }
        st.success("Plan ready to save.")
    else:
        st.warning(
            "Please click Calculate Dredging Plan first, "
            "then save the plan."
        )

if st.button("Confirm and Store Dredging Record"):
    record = st.session_state.get("last_dredging_plan")

    if record is None:
        st.warning("Calculate and prepare a plan first.")
    else:
        new_record = pd.DataFrame([record])

        if HISTORY_FILE.exists():
            old_records = pd.read_csv(HISTORY_FILE)
            all_records = pd.concat(
                [old_records, new_record],
                ignore_index=True
            )
        else:
            all_records = new_record

        all_records.to_csv(HISTORY_FILE, index=False)
        st.session_state["last_dredging_plan"] = None
        st.success("Dredging record saved successfully!")

if HISTORY_FILE.exists():
    history_df = pd.read_csv(HISTORY_FILE)

    st.subheader("Saved Operations")
    st.dataframe(history_df, use_container_width=True)

    total_volume = history_df["estimated_volume_m3"].sum()
    total_cost = history_df["estimated_cost"].sum()

    col1, col2 = st.columns(2)
    col1.metric("Cumulative Volume (m³)", f"{total_volume:,.2f}")
    col2.metric("Cumulative Estimated Cost", f"₹{total_cost:,.2f}")

    st.download_button(
        "Download Dredging History CSV",
        data=history_df.to_csv(index=False).encode("utf-8"),
        file_name="dredging_history.csv",
        mime="text/csv"
    )
else:
    st.info("No saved dredging records yet.")
