import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# Page Configuration
st.set_page_config(page_title="Triage Support System (KTAS)", layout="centered")

# Load external CSS
css_path = Path(__file__).parent / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# Page header
st.title("Triage Decision Support – KTAS")
st.write("Enter patient data to obtain the suggested KTAS category and the model decision explanation.")


# Model Loading (cached to prevent reloading on every click)
@st.cache_resource
def load_model():
    return joblib.load("final_model.pkl")


try:
    final_model = load_model()
    xgb_model = final_model.named_estimators_['xgb']
    explainer = shap.TreeExplainer(xgb_model)
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# User Interface
st.header("Patient Data")

# Dynamic pain assessment
st.subheader("Pain Assessment")
has_pain = st.radio(
    "Does the patient report pain?",
    options=["Yes", "No (or unresponsive)"],
    horizontal=True
)

if has_pain == "Yes":
    nrs_pain = st.slider("NRS Pain Scale", min_value=1, max_value=10, value=5)
    pain_raw = 1
else:
    st.slider("NRS Pain Scale (Disabled – No Pain)", min_value=0, max_value=1, value=0, disabled=True)
    nrs_pain = 0
    pain_raw = 0


# Dynamic oxygen saturation assessment
st.subheader("Oxygen Saturation")
sat_missing = st.checkbox("Oxygen saturation measurement missing")

if sat_missing:
    st.number_input("Oxygen Saturation (SpO₂ %)", min_value=0, max_value=100, value=0, disabled=True,
                    help="Measurement marked as missing.")
    saturation = np.nan
else:
    saturation = st.number_input("Oxygen Saturation (SpO₂ %)", min_value=50, max_value=100, value=98)


# Main input form
with st.form("triage_form"):
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Demographics & History")
        sex_raw = st.selectbox("Sex", options=["Female", "Male"])
        age = st.number_input("Age", min_value=0, max_value=150, value=35)
        injury_raw = st.selectbox("Injury", options=["No", "Yes"])
        mental = st.selectbox("Mental Status",
                              options=[1, 2, 3, 4],
                              format_func=lambda x:
                              {1: "1 – Alert", 2: "2 – Verbal Response",
                               3: "3 – Pain Response", 4: "4 – Unresponsive"}[x])

    with col2:
        st.subheader("Vital Signs")
        sbp = st.number_input("Systolic Blood Pressure (SBP)", min_value=40, max_value=300, value=120)
        dbp = st.number_input("Diastolic Blood Pressure (DBP)", min_value=40, max_value=200, value=80)
        hr = st.number_input("Heart Rate (HR)", min_value=40, max_value=150, value=75)
        rr = st.number_input("Respiratory Rate (RR)", min_value=10, max_value=40, value=16)
        bt = st.number_input("Body Temperature (BT °C)", min_value=30.0, max_value=50.0, value=36.6, step=0.1)

    st.markdown("---")
    col4_only, = st.columns(1)

    with col4_only:
        st.subheader("Chief Complaint & Arrival")
        col_c, col_a = st.columns(2)
        with col_c:
            complaint_options = {
                "C_Open Wound": "Open Wound", "C_Other": "Other", "C_abd pain": "Abdominal pain",
                "C_ant. chest pain": "Anterior chest pain", "C_dizziness": "Dizziness", "C_dyspnea": "Dyspnea",
                "C_epigastric pain": "Epigastric pain", "C_fever": "Fever", "C_general weakness": "General weakness",
                "C_headache": "Headache", "C_mental change": "Mental change"
            }
            selected_complaint = st.selectbox("Select chief complaint", options=list(complaint_options.keys()),
                                              format_func=lambda x: complaint_options[x])
        with col_a:
            arrival_options = {
                "Arrival_1": "Walking", "Arrival_2": "Public Ambulance", "Arrival_3": "Private Vehicle",
                "Arrival_4": "Private Ambulance", "Arrival_5": "Other"
            }
            selected_arrival = st.selectbox("Mode of transportation", options=list(arrival_options.keys()),
                                            format_func=lambda x: arrival_options[x])

        patients_per_hour = st.number_input("Patients arriving at ED per hour", min_value=0, max_value=50, value=5)

    submitted = st.form_submit_button("Run Triage")


# Data processing
if submitted:
    sex = 1 if sex_raw == "Female" else 2
    injury = 2 if injury_raw == "Yes" else 1

    data = {
        'Sex': sex, 'Age': age, 'Patients number per hour': patients_per_hour,
        'Injury': injury, 'Mental': mental, 'NRS_pain': nrs_pain,
        'SBP': sbp, 'DBP': dbp, 'HR': hr, 'RR': rr, 'BT': bt, 'Saturation': saturation
    }

    for c in complaint_options.keys():
        data[c] = int(c == selected_complaint)

    data["Saturation_is_missing"] = 1 if sat_missing else 0
    data["pain_unmeasurable"] = 1 if (mental >= 3 and pain_raw == 0) else 0

    for arr in ["Arrival_2", "Arrival_3", "Arrival_4", "Arrival_5"]:
        data[arr] = int(selected_arrival == arr)

    data["is_hypotensive"] = int((sbp < 90) or (dbp < 60))
    data["is_hypertensive"] = int((sbp > 140) or (dbp > 90))
    data["has_fever"] = int(bt > 38.0)
    data["low_oxygen"] = int(saturation < 94) if not sat_missing else 0
    data["tachycardia"] = int(hr > 100)
    data["bradycardia"] = int(hr < 60)
    data["shock_index"] = hr / sbp if sbp > 0 else 0
    data["map"] = (sbp + 2 * dbp) / 3

    feature_order = [
        'Sex', 'Age', 'Patients number per hour', 'Injury', 'Mental', 'NRS_pain',
        'SBP', 'DBP', 'HR', 'RR', 'BT', 'Saturation',
        'C_Open Wound', 'C_Other', 'C_abd pain', 'C_ant. chest pain', 'C_dizziness',
        'C_dyspnea', 'C_epigastric pain', 'C_fever', 'C_general weakness', 'C_headache', 'C_mental change',
        'Saturation_is_missing', 'pain_unmeasurable', 'Arrival_2', 'Arrival_3', 'Arrival_4', 'Arrival_5',
        'is_hypotensive', 'is_hypertensive', 'has_fever', 'low_oxygen', 'tachycardia', 'bradycardia',
        'shock_index', 'map'
    ]

    patient_df = pd.DataFrame([data])[feature_order]
    patient_df = patient_df.astype(float)

    # Prediction
    try:
        probabilities = final_model.predict_proba(patient_df)[0]
        predicted_class = final_model.predict(patient_df)[0]
        class_idx = int(predicted_class)

        model_uses_zero_indexing = (min(final_model.classes_) == 0)
        display_class = class_idx + 1 if model_uses_zero_indexing else class_idx

        st.markdown("---")
        st.header("Triage Classification Result")

        color_map = {1: "#dc2626", 2: "#f97316", 3: "#ca8a04", 4: "#16a34a", 5: "#2563eb"}
        color = color_map.get(display_class, "#111827")

        st.markdown(
            f"### Suggested Category: "
            f"<span style='color:{color}; font-size:32px; font-weight:600; "
            f"font-family:\"IBM Plex Mono\",monospace;'>KTAS {display_class}</span>",
            unsafe_allow_html=True)

        # Probability chart
        st.subheader("KTAS Class Probability Distribution")

        classes_labels, colors_list = [], []
        for c in final_model.classes_:
            actual_ktas = int(c) + 1 if model_uses_zero_indexing else int(c)
            classes_labels.append(f"KTAS {actual_ktas}")
            colors_list.append(color_map.get(actual_ktas, "#9ca3af"))

        fig_prob, ax_prob = plt.subplots(figsize=(9, 2.8))
        fig_prob.patch.set_facecolor('#f7f8fa')
        ax_prob.set_facecolor('#f7f8fa')

        bars = ax_prob.barh(classes_labels, probabilities * 100,
                            color=colors_list, alpha=.85, height=.55)

        for bar in bars:
            width = bar.get_width()
            ax_prob.text(width + .8, bar.get_y() + bar.get_height() / 2,
                         f'{width:.1f}%',
                         va='center', ha='left', fontsize=9,
                         fontweight='500', color='#4b5563',
                         fontfamily='monospace')

        ax_prob.set_xlim(0, 115)
        ax_prob.set_xlabel("Probability (%)", fontsize=9, color='#9ca3af')
        ax_prob.invert_yaxis()
        ax_prob.tick_params(colors='#9ca3af', labelsize=9)
        for spine in ax_prob.spines.values():
            spine.set_color('#e2e6ea')
        plt.tight_layout()
        st.pyplot(fig_prob)

        # SHAP
        st.markdown("---")
        st.subheader(f"Why KTAS {display_class}? — SHAP Analysis")

        with st.spinner("Generating explanation…"):
            shap_values_patient = explainer(patient_df)

            fig_shap, _ = plt.subplots(figsize=(9, 5))
            fig_shap.patch.set_facecolor('#f7f8fa')
            shap.plots.waterfall(shap_values_patient[0, :, class_idx], show=False)
            plt.tight_layout()
            st.pyplot(fig_shap)

            with st.expander("ℹ️ How to interpret the SHAP plot — Clinician's Guide"):
                st.markdown("""
The **SHAP waterfall** plot shows how each patient parameter shifted the model's score
away from the population average toward (or away from) this KTAS category.

| Element | Meaning |
|---|---|
| **Base value** *E[f(X)]* | Average model score across the training dataset |
| **Final value** *f(x)* | Score for this specific patient |
| 🔴 **Red bars** | Feature *increased* urgency score |
| 🔵 **Blue bars** | Feature *decreased* urgency score |

Features are sorted top-to-bottom by their impact magnitude for this patient.
""")

    except Exception as e:
        st.error(f"An error occurred during prediction or chart generation: {e}")

st.markdown("---")
st.caption(
    "⚠️ **Disclaimer:** This system is for clinical decision support only. "
    "The final triage category remains the sole responsibility of the attending clinician.")