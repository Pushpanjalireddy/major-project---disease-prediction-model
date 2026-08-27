"""
Clinical Disease Prediction System & XGBoost Machine Learning Assistant.
Features:
- Dual Voice & Text Input with Web Speech Recognition & Audio Synthesis
- Focused on the Best-in-Class XGBoost Classifier (99.86% Accuracy)
- Complete Mathematical Derivations & Formulas for XGBoost
- Clinical Triage Engine with Urgency Assessment & Red Flag Detection
- Actionable Home Remedies, Ayurvedic Guidance, Dietary Advice, & Specialist Recommendations
- Interactive Disease Encyclopedia & Feature Importance Explorer
"""

import json
import os
import re
import io
import tempfile
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import matplotlib.pyplot as plt
import seaborn as sns
import speech_recognition as sr
from streamlit_mic_recorder import mic_recorder, speech_to_text
from gtts import gTTS

from clinical_knowledge import (
    DISEASE_NAME_MAP,
    DISEASE_KNOWLEDGE,
    RED_FLAGS,
    SYMPTOM_SYNONYMS,
    parse_symptoms,
    assess_urgency,
    assess_urgency_comprehensive,
    get_remedies_for_disease,
    predict_clinical_hybrid,
    predict_clinical_comprehensive,
    get_comorbidity_tailored_precautions
)
from home_remedies_guide import (
    REMEDY_RECIPES,
    CONDITION_GUIDES,
    CATEGORIES_LIST
)

BASE_DIR = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Clinical Disease AI Assistant (XGBoost)",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Healthcare Dark Mode Styling
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        
        * {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        .stApp {
            background: linear-gradient(135deg, #090d16 0%, #0f172a 50%, #1e1b4b 100%);
            color: #f8fafc;
        }

        /* Sidebar */
        div[data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.96);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        div[data-testid="stSidebar"] label,
        div[data-testid="stSidebar"] p,
        div[data-testid="stSidebar"] span {
            color: #f8fafc !important;
            font-weight: 600 !important;
        }

        /* Glass Cards */
        .glass-card {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 1.25rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
        }

        .hero-banner {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(168, 85, 247, 0.25) 100%);
            border: 1px solid rgba(168, 85, 247, 0.35);
            border-radius: 20px;
            padding: 1.8rem;
            margin-bottom: 1.5rem;
        }

        .urgency-badge {
            display: inline-block;
            padding: 6px 16px;
            border-radius: 20px;
            font-weight: 700;
            font-size: 0.9rem;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        .badge-emergency {
            background: rgba(239, 68, 68, 0.2);
            color: #f87171;
            border: 1px solid #ef4444;
        }

        .badge-doctor-soon {
            background: rgba(245, 158, 11, 0.2);
            color: #fbbf24;
            border: 1px solid #f59e0b;
        }

        .badge-monitor {
            background: rgba(59, 130, 246, 0.2);
            color: #60a5fa;
            border: 1px solid #3b82f6;
        }

        .badge-self-care {
            background: rgba(16, 185, 129, 0.2);
            color: #34d399;
            border: 1px solid #10b981;
        }

        .symptom-tag {
            display: inline-block;
            background: rgba(99, 102, 241, 0.2);
            color: #c7d2fe;
            border: 1px solid rgba(99, 102, 241, 0.4);
            border-radius: 20px;
            padding: 4px 12px;
            margin: 3px;
            font-size: 0.85rem;
            font-weight: 500;
        }

        .formula-box {
            background: rgba(15, 23, 42, 0.85);
            border-left: 4px solid #38bdf8;
            padding: 1rem 1.25rem;
            border-radius: 8px;
            margin: 0.75rem 0;
            color: #e2e8f0;
        }

        /* High-Contrast Headings & Typography */
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff !important;
            font-weight: 700 !important;
        }

        p, span, label, div {
            color: #f1f5f9;
        }

        /* Streamlit Radio Buttons Custom Styling (Ultra-Visible & High Contrast) */
        div[data-testid="stRadio"] {
            background: rgba(15, 23, 42, 0.92) !important;
            border: 1.5px solid rgba(99, 102, 241, 0.45) !important;
            border-radius: 16px !important;
            padding: 18px 22px !important;
            margin: 12px 0 20px 0 !important;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4) !important;
        }

        div[data-testid="stRadio"] > label {
            color: #38bdf8 !important;
            font-size: 1.1rem !important;
            font-weight: 800 !important;
            margin-bottom: 12px !important;
            display: block !important;
            letter-spacing: 0.3px !important;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] {
            gap: 10px !important;
            display: flex !important;
            flex-direction: column !important;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] label {
            background: rgba(30, 41, 59, 0.9) !important;
            border: 1.5px solid rgba(255, 255, 255, 0.22) !important;
            border-radius: 12px !important;
            padding: 12px 18px !important;
            transition: all 0.2s ease-in-out !important;
            cursor: pointer !important;
            display: flex !important;
            align-items: center !important;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
            background: rgba(56, 189, 248, 0.2) !important;
            border-color: #38bdf8 !important;
            transform: translateX(4px);
        }

        div[data-testid="stRadio"] div[role="radiogroup"] label p {
            color: #ffffff !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            opacity: 1 !important;
            margin: 0 !important;
            line-height: 1.4 !important;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] label span {
            color: #ffffff !important;
            font-weight: 700 !important;
        }

        div[data-testid="stMarkdownContainer"] h3 {
            color: #ffffff !important;
            font-weight: 800 !important;
        }

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 12px;
            background: rgba(15, 23, 42, 0.6);
            padding: 8px;
            border-radius: 14px;
            border: 1px solid rgba(255, 255, 255, 0.05);
        }

        .stTabs [data-baseweb="tab"] {
            border-radius: 10px;
            padding: 8px 18px;
            color: #94a3b8;
            font-weight: 600;
        }

        .stTabs [aria-selected="true"] {
            background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
            color: white !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_assets():
    """Loads XGBoost model, label encoder, feature set, dataset, and metrics."""
    xgb_path = BASE_DIR / "disease_xgb_model.joblib"
    label_path = BASE_DIR / "disease_label_encoder.joblib"
    feature_path = BASE_DIR / "symptom_feature_cols.joblib"
    dataset_path = BASE_DIR / "day_to_day_clinical_disease_dataset.csv"
    metrics_path = BASE_DIR / "xgboost_performance_metrics.json"

    if not all([xgb_path.exists(), label_path.exists(), feature_path.exists(), dataset_path.exists()]):
        raise FileNotFoundError(
            "Required assets not found. Please run 'python train_model.py' to train and export the XGBoost model."
        )

    model = joblib.load(xgb_path)
    label_encoder = joblib.load(label_path)
    feature_cols = joblib.load(feature_path)
    dataset = pd.read_csv(dataset_path)

    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    return model, label_encoder, feature_cols, dataset, metrics


try:
    xgb_model, label_encoder, feature_cols, dataset, metrics_data = load_assets()
except Exception as e:
    st.error(f"Error loading system assets: {e}")
    st.stop()


def render_voice_input_widget():
    """
    Renders an interactive Voice Input widget with real-time Speech Recognition
    powered by streamlit_mic_recorder and Google Speech Recognition.
    Captures spoken audio directly into Streamlit, transcribes it, updates the text box,
    and supports instant disease prediction.
    """
    st.markdown(
        """
        <div style="background:rgba(15,23,42,0.85);border:1px solid rgba(99,102,241,0.4);
                    border-radius:14px;padding:14px 18px;margin:4px 0 12px 0;
                    box-shadow:0 8px 24px rgba(0,0,0,0.3);">
            <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
                <div style="font-weight:700;font-size:1rem;color:#38bdf8;display:flex;align-items:center;gap:8px;">
                    <span>🎙️ Live Voice Input & Speech-to-Diagnosis</span>
                    <span style="font-size:0.75rem;background:rgba(56,189,248,0.18);color:#7dd3fc;padding:2px 10px;border-radius:12px;border:1px solid rgba(56,189,248,0.35);">
                        Google Speech Engine
                    </span>
                </div>
                <span style="font-size:0.8rem;color:#94a3b8;">Click mic below to speak symptoms clearly</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    v_c1, v_c2 = st.columns([1.3, 1.1])
    with v_c1:
        lang_choice = st.selectbox(
            "🌐 Spoken Language:",
            ["English (US / Global)", "English (India)", "Hindi (हिन्दी)"],
            index=0,
            key="voice_lang_choice",
            help="Select the language you will speak in for optimal transcription accuracy."
        )
        lang_map = {
            "English (US / Global)": "en-US",
            "English (India)": "en-IN",
            "Hindi (हिन्दी)": "hi-IN"
        }
        selected_lang = lang_map.get(lang_choice, "en-US")

    with v_c2:
        auto_predict = st.checkbox(
            "⚡ Auto-predict disease on speech",
            value=True,
            help="When checked, disease prediction will run automatically as soon as you finish speaking.",
            key="voice_auto_predict_toggle"
        )

    # Microphone recording component from streamlit_mic_recorder
    spoken_text = speech_to_text(
        start_prompt="🎙️ Click to Speak Symptoms",
        stop_prompt="⏹️ Stop Speaking & Process Diagnosis",
        language=selected_lang,
        use_container_width=True,
        just_once=True,
        key="symptom_voice_speech_recorder"
    )

    # Handle transcribed speech from microphone
    if spoken_text and spoken_text.strip():
        cleaned_speech = spoken_text.strip()
        last_voice = st.session_state.get("last_processed_voice_text")
        if last_voice != cleaned_speech:
            st.session_state["last_processed_voice_text"] = cleaned_speech
            st.session_state["patient_symptoms_text_box"] = cleaned_speech
            st.session_state["input_text"] = cleaned_speech
            st.session_state["voice_banner_msg"] = cleaned_speech
            if auto_predict:
                st.session_state["trigger_predict_now"] = True
                st.session_state["active_view"] = "report"
            st.rerun()

    # Optional collapsible file uploader for audio files (WAV, MP3, etc.)
    with st.expander("📁 Upload Voice Note / Audio File (.wav, .mp3, .m4a)", expanded=False):
        uploaded_audio = st.file_uploader(
            "Upload audio recording of symptoms:",
            type=["wav", "mp3", "m4a", "ogg", "flac"],
            key="voice_audio_file_uploader"
        )
        if uploaded_audio is not None:
            if st.button("🎧 Transcribe & Predict Audio", key="btn_transcribe_audio_file"):
                with st.spinner("Processing audio recording..."):
                    try:
                        r = sr.Recognizer()
                        with sr.AudioFile(uploaded_audio) as source:
                            audio_data = r.record(source)
                            transcribed = r.recognize_google(audio_data, language=selected_lang)
                            if transcribed:
                                st.session_state["patient_symptoms_text_box"] = transcribed
                                st.session_state["input_text"] = transcribed
                                st.session_state["voice_banner_msg"] = transcribed
                                if auto_predict:
                                    st.session_state["trigger_predict_now"] = True
                                    st.session_state["active_view"] = "report"
                                st.success(f"✅ Transcribed: \"{transcribed}\"")
                                st.rerun()
                    except Exception as ex:
                        st.error(f"⚠️ Audio transcription error: {ex}. Please ensure the file is a valid clear audio recording.")


def render_tts_button(text_to_speak: str):
    """Text-to-Speech audio reader button using instant browser SpeechSynthesis."""
    clean_text = text_to_speak.replace('"', '\\"').replace("\n", " ")
    tts_html = f"""
    <div style="margin:8px 0;">
      <button id="tts-btn" style="background:rgba(56,189,248,0.15);color:#38bdf8;border:1px solid #38bdf8;padding:8px 18px;border-radius:24px;cursor:pointer;font-weight:600;font-size:13px;display:inline-flex;align-items:center;gap:8px;transition:all 0.2s;">
        <span style="font-size:15px;">🔊</span> Listen to Diagnosis & Care Plan (Instant Audio Readout)
      </button>
      <script>
        document.getElementById('tts-btn').addEventListener('click', () => {{
          if ('speechSynthesis' in window) {{
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance("{clean_text}");
            utterance.rate = 0.95;
            utterance.pitch = 1.0;
            utterance.lang = 'en-US';
            window.speechSynthesis.speak(utterance);
          }} else {{
            alert('Speech synthesis not supported in this browser.');
          }}
        }});
      </script>
    </div>
    """
    components.html(tts_html, height=45)


# ---------------------------------------------------------
# Sidebar: Patient Details & Triage Inputs
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🧑‍⚕️ Patient Information")
    age = st.number_input("Patient Age", min_value=1, max_value=120, value=28, step=1, key="sidebar_patient_age")
    gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"], key="sidebar_patient_gender")
    days = st.slider("Duration of Symptoms (Days)", min_value=1, max_value=60, value=3, key="sidebar_patient_days")
    severity = st.slider("Discomfort Severity (1-10)", min_value=1, max_value=10, value=5, key="sidebar_patient_severity")
    existing_cond = st.multiselect(
        "Pre-existing Conditions",
        ["None", "Diabetes", "Hypertension", "Asthma", "Heart Disease", "Kidney Disease", "Thyroid Disorder"],
        default=["None"],
        key="sidebar_patient_conditions"
    )

    st.markdown("---")
    st.markdown("### 🤖 Model Architecture")
    st.markdown(
        """
        <div style="background:rgba(99,102,241,0.15);border:1px solid rgba(99,102,241,0.3);border-radius:10px;padding:12px;">
            <div style="font-weight:700;color:#38bdf8;font-size:0.95rem;">⚡ XGBoost Classifier</div>
            <div style="font-size:0.8rem;color:#cbd5e1;margin-top:4px;">
                Extreme Gradient Boosting with Softmax Multi-Class Probability
            </div>
            <div style="margin-top:8px;font-size:0.8rem;color:#34d399;font-weight:600;">
                ✓ Test Accuracy: 99.86%<br>
                ✓ 5-Fold CV: 99.78% ± 0.11%
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("---")
    st.caption("🔒 Educational Clinical Decision Support • Fast & Scalable")


# ---------------------------------------------------------
# Main Page Hero Header
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <div style="display:flex;align-items:center;gap:15px;">
            <span style="font-size:2.4rem;">🩺</span>
            <div>
                <h1 style="margin:0;font-size:2rem;font-weight:800;background:linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                    Clinical Disease AI Prediction System
                </h1>
                <p style="margin:4px 0 0 0;color:#94a3b8;font-size:1rem;font-weight:500;">
                    Voice & Text Input • Powered by XGBoost (99.86% Accuracy) • Comprehensive Home Remedies & Triage
                </p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🩺 AI Clinical Diagnosis & Bot",
    "🧮 XGBoost Mathematical Engine & Model Performance",
    "📖 Disease & Remedy Encyclopedia",
    "⚠️ Emergency Red Flags",
    "🧪 Dataset & Feature Explorer"
])


# =========================================================
# TAB 1: AI Clinical Diagnosis & Bot
# =========================================================
with tab1:
    # Initialize view mode in session state
    if "active_view" not in st.session_state:
        st.session_state["active_view"] = "input"

    # 1. Ingest query parameter from voice input safely and synchronize session state
    if "voice_query" in st.query_params:
        v_param = st.query_params.get("voice_query", "").strip()
        v_auto = st.query_params.get("voice_predict", "") == "1"
        if v_param:
            st.session_state["patient_symptoms_text_box"] = v_param
            st.session_state["input_text"] = v_param
            st.session_state["voice_banner_msg"] = v_param
            if v_auto:
                st.session_state["trigger_predict_now"] = True
                st.session_state["active_view"] = "report"
        st.query_params.clear()

    # ---------------------------------------------------------
    # ROUTE 1: DEDICATED FULL-PAGE DIAGNOSTIC REPORT VIEW
    # ---------------------------------------------------------
    if st.session_state.get("active_view") == "report" and "last_prediction_data" in st.session_state:
        p_data = st.session_state["last_prediction_data"]
        top_disease = p_data["top_disease"]
        top_confidence = p_data["top_confidence"]
        urgency_level = p_data["urgency_level"]
        action_advice = p_data["action_advice"]
        triage_reasons = p_data["triage_reasons"]
        vuln_score = p_data["vuln_score"]
        vuln_tier = p_data["vuln_tier"]
        clinical_reasons = p_data.get("clinical_reasons", [])
        comorbidity_precautions = p_data.get("comorbidity_precautions", [])
        remedy_data = p_data["remedy_data"]
        predictions = p_data["predictions"]
        matched_keys = p_data["matched_keys"]
        snap = p_data.get("patient_snapshot", {"age": 28, "gender": "Male", "days": 3, "severity": 5, "conditions": ["None"]})

        # TOP NAVIGATION BAR
        nav_col1, nav_col2 = st.columns([1.6, 2.4])
        with nav_col1:
            if st.button("⬅️ Start New Assessment / Edit Symptoms", type="primary", use_container_width=True, key="btn_nav_back_to_input"):
                st.session_state["active_view"] = "input"
                st.rerun()
        with nav_col2:
            st.markdown(
                """
                <div style="display:flex;align-items:center;justify-content:flex-end;height:100%;gap:10px;padding-top:4px;">
                    <span style="background:rgba(16,185,129,0.15);color:#34d399;border:1px solid rgba(16,185,129,0.3);padding:6px 16px;border-radius:20px;font-size:0.85rem;font-weight:600;">
                        ✓ AI Diagnosis Verified (99.86% XGBoost Engine)
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.1);margin:16px 0 20px 0;'>", unsafe_allow_html=True)

        # 1. EVALUATED PATIENT PROFILE CONTEXT CARD
        active_conds_str = ", ".join([c for c in snap.get("conditions", []) if c != "None"]) or "None (Healthy Baseline)"
        duration_days = snap.get("days", 3)
        if duration_days <= 3:
            phase_str = f"Acute Phase ({duration_days} Days)"
        elif duration_days <= 14:
            phase_str = f"Subacute Phase ({duration_days} Days)"
        else:
            phase_str = f"Chronic / Persistent ({duration_days} Days)"

        sev_num = snap.get("severity", 5)
        vuln_color = "#ef4444" if vuln_score >= 75 else ("#f59e0b" if vuln_score >= 50 else ("#38bdf8" if vuln_score >= 30 else "#34d399"))

        st.markdown(
            f"""
            <div class="glass-card" style="border:1px solid rgba(56,189,248,0.3);background:rgba(15,23,42,0.85);margin-bottom:18px;">
                <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;margin-bottom:12px;border-bottom:1px solid rgba(255,255,255,0.08);padding-bottom:8px;">
                    <div style="font-weight:700;font-size:1.05rem;color:#f8fafc;display:flex;align-items:center;gap:8px;">
                        <span>🧑‍⚕️ Evaluated Patient Clinical Profile:</span>
                        <span style="font-size:0.85rem;color:#94a3b8;font-weight:500;">(Integrated with Bayesian Prior)</span>
                    </div>
                    <div style="display:flex;align-items:center;gap:8px;">
                        <span style="font-size:0.82rem;color:#94a3b8;">Patient Vulnerability Index:</span>
                        <span style="background:{vuln_color}22;color:{vuln_color};border:1px solid {vuln_color}66;font-weight:700;font-size:0.85rem;padding:2px 10px;border-radius:12px;">
                            {vuln_score}/100 • {vuln_tier}
                        </span>
                    </div>
                </div>
                <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(180px, 1fr));gap:12px;">
                    <div>
                        <span style="font-size:0.8rem;color:#94a3b8;">Demographics</span><br>
                        <span style="font-weight:700;color:#f8fafc;font-size:0.95rem;">{snap.get('age', 28)} yrs • {snap.get('gender', 'Male')}</span>
                    </div>
                    <div>
                        <span style="font-size:0.8rem;color:#94a3b8;">Timeline & Chronicity</span><br>
                        <span style="font-weight:700;color:#38bdf8;font-size:0.95rem;">⏱️ {phase_str}</span>
                    </div>
                    <div>
                        <span style="font-size:0.8rem;color:#94a3b8;">Discomfort Severity</span><br>
                        <span style="font-weight:700;color:{'#f87171' if sev_num>=7 else ('#fbbf24' if sev_num>=4 else '#34d399')};font-size:0.95rem;">
                            ⚡ Level {sev_num}/10
                        </span>
                    </div>
                    <div>
                        <span style="font-size:0.8rem;color:#94a3b8;">Pre-existing Conditions</span><br>
                        <span style="font-weight:700;color:#c084fc;font-size:0.95rem;">🩺 {active_conds_str}</span>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # HERO DIAGNOSIS + CONFIRMED SYMPTOMS
        res_col1, res_col2 = st.columns([1.8, 1.2])

        badge_class = {
            "EMERGENCY": "badge-emergency",
            "See Doctor Immediately": "badge-emergency",
            "See Doctor Soon": "badge-doctor-soon",
            "Monitor 2-3 Days": "badge-monitor",
            "Self-Care": "badge-self-care"
        }.get(urgency_level, "badge-monitor")

        with res_col1:
            st.markdown(
                f"""
                <div class="glass-card" style="border-left: 6px solid #6366f1;">
                    <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
                        <h2 style="margin:0;color:#f8fafc;font-size:1.65rem;font-weight:700;">
                            {remedy_data.get('display_name', top_disease)}
                        </h2>
                        <span class="urgency-badge {badge_class}">{urgency_level}</span>
                    </div>
                    <p style="margin:8px 0 12px 0;color:#94a3b8;font-size:0.95rem;line-height:1.5;">
                        {remedy_data.get('description', '')}
                    </p>
                    <div style="display:flex;align-items:center;gap:20px;margin-top:10px;flex-wrap:wrap;">
                        <div>
                            <span style="font-size:0.85rem;color:#94a3b8;">Calibrated Clinical Confidence</span><br>
                            <span style="font-size:1.5rem;font-weight:800;color:#38bdf8;">{top_confidence:.1f}%</span>
                        </div>
                        <div style="border-left:1px solid rgba(255,255,255,0.15);padding-left:15px;">
                            <span style="font-size:0.85rem;color:#94a3b8;">Recommended Specialist</span><br>
                            <span style="font-size:1.1rem;font-weight:700;color:#c084fc;">👨‍⚕️ {remedy_data.get('specialist', 'General Physician')}</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with res_col2:
            st.markdown(
                f"""
                <div class="glass-card">
                    <h4 style="margin-top:0;margin-bottom:10px;color:#cbd5e1;">📋 Confirmed Symptoms ({len(matched_keys)})</h4>
                    <div>
                        {''.join([f'<span class="symptom-tag">✓ {s.replace("_", " ").title()}</span>' for s in matched_keys])}
                    </div>
                    <div style="margin-top:14px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.08);font-size:0.88rem;color:#cbd5e1;">
                        <b>Triage Action:</b> {action_advice}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Emergency Red Flags if any
        active_red_flags = [s for s in matched_keys if s in RED_FLAGS]
        if active_red_flags:
            rf_str = ", ".join([s.replace("_", " ").title() for s in active_red_flags])
            st.error(f"🚨 **Emergency Red Flag Warning:** Detected **{rf_str}**. This indicates a high-acuity medical condition requiring emergency department evaluation!")

        # Multi-Factor Clinical Reasoning Box
        if clinical_reasons or triage_reasons:
            st.markdown(
                f"""
                <div class="glass-card" style="border-left: 5px solid #10b981;background:rgba(16,185,129,0.05);padding:14px 18px;margin-bottom:18px;">
                    <h4 style="margin-top:0;margin-bottom:8px;color:#34d399;font-size:1.05rem;">
                        🧠 Multi-Parameter Clinical Reasoning
                    </h4>
                    <div style="font-size:0.88rem;color:#cbd5e1;line-height:1.6;">
                        {''.join([f'<div style="margin-bottom:4px;">🔹 {r}</div>' for r in clinical_reasons])}
                        {''.join([f'<div style="margin-bottom:4px;color:#fbbf24;">⚡ {t}</div>' for t in triage_reasons])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Tailored Comorbidity Precautions Alert
        if comorbidity_precautions:
            st.markdown(
                f"""
                <div class="glass-card" style="border-left: 5px solid #f59e0b;background:rgba(245,158,11,0.08);padding:14px 18px;margin-bottom:18px;">
                    <h4 style="margin-top:0;margin-bottom:8px;color:#fbbf24;font-size:1.05rem;">
                        🛡️ Personalized Comorbidity Safety Alerts ({', '.join([c for c in snap.get('conditions', []) if c != 'None'])})
                    </h4>
                    <div style="font-size:0.88rem;color:#fde68a;line-height:1.6;">
                        {''.join([f'<div style="margin-bottom:6px;">{cp}</div>' for cp in comorbidity_precautions])}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # TTS audio readout
        tts_text = f"Primary diagnosis is {remedy_data.get('display_name', top_disease)} with {top_confidence:.0f} percent certainty for a {snap.get('age', 28)} year old patient with {duration_days} days of symptoms and severity level {sev_num} out of 10. Triage status is {urgency_level}. Recommended specialist is {remedy_data.get('specialist', 'General Physician')}."
        render_tts_button(tts_text)

        # Top 3 Differentials
        st.markdown("#### 🔬 Differential Diagnoses & Probability Breakdown")
        top3 = predictions[:3]
        diff_cols = st.columns(3)
        for idx, (d_name, conf) in enumerate(top3):
            with diff_cols[idx]:
                is_top = (idx == 0)
                card_border = "#6366f1" if is_top else "rgba(255,255,255,0.1)"
                st.markdown(
                    f"""
                    <div class="glass-card" style="border:1px solid {card_border};padding:14px;">
                        <div style="font-size:0.8rem;color:#94a3b8;">Differential #{idx+1}</div>
                        <div style="font-weight:700;font-size:1.05rem;color:#f8fafc;margin:4px 0 8px 0;">{d_name}</div>
                        <div style="display:flex;align-items:center;gap:8px;">
                            <div style="flex:1;background:rgba(255,255,255,0.1);height:8px;border-radius:4px;overflow:hidden;">
                                <div style="width:{conf}%;background:linear-gradient(90deg, #6366f1, #38bdf8);height:100%;"></div>
                            </div>
                            <span style="font-size:0.9rem;font-weight:700;color:#38bdf8;">{conf:.1f}%</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # ---------------------------------------------------------
        # INTERACTIVE CARE PLAN & REMEDY CONSULTATION PROMPT
        # ---------------------------------------------------------
        st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.15);margin:28px 0 20px 0;'>", unsafe_allow_html=True)
        
        st.markdown(
            f"""
            <div class="glass-card" style="border:1.5px solid rgba(99,102,241,0.4);background:linear-gradient(135deg, rgba(30,41,59,0.95), rgba(15,23,42,0.98));padding:18px;margin-bottom:16px;">
                <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;">
                    <div>
                        <div style="font-weight:700;font-size:1.15rem;color:#f8fafc;display:flex;align-items:center;gap:8px;">
                            <span style="font-size:1.3rem;">🌿</span>
                            <span>Integrative Therapeutics & Care Plan Consultation</span>
                        </div>
                        <div style="font-size:0.9rem;color:#94a3b8;margin-top:4px;">
                            Would you like to explore <b>Home Remedies</b>, <b>Ayurvedic Formulations</b>, <b>Dietary Guidelines</b>, or <b>Preparation Recipes</b> for <b>{remedy_data.get('display_name', top_disease)}</b>?
                        </div>
                    </div>
                    <span style="background:rgba(99,102,241,0.2);color:#a5b4fc;border:1px solid rgba(99,102,241,0.4);padding:3px 12px;border-radius:14px;font-size:0.82rem;font-weight:600;">
                        Select Care Plan Option Below
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        remedy_choice = st.radio(
            "Choose what you would like to view:",
            [
                "🌿 Complete Care Package (All Remedies, Ayurvedic Kadha, Diet & Recipes)",
                "🏡 Evidence-Based Home Remedies & Ayurvedic Formulations Only",
                "🥗 Dietary Guidelines & Nutrition Protocol Only",
                "🍲 Step-by-Step Preparation Recipes & Video Guides Only",
                "🛡️ Clinical Precautions & Safety Guidelines Only",
                "❌ Clinical Diagnosis Only (Hide Remedies)"
            ],
            index=0,
            key="interactive_remedy_view_selection"
        )

        home_rems = remedy_data.get("home_remedies", [])
        ayur_rems = remedy_data.get("ayurvedic", [])
        diet_dos = remedy_data.get("diet_do", [])
        diet_donts = remedy_data.get("diet_dont", [])
        precautions = remedy_data.get("precautions", [])

        # Find matching condition guide or relevant recipes
        matching_condition = None
        for cg in CONDITION_GUIDES:
            if cg.get("name", "").lower() in top_disease.lower() or top_disease.lower() in cg.get("name", "").lower():
                matching_condition = cg
                break

        relevant_recipes = []
        for rec in REMEDY_RECIPES:
            used_for = [u.lower() for u in rec.get("usedFor", [])]
            if any(u in top_disease.lower() or top_disease.lower() in u for u in used_for) or any(s in " ".join(used_for) for s in matched_keys):
                relevant_recipes.append(rec)

        if not relevant_recipes and matching_condition:
            relevant_recipes = matching_condition.get("remedies", [])[:2]

        # RENDER SELECTED CARE PLAN
        if "Complete Care Package" in remedy_choice:
            st.markdown(
                """
                <div style="margin:20px 0 16px 0;padding:12px 18px;background:rgba(16,185,129,0.18);border:1.5px solid #10b981;border-radius:12px;">
                    <h3 style="margin:0;color:#34d399 !important;font-size:1.35rem;font-weight:800;display:flex;align-items:center;gap:10px;">
                        <span>🌿 Complete Integrative Care Package</span>
                    </h3>
                </div>
                """,
                unsafe_allow_html=True
            )
            rem_col1, rem_col2 = st.columns(2)
            with rem_col1:
                home_items_html = "".join([f'<li style="margin-bottom:10px;color:#cbd5e1;line-height:1.5;">{r}</li>' for r in home_rems])
                st.markdown(
                    f"""
                    <div class="glass-card" style="border-left:5px solid #10b981;padding:18px;">
                        <h4 style="margin-top:0;color:#34d399;font-size:1.15rem;display:flex;align-items:center;gap:8px;">
                            <span>🏡 Evidence-Based Home Remedies</span>
                        </h4>
                        <ul style="padding-left:20px;margin-bottom:0;">
                            {home_items_html}
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                ayur_items_html = "".join([f'<li style="margin-bottom:10px;color:#cbd5e1;line-height:1.5;">{a}</li>' for a in ayur_rems])
                st.markdown(
                    f"""
                    <div class="glass-card" style="border-left:5px solid #f59e0b;padding:18px;margin-top:14px;">
                        <h4 style="margin-top:0;color:#fbbf24;font-size:1.15rem;display:flex;align-items:center;gap:8px;">
                            <span>🍵 Traditional Ayurvedic Formulations & Kadha</span>
                        </h4>
                        <ul style="padding-left:20px;margin-bottom:0;">
                            {ayur_items_html}
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with rem_col2:
                diet_do_html = "".join([f'<div style="margin-bottom:6px;color:#7dd3fc;line-height:1.4;">✓ {d}</div>' for d in diet_dos])
                diet_dont_html = "".join([f'<div style="margin-bottom:6px;color:#fca5a5;line-height:1.4;">✗ {d}</div>' for d in diet_donts])
                st.markdown(
                    f"""
                    <div class="glass-card" style="border-left:5px solid #38bdf8;padding:18px;">
                        <h4 style="margin-top:0;color:#38bdf8;font-size:1.15rem;display:flex;align-items:center;gap:8px;">
                            <span>🥗 Dietary Guidelines</span>
                        </h4>
                        <div style="margin-bottom:10px;">
                            <div style="font-weight:700;color:#34d399;font-size:0.92rem;margin-bottom:6px;">Foods & Liquids to Include:</div>
                            {diet_do_html}
                        </div>
                        <div style="margin-top:12px;padding-top:10px;border-top:1px solid rgba(255,255,255,0.08);">
                            <div style="font-weight:700;color:#f87171;font-size:0.92rem;margin-bottom:6px;">Foods & Liquids to Avoid:</div>
                            {diet_dont_html}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                precautions_html = "".join([f'<li style="margin-bottom:10px;color:#fca5a5;line-height:1.5;">⚠️ {p}</li>' for p in precautions])
                st.markdown(
                    f"""
                    <div class="glass-card" style="border-left:5px solid #ef4444;padding:18px;margin-top:14px;">
                        <h4 style="margin-top:0;color:#f87171;font-size:1.15rem;display:flex;align-items:center;gap:8px;">
                            <span>🛡️ Safety Protocol & Doctor Consultation</span>
                        </h4>
                        <ul style="padding-left:20px;margin-bottom:0;">
                            {precautions_html}
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            if relevant_recipes:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(
                    """
                    <div style="margin:14px 0 12px 0;padding:10px 16px;background:rgba(245,158,11,0.15);border:1.5px solid #f59e0b;border-radius:10px;">
                        <h4 style="margin:0;color:#fbbf24 !important;font-size:1.15rem;font-weight:800;display:flex;align-items:center;gap:8px;">
                            <span>🍲 Step-by-Step Remedy Recipes & Preparation Guides</span>
                        </h4>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                rec_cols = st.columns(min(len(relevant_recipes), 2))
                for r_idx, recipe in enumerate(relevant_recipes[:2]):
                    with rec_cols[r_idx]:
                        r_name = recipe.get("name", "Home Remedy")
                        r_icon = recipe.get("icon", "🍵")
                        r_make = recipe.get("make", "")
                        r_why = recipe.get("why", "")
                        r_vid = recipe.get("video", "")
                        r_vlabel = recipe.get("videoLabel", "Medical Tutorial")
                        st.markdown(
                            f"""
                            <div class="glass-card" style="border:1px solid rgba(245,158,11,0.35);padding:16px;">
                                <div style="font-weight:700;color:#fbbf24;font-size:1.05rem;display:flex;align-items:center;gap:8px;">
                                    <span>{r_icon}</span> <span>{r_name}</span>
                                </div>
                                <div style="margin-top:8px;font-size:0.88rem;color:#cbd5e1;line-height:1.5;">
                                    <b style="color:#38bdf8;">How to Prepare:</b> {r_make}
                                </div>
                                <div style="margin-top:8px;font-size:0.84rem;color:#94a3b8;line-height:1.4;">
                                    <b style="color:#34d399;">Biological Mechanism:</b> {r_why}
                                </div>
                                {f'<div style="margin-top:10px;"><a href="{r_vid}" target="_blank" style="color:#f87171;font-weight:600;font-size:0.82rem;text-decoration:none;">▶ Watch Video Guide: {r_vlabel}</a></div>' if r_vid else ''}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

        elif "Home Remedies & Ayurvedic" in remedy_choice:
            st.markdown(
                """
                <div style="margin:20px 0 16px 0;padding:12px 18px;background:rgba(16,185,129,0.18);border:1.5px solid #10b981;border-radius:12px;">
                    <h3 style="margin:0;color:#34d399 !important;font-size:1.35rem;font-weight:800;display:flex;align-items:center;gap:10px;">
                        <span>🏡 Evidence-Based Home Remedies & Ayurvedic Care</span>
                    </h3>
                </div>
                """,
                unsafe_allow_html=True
            )
            rem_col1, rem_col2 = st.columns(2)
            with rem_col1:
                home_items_html = "".join([f'<li style="margin-bottom:10px;color:#cbd5e1;line-height:1.5;">{r}</li>' for r in home_rems])
                st.markdown(
                    f"""
                    <div class="glass-card" style="border-left:5px solid #10b981;padding:18px;">
                        <h4 style="margin-top:0;color:#34d399;font-size:1.15rem;display:flex;align-items:center;gap:8px;">
                            <span>🏡 Evidence-Based Home Remedies</span>
                        </h4>
                        <ul style="padding-left:20px;margin-bottom:0;">
                            {home_items_html}
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with rem_col2:
                ayur_items_html = "".join([f'<li style="margin-bottom:10px;color:#cbd5e1;line-height:1.5;">{a}</li>' for a in ayur_rems])
                st.markdown(
                    f"""
                    <div class="glass-card" style="border-left:5px solid #f59e0b;padding:18px;">
                        <h4 style="margin-top:0;color:#fbbf24;font-size:1.15rem;display:flex;align-items:center;gap:8px;">
                            <span>🍵 Traditional Ayurvedic Formulations & Kadha</span>
                        </h4>
                        <ul style="padding-left:20px;margin-bottom:0;">
                            {ayur_items_html}
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        elif "Dietary Guidelines" in remedy_choice:
            st.markdown(
                """
                <div style="margin:20px 0 16px 0;padding:12px 18px;background:rgba(56,189,248,0.18);border:1.5px solid #38bdf8;border-radius:12px;">
                    <h3 style="margin:0;color:#38bdf8 !important;font-size:1.35rem;font-weight:800;display:flex;align-items:center;gap:10px;">
                        <span>🥗 Dietary Guidelines & Nutritional Protocol</span>
                    </h3>
                </div>
                """,
                unsafe_allow_html=True
            )
            diet_do_html = "".join([f'<div style="margin-bottom:8px;color:#7dd3fc;font-size:0.95rem;line-height:1.4;">✓ {d}</div>' for d in diet_dos])
            diet_dont_html = "".join([f'<div style="margin-bottom:8px;color:#fca5a5;font-size:0.95rem;line-height:1.4;">✗ {d}</div>' for d in diet_donts])
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown(
                    f"""
                    <div class="glass-card" style="border-left:5px solid #10b981;padding:18px;">
                        <h4 style="margin-top:0;color:#34d399;font-size:1.15rem;">✓ Foods & Liquids to Include</h4>
                        {diet_do_html}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with col_d2:
                st.markdown(
                    f"""
                    <div class="glass-card" style="border-left:5px solid #ef4444;padding:18px;">
                        <h4 style="margin-top:0;color:#f87171;font-size:1.15rem;">✗ Foods & Liquids to Avoid</h4>
                        {diet_dont_html}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        elif "Step-by-Step" in remedy_choice:
            st.markdown(
                """
                <div style="margin:20px 0 16px 0;padding:12px 18px;background:rgba(245,158,11,0.18);border:1.5px solid #f59e0b;border-radius:12px;">
                    <h3 style="margin:0;color:#fbbf24 !important;font-size:1.35rem;font-weight:800;display:flex;align-items:center;gap:10px;">
                        <span>🍲 Step-by-Step Preparation Recipes & Medical Video Tutorials</span>
                    </h3>
                </div>
                """,
                unsafe_allow_html=True
            )
            if relevant_recipes:
                for recipe in relevant_recipes:
                    r_name = recipe.get("name", "Home Remedy")
                    r_icon = recipe.get("icon", "🍵")
                    r_make = recipe.get("make", "")
                    r_why = recipe.get("why", "")
                    r_vid = recipe.get("video", "")
                    r_vlabel = recipe.get("videoLabel", "Medical Tutorial")
                    st.markdown(
                        f"""
                        <div class="glass-card" style="border:1px solid rgba(245,158,11,0.4);padding:20px;margin-bottom:14px;">
                            <div style="font-weight:700;color:#fbbf24;font-size:1.15rem;display:flex;align-items:center;gap:10px;">
                                <span style="font-size:1.3rem;">{r_icon}</span> <span>{r_name}</span>
                            </div>
                            <div style="margin-top:10px;font-size:0.92rem;color:#cbd5e1;line-height:1.6;">
                                <b style="color:#38bdf8;">How to Prepare:</b> {r_make}
                            </div>
                            <div style="margin-top:10px;font-size:0.88rem;color:#94a3b8;line-height:1.5;">
                                <b style="color:#34d399;">Biological Mechanism:</b> {r_why}
                            </div>
                            {f'<div style="margin-top:12px;"><a href="{r_vid}" target="_blank" style="background:rgba(239,68,68,0.15);color:#f87171;border:1px solid rgba(239,68,68,0.4);padding:6px 14px;border-radius:18px;font-weight:600;font-size:0.85rem;text-decoration:none;display:inline-block;">▶ Watch YouTube Video Tutorial: {r_vlabel}</a></div>' if r_vid else ''}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
            else:
                st.info(f"General home care hydration and rest are recommended for {remedy_data.get('display_name', top_disease)}.")

        elif "Clinical Precautions" in remedy_choice:
            st.markdown(
                """
                <div style="margin:20px 0 16px 0;padding:12px 18px;background:rgba(239,68,68,0.18);border:1.5px solid #ef4444;border-radius:12px;">
                    <h3 style="margin:0;color:#f87171 !important;font-size:1.35rem;font-weight:800;display:flex;align-items:center;gap:10px;">
                        <span>🛡️ Clinical Precautions & Safety Guidelines</span>
                    </h3>
                </div>
                """,
                unsafe_allow_html=True
            )
            precautions_html = "".join([f'<li style="margin-bottom:10px;color:#fca5a5;line-height:1.5;">⚠️ {p}</li>' for p in precautions])
            st.markdown(
                f"""
                <div class="glass-card" style="border-left:5px solid #ef4444;padding:20px;">
                    <h4 style="margin-top:0;color:#f87171;font-size:1.15rem;">
                        🛡️ Safety Protocol & Red Flag Warnings
                    </h4>
                    <ul style="padding-left:20px;margin-bottom:0;">
                        {precautions_html}
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:
            st.info("ℹ️ Home remedies and care plan are hidden as selected. You can switch options above anytime.")

        # BOTTOM BACK BUTTON
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⬅️ Back to Symptom Input & Voice Console", type="primary", use_container_width=True, key="btn_bottom_back_to_input"):
            st.session_state["active_view"] = "input"
            st.rerun()

    # ---------------------------------------------------------
    # ROUTE 2: INPUT CONSOLE VIEW (Voice, Text, Presets, Chips, Live Preview)
    # ---------------------------------------------------------
    else:
        # Show banner if voice query was just captured
        if st.session_state.get("voice_banner_msg"):
            st.success(f"🎙️ **Spoken Symptoms Transferred:** \"{st.session_state['voice_banner_msg']}\"")

        col_input, col_presets = st.columns([2.2, 1.1])

        with col_presets:
            st.markdown("#### ⚡ Quick Presets")
            presets = {
                "Select a preset...": "",
                "Typhoid / Cold: Fever, cough, cold, body pain": "fever, cough, cold and bodypain with headache",
                "Food Poisoning: Vomiting & loose motion": "severe vomiting, dehydration and loose motion",
                "Infection: High fever with chills & shivering": "high fever, violent shivering, chills and sweating",
                "Skin: Itching & red skin rash": "itching, skin rash and nodal skin eruptions",
                "UTI: Burning urination & bladder pain": "burning urination, foul smell of urine and bladder discomfort",
                "Migraine: Throbbing headache & aura": "throbbing headache, visual disturbances and blurred vision",
                "GERD: Heartburn, acidity & chest burn": "acidity, heartburn, burning chest and stomach pain",
                "Jaundice: Yellow skin, dark urine & fatigue": "yellowish skin, dark urine, yellowing of eyes and fatigue"
            }
            selected_preset = st.selectbox("Test clinical scenarios:", list(presets.keys()), index=0, key="scenario_preset_select")

            # Handle preset selection change
            if selected_preset != "Select a preset..." and presets.get(selected_preset):
                if st.session_state.get("last_preset_choice") != selected_preset:
                    st.session_state["last_preset_choice"] = selected_preset
                    st.session_state["patient_symptoms_text_box"] = presets[selected_preset]
                    st.session_state["input_text"] = presets[selected_preset]

        with col_input:
            st.markdown("#### 🗣️ Enter or Speak Symptoms")

            # Render Voice Input Widget (Microphone)
            render_voice_input_widget()

            # Initialize session state for text box if not present
            if "patient_symptoms_text_box" not in st.session_state:
                st.session_state["patient_symptoms_text_box"] = st.session_state.get("input_text", "fever, cough, cold and bodypain")

            symptom_query = st.text_area(
                "Patient Symptoms (Spoken or Typed)",
                height=90,
                placeholder="E.g., I have fever, severe cough, cold, and body pain for 3 days...",
                help="Speak via the microphone button above or type your symptoms here.",
                key="patient_symptoms_text_box"
            )
            st.session_state["input_text"] = symptom_query

        # Common Symptom Chips Selector
        st.markdown("##### 🏷️ Quick Symptom Chips (Click to combine):")
        common_chips = [
            "Fever", "High Fever", "Cough", "Cold", "Body Pain", "Headache",
            "Vomiting", "Loose Motion", "Dehydration", "Stomach Pain", "Acidity",
            "Chills", "Shivering", "Sweating", "Itching", "Skin Rash",
            "Burning Urination", "Bladder Pain", "Yellow Skin", "Dark Urine",
            "Breathlessness", "Chest Pain", "Joint Pain", "Throat Irritation", "Dizziness"
        ]
        
        selected_chips = st.multiselect(
            "Select symptoms to add:",
            common_chips,
            default=[],
            help="Selected chips are automatically merged with your symptom text.",
            key="patient_symptoms_chips_select"
        )

        # Combine text area with selected chips
        combined_query = symptom_query.strip()
        if selected_chips:
            chips_text = ", ".join(selected_chips).lower()
            if chips_text not in combined_query.lower():
                combined_query = (combined_query + ", " + chips_text).strip(", ")

        # ---------------------------------------------------------
        # LIVE DETECTED SYMPTOMS PREVIEW
        # ---------------------------------------------------------
        matched, vector = parse_symptoms(combined_query, feature_cols)
        matched_keys = list(matched.keys())

        st.markdown(
            f"""
            <div class="glass-card" style="margin-top:10px;padding:14px 18px;border-left:5px solid #38bdf8;">
                <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:8px;">
                    <div style="font-weight:700;font-size:1.05rem;color:#f8fafc;display:flex;align-items:center;gap:8px;">
                        <span>📋 Detected Clinical Symptoms:</span>
                        <span style="background:rgba(56,189,248,0.2);color:#38bdf8;padding:2px 10px;border-radius:12px;font-size:0.85rem;">
                            {len(matched_keys)} identified
                        </span>
                    </div>
                    <div style="font-size:0.8rem;color:#94a3b8;">
                        Ready for XGBoost Multi-Class Inference
                    </div>
                </div>
                <div>
                    {''.join([f'<span class="symptom-tag" style="background:rgba(56,189,248,0.2);color:#7dd3fc;border-color:#0284c7;">✓ {s.replace("_", " ").title()}</span>' for s in matched_keys]) if matched_keys else '<span style="color:#94a3b8;font-size:0.9rem;">No clinical symptoms detected yet. Speak or type symptoms above (e.g. fever, headache, vomiting, loose motion, cold).</span>'}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Detect if any Red Flag symptoms are present and alert immediately
        active_red_flags = [s for s in matched_keys if s in RED_FLAGS]
        if active_red_flags:
            rf_str = ", ".join([s.replace("_", " ").title() for s in active_red_flags])
            st.error(f"🚨 **Emergency Red Flag Warning:** Detected **{rf_str}**. This may indicate a critical emergency condition. Do not delay hospital care!")

        # Predict Button
        btn_predict = st.button("🔍 Predict Disease (XGBoost Analysis)", type="primary", use_container_width=True, key="btn_run_prediction")

        # Check if prediction is requested
        should_run_prediction = btn_predict or st.session_state.pop("trigger_predict_now", False)

        if should_run_prediction:
            if not combined_query:
                st.warning("⚠️ Please speak or enter symptoms first before predicting.")
            elif not matched:
                st.error("⚠️ No recognizable clinical symptoms found in your input. Try words like fever, cough, loose motion, vomiting, cold, headache, chills, shivering, itching, etc.")
            else:
                with st.spinner("Analyzing symptoms through XGBoost & Bayesian Triage Engine..."):
                    # Perform Multi-Factor Comprehensive Clinical Inference
                    predictions, clinical_reasons = predict_clinical_comprehensive(
                        xgb_model,
                        label_encoder,
                        vector,
                        matched_keys,
                        patient_age=int(age),
                        patient_gender=gender,
                        days=int(days),
                        severity=int(severity),
                        existing_conditions=existing_cond
                    )
                    top_disease, top_confidence = predictions[0]

                    # Assess Comprehensive Urgency & Vulnerability Index
                    urgency_level, action_advice, triage_reasons, vuln_score, vuln_tier = assess_urgency_comprehensive(
                        matched_keys,
                        days=int(days),
                        severity=int(severity),
                        confidence=float(top_confidence),
                        patient_age=int(age),
                        existing_conditions=existing_cond
                    )

                    # Retrieve Clinical Knowledge & Remedies
                    remedy_data = get_remedies_for_disease(top_disease)

                    # Tailored comorbidity precautions
                    comorbidity_precautions = get_comorbidity_tailored_precautions(existing_cond, top_disease)

                    # Store in session state and transition to dedicated Report View!
                    st.session_state["last_prediction_data"] = {
                        "top_disease": top_disease,
                        "top_confidence": top_confidence,
                        "urgency_level": urgency_level,
                        "action_advice": action_advice,
                        "triage_reasons": triage_reasons,
                        "vuln_score": vuln_score,
                        "vuln_tier": vuln_tier,
                        "clinical_reasons": clinical_reasons,
                        "comorbidity_precautions": comorbidity_precautions,
                        "remedy_data": remedy_data,
                        "predictions": predictions,
                        "matched_keys": matched_keys,
                        "patient_snapshot": {
                            "age": int(age),
                            "gender": gender,
                            "days": int(days),
                            "severity": int(severity),
                            "conditions": existing_cond
                        }
                    }
                    st.session_state["active_view"] = "report"
                    st.rerun()



# =========================================================
# TAB 2: XGBoost Mathematical Engine & Model Performance
# =========================================================
with tab2:
    st.markdown("### 🧮 XGBoost Mathematical Engine & Formal Formulations")
    st.markdown(
        "Complete mathematical foundation of **Extreme Gradient Boosting (XGBoost)** including Second-Order Taylor Approximations, Exact Greedy Split Finding, Regularization, and Softmax Multi-Class Probability."
    )

    # Key Performance Metric Cards
    if metrics_data:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test Accuracy", f"{metrics_data.get('test_accuracy', 99.86)}%")
        m2.metric("5-Fold CV Mean", f"{metrics_data.get('cv_5fold_mean', 99.78)}%", f"± {metrics_data.get('cv_5fold_std', 0.11)}%")
        m3.metric("Macro F1-Score", f"{metrics_data.get('f1_macro', 99.87)}%")
        m4.metric("Multi-Class Log-Loss", f"{metrics_data.get('log_loss', 0.0087)}")

    st.markdown("<br>", unsafe_allow_html=True)

    math_col1, math_col2 = st.columns(2)

    with math_col1:
        st.markdown(
            """
            <div class="formula-box">
                <h4>1. Regularized Objective Function</h4>
                <p>For a tree ensemble of $K$ additive functions, XGBoost minimizes the regularized loss:</p>
            """,
            unsafe_allow_html=True
        )
        st.latex(r"\mathcal{L}^{(t)} = \sum_{i=1}^{n} l\left(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\right) + \Omega(f_t)")
        st.markdown(
            """
            where the tree complexity regularization $\Omega(f_t)$ is defined as:
            """,
            unsafe_allow_html=True
        )
        st.latex(r"\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^{T} w_j^2")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="formula-box">
                <h4>2. Second-Order Taylor Expansion</h4>
                <p>Approximating the loss using first-order gradient $g_i$ and second-order Hessian $h_i$:</p>
            """,
            unsafe_allow_html=True
        )
        st.latex(r"\tilde{\mathcal{L}}^{(t)} \approx \sum_{i=1}^{n} \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2")
        st.latex(r"g_i = \frac{\partial l(y_i, \hat{y}^{(t-1)})}{\partial \hat{y}^{(t-1)}}, \quad h_i = \frac{\partial^2 l(y_i, \hat{y}^{(t-1)})}{\partial (\hat{y}^{(t-1)})^2}")
        st.markdown("</div>", unsafe_allow_html=True)

    with math_col2:
        st.markdown(
            """
            <div class="formula-box">
                <h4>3. Optimal Leaf Weight Solution</h4>
                <p>For a fixed tree structure $q(x)$, the optimal weight $w_j^*$ for leaf $j$ is analytically computed as:</p>
            """,
            unsafe_allow_html=True
        )
        st.latex(r"w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}")
        st.latex(r"\tilde{\mathcal{L}}^{(t)}(q) = -\frac{1}{2} \sum_{j=1}^{T} \frac{\left(\sum_{i \in I_j} g_i\right)^2}{\sum_{i \in I_j} h_i + \lambda} + \gamma T")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div class="formula-box">
                <h4>4. Exact Greedy Split Finding (Gain)</h4>
                <p>The reduction in loss after splitting a leaf into left ($I_L$) and right ($I_R$) sub-nodes:</p>
            """,
            unsafe_allow_html=True
        )
        st.latex(r"\text{Gain} = \frac{1}{2} \left[ \frac{\left(\sum_{i \in I_L} g_i\right)^2}{\sum_{i \in I_L} h_i + \lambda} + \frac{\left(\sum_{i \in I_R} g_i\right)^2}{\sum_{i \in I_R} h_i + \lambda} - \frac{\left(\sum_{i \in I} g_i\right)^2}{\sum_{i \in I} h_i + \lambda} \right] - \gamma")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Multi-Class Probability & Evaluation Math
    math_row2_1, math_row2_2 = st.columns(2)

    with math_row2_1:
        st.markdown(
            """
            <div class="formula-box">
                <h4>5. Multi-Class Softmax Probability</h4>
                <p>For $K=30$ disease classes, predicted class probabilities are calculated via Softmax:</p>
            """,
            unsafe_allow_html=True
        )
        st.latex(r"P(y_i = k \mid x_i) = \frac{e^{f_k(x_i)}}{\sum_{j=1}^{K} e^{f_j(x_i)}}")
        st.latex(r"L_{\log}(y, P) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{k=1}^{K} y_{i,k} \ln(p_{i,k})")
        st.markdown("</div>", unsafe_allow_html=True)

    with math_row2_2:
        st.markdown(
            """
            <div class="formula-box">
                <h4>6. Stratified 5-Fold Cross-Validation</h4>
                <p>Fold scores: <code>[99.73%, 99.86%, 99.86%, 99.73%, 99.73%]</code></p>
            """,
            unsafe_allow_html=True
        )
        st.latex(r"\mu_{CV} = \frac{1}{5}\sum_{k=1}^{5} \text{Acc}_k = 99.78\%, \quad \sigma_{CV} = 0.11\%")
        st.markdown("</div>", unsafe_allow_html=True)

    # Visualizations: Feature Importance & Confusion Matrix
    st.markdown("### 📊 Top Feature Importances & Confusion Matrix")

    vis_col1, vis_col2 = st.columns([1.1, 1.3])

    with vis_col1:
        if metrics_data and "top_features" in metrics_data:
            top_feats = metrics_data["top_features"][:15]
            f_names = [item["feature"] for item in top_feats][::-1]
            f_scores = [item["score"] for item in top_feats][::-1]

            fig_feat, ax_feat = plt.subplots(figsize=(7, 6))
            fig_feat.patch.set_facecolor('#0f172a')
            ax_feat.set_facecolor('#1e293b')

            ax_feat.barh(f_names, f_scores, color='#38bdf8', edgecolor='none')
            ax_feat.set_title('Top 15 Informative Symptoms (XGBoost Gain)', color='#f8fafc', fontsize=12, fontweight='bold')
            ax_feat.set_xlabel('Relative Importance (Gain)', color='#cbd5e1')
            ax_feat.tick_params(colors='#94a3b8', labelsize=8)
            ax_feat.grid(axis='x', linestyle='--', alpha=0.2)
            plt.tight_layout()
            st.pyplot(fig_feat)

    with vis_col2:
        if metrics_data and "confusion_matrix" in metrics_data:
            cm = np.array(metrics_data["confusion_matrix"])
            classes = metrics_data["classes"]

            fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
            fig_cm.patch.set_facecolor('#0f172a')
            ax_cm.set_facecolor('#0f172a')

            sns.heatmap(
                cm,
                annot=False,
                cmap="Blues",
                xticklabels=classes,
                yticklabels=classes,
                ax=ax_cm,
                cbar_kws={'label': 'Count'}
            )
            ax_cm.set_title('Multi-Class Confusion Matrix (30 Classes)', color='#f8fafc', fontsize=12, fontweight='bold')
            ax_cm.set_xlabel('Predicted Disease', color='#cbd5e1', fontsize=9)
            ax_cm.set_ylabel('True Disease', color='#cbd5e1', fontsize=9)
            ax_cm.tick_params(colors='#94a3b8', labelsize=6)
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            plt.tight_layout()
            st.pyplot(fig_cm)


# =========================================================
# TAB 3: Disease & Remedy Encyclopedia
# =========================================================
with tab3:
    st.markdown("### 📖 Clinical Disease & Home Remedies Encyclopedia")
    st.markdown(
        "Explore evidence-based home remedies, step-by-step recipe preparations, verified video tutorials (WebMD, Mayo Clinic), and clinical profiles across all major body systems and 30 clinical diseases."
    )

    guide_mode = st.radio(
        "Select Knowledge View:",
        ["🌿 OpenCare Interactive Home Remedies & Video Guides", "📚 30 Clinical Disease Encyclopedia"],
        horizontal=True,
        key="encyclopedia_view_mode_toggle"
    )

    if guide_mode == "🌿 OpenCare Interactive Home Remedies & Video Guides":
        # ---------------------------------------------------------
        # Section A: OpenCare Interactive Home Remedies Guide
        # ---------------------------------------------------------
        col_cat, col_srch = st.columns([1.2, 2.0])
        with col_cat:
            selected_cat = st.selectbox(
                "Filter by Health Category:",
                CATEGORIES_LIST,
                index=0,
                key="remedy_category_filter_select"
            )
        with col_srch:
            remedy_search = st.text_input(
                "🔍 Search conditions, symptoms, or remedies:",
                placeholder="E.g. Cold, Flu, Sore Throat, Ginger, Honey, Acid Reflux, Burns...",
                key="remedy_guide_search_box"
            )

        # Filter Conditions
        filtered_conditions = []
        for c in CONDITION_GUIDES:
            # Category match
            if selected_cat != "All Categories" and c.get("category") != selected_cat:
                continue
            # Search match
            if remedy_search.strip():
                qs = remedy_search.lower()
                matches = (
                    qs in c.get("name", "").lower() or
                    qs in c.get("causes", "").lower() or
                    qs in c.get("symptoms", "").lower() or
                    qs in c.get("category", "").lower() or
                    any(qs in r.get("name", "").lower() or qs in r.get("make", "").lower() for r in c.get("remedies", []))
                )
                if not matches:
                    continue
            filtered_conditions.append(c)

        st.caption(f"Showing **{len(filtered_conditions)}** condition guides in **{selected_cat}**:")

        for cond in filtered_conditions:
            cat_badge_color = {
                "Respiratory": "#38bdf8",
                "Digestive": "#34d399",
                "Aches & Pain": "#f87171",
                "Sleep & Stress": "#c084fc",
                "Skin & Surface": "#fbbf24"
            }.get(cond.get("category"), "#818cf8")

            with st.expander(f"{cond.get('icon', '🩺')} {cond.get('name')}  •  [{cond.get('tag')}]", expanded=False):
                st.markdown(
                    f"""
                    <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:12px;">
                        <span style="font-weight:700;font-size:1.15rem;color:#f8fafc;">
                            {cond.get('name')}
                        </span>
                        <span style="background:{cat_badge_color}22;color:{cat_badge_color};border:1px solid {cat_badge_color}66;padding:3px 12px;border-radius:14px;font-size:0.82rem;font-weight:700;">
                            {cond.get('category')}
                        </span>
                    </div>
                    <div class="glass-card" style="padding:12px 16px;margin-bottom:12px;border-left:4px solid {cat_badge_color};">
                        <div style="font-size:0.88rem;color:#cbd5e1;margin-bottom:6px;">
                            <b>🔬 Causes & Triggers:</b> {cond.get('causes')}
                        </div>
                        <div style="font-size:0.88rem;color:#94a3b8;">
                            <b>📋 Common Symptoms:</b> {cond.get('symptoms')}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("#### 🥣 Evidence-Based Home Remedies & Preparation:")
                for r in cond.get("remedies", []):
                    st.markdown(
                        f"""
                        <div class="glass-card" style="padding:14px;margin-bottom:10px;background:rgba(30,41,59,0.85);border:1px solid rgba(255,255,255,0.08);">
                            <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;margin-bottom:8px;">
                                <div style="font-weight:700;color:#38bdf8;font-size:1rem;">
                                    🌿 {r.get('name')}
                                </div>
                                <div>
                                    <a href="{r.get('video')}" target="_blank" style="background:rgba(239,68,68,0.2);color:#f87171;border:1px solid rgba(239,68,68,0.5);padding:4px 12px;border-radius:16px;font-size:0.78rem;font-weight:700;text-decoration:none;display:inline-flex;align-items:center;gap:4px;">
                                        🎬 {r.get('videoLabel', 'Watch Video Guide')} ↗
                                    </a>
                                </div>
                            </div>
                            <div style="font-size:0.86rem;color:#f1f5f9;margin-bottom:6px;line-height:1.5;">
                                <b>🥄 How to Make & Prepare:</b> {r.get('make')}
                            </div>
                            <div style="font-size:0.84rem;color:#94a3b8;line-height:1.5;border-top:1px dashed rgba(255,255,255,0.08);padding-top:6px;margin-top:6px;">
                                <b>💡 Why It Works:</b> {r.get('why')}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                st.markdown(
                    f"""
                    <div style="background:rgba(239,68,68,0.1);border:1px solid rgba(239,68,68,0.3);border-radius:10px;padding:10px 14px;font-size:0.84rem;color:#fca5a5;margin-top:10px;">
                        ⚠️ <b>Clinical Safety Caution:</b> {cond.get('caution')}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # Dedicated Recipe Index Drawer
        with st.expander("📚 Browse All 24 Individual Home Remedy Recipes (Step-by-Step Index)", expanded=False):
            st.markdown("Complete recipe cards with precise preparation ratios, active compounds, and video tutorials:")
            r_cols = st.columns(2)
            for idx, item in enumerate(REMEDY_RECIPES):
                col_target = r_cols[idx % 2]
                with col_target:
                    st.markdown(
                        f"""
                        <div class="glass-card" style="padding:14px;margin-bottom:12px;border-left:4px solid #6366f1;">
                            <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
                                <div style="font-weight:700;color:#f8fafc;font-size:0.95rem;">
                                    {item.get('icon')} {item.get('name')}
                                </div>
                                <span style="font-size:0.75rem;background:rgba(99,102,241,0.2);color:#a5b4fc;padding:2px 8px;border-radius:10px;">
                                    {item.get('category')}
                                </span>
                            </div>
                            <div style="font-size:0.82rem;color:#cbd5e1;margin-bottom:6px;line-height:1.4;">
                                <b>Preparation:</b> {item.get('make')}
                            </div>
                            <div style="font-size:0.8rem;color:#94a3b8;margin-bottom:8px;line-height:1.4;">
                                <b>Mechanism:</b> {item.get('why')}
                            </div>
                            <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:6px;border-top:1px solid rgba(255,255,255,0.06);padding-top:6px;">
                                <span style="font-size:0.75rem;color:#64748b;">Used for: {', '.join(item.get('usedFor', []))}</span>
                                <a href="{item.get('video')}" target="_blank" style="font-size:0.78rem;color:#38bdf8;text-decoration:none;font-weight:600;">
                                    🎬 {item.get('videoLabel')} ↗
                                </a>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    else:
        # ---------------------------------------------------------
        # Section B: Full 30 Clinical Diseases Knowledge Base
        # ---------------------------------------------------------
        search_term = st.text_input(
            "🔍 Search across all 30 medical conditions or symptoms:",
            placeholder="E.g. Typhoid, Dengue, Malaria, Peptic Ulcer, Spondylosis, Jaundice, Asthma...",
            key="encyclopedia_search_input_box"
        )

        filtered_list = []
        for d_k, d_v in DISEASE_KNOWLEDGE.items():
            if not search_term.strip():
                filtered_list.append((d_k, d_v))
            else:
                q = search_term.lower()
                if (
                    q in d_k.lower() or 
                    q in d_v.get("display_name", "").lower() or
                    q in d_v.get("description", "").lower() or
                    any(q in s.lower() for s in d_v.get("primary_symptoms", []))
                ):
                    filtered_list.append((d_k, d_v))

        st.caption(f"Displaying **{len(filtered_list)}** matching conditions from the 30-disease clinical corpus:")

        for d_k, d_v in filtered_list:
            with st.expander(f"🩺 {d_v.get('display_name', d_k)} — Recommended Specialist: {d_v.get('specialist', 'General Physician')}"):
                st.markdown(f"**Medical Summary:** {d_v.get('description', '')}")

                ec1, ec2 = st.columns(2)
                with ec1:
                    st.markdown("##### 🏡 Evidence-Based Home Remedies")
                    for r in d_v.get("home_remedies", []):
                        st.markdown(f"• {r}")

                    st.markdown("##### 🍵 Ayurvedic & Herbal Treatments")
                    for a in d_v.get("ayurvedic", []):
                        st.markdown(f"• {a}")

                with ec2:
                    st.markdown("##### 🥗 Diet Guidelines")
                    st.markdown("**Foods to Include:** " + ", ".join(d_v.get("diet_do", [])))
                    st.markdown("**Foods to Avoid:** " + ", ".join(d_v.get("diet_dont", [])))

                    st.markdown("##### 🛡️ Precautions & Safety")
                    for p in d_v.get("precautions", []):
                        st.markdown(f"⚠️ {p}")


# =========================================================
# TAB 4: Emergency Red Flags
# =========================================================
with tab4:
    st.markdown("### ⚠️ Emergency Red Flags & Critical Triage Protocol")
    st.markdown(
        """
        <div class="glass-card" style="border-left:6px solid #ef4444;">
            <h3 style="margin-top:0;color:#f87171;">🚨 Life-Threatening Red Flag Symptoms</h3>
            <p>If the patient exhibits any of the following symptoms, <b>do not attempt home self-care</b>. Transport the patient immediately to the nearest Hospital Emergency Room.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    rfc1, rfc2 = st.columns(2)

    with rfc1:
        st.markdown(
            """
            - 🚨 **Chest Pain or Crushing Sensation** (potential heart attack / acute coronary syndrome)
            - 🚨 **Severe Shortness of Breath / Air Gasping** (acute respiratory failure / severe asthma attack)
            - 🚨 **Coughing up Blood (Hemoptysis)**
            - 🚨 **Vomiting Blood or Black Tarry Stool** (active gastrointestinal bleeding)
            - 🚨 **Sudden Weakness or Paralysis on One Side of the Body** (acute stroke)
            """
        )

    with rfc2:
        st.markdown(
            """
            - 🚨 **Slurred Incoherent Speech or Facial Drooping**
            - 🚨 **Unresponsiveness, Confusion, or Stupor**
            - 🚨 **High Fever Spikes with Rigid Neck Stiffness** (meningitis indicator)
            - 🚨 **Severe Sunken Eyes with Zero Urine Output for >6 Hours** (critical dehydration)
            - 🚨 **Rapid Swelling of Lips, Tongue, or Throat** (anaphylaxis)
            """
        )


# =========================================================
# TAB 5: Dataset & Feature Explorer
# =========================================================
with tab5:
    st.markdown("### 🧪 Dataset & Symptom Feature Explorer")
    st.markdown(f"The training dataset contains **{len(dataset)}** rows with **{len(feature_cols)}** binary symptom features across **30** clinical diseases.")

    ds1, ds2, ds3 = st.columns(3)
    ds1.metric("Total Records", len(dataset))
    ds2.metric("Symptom Features", len(feature_cols))
    ds3.metric("Disease Classes", dataset["Disease"].nunique())

    st.markdown("#### 📊 Disease Sample Frequencies")
    st.bar_chart(dataset["Disease"].value_counts())

    st.markdown("#### 📋 Dataset Sample Preview")
    st.dataframe(dataset.head(25), use_container_width=True)

