"""
Clinical Disease Prediction System & XGBoost Machine Learning Assistant.
Features:
- Dedicated Two-Page Flow: Clean Input Console (Page 1) -> Full Diagnostic Report (Page 2)
- Interactive Home Remedy Choice: Asks user whether to view Home Remedies or Clinical Report only
- Automatic 'See Doctor Immediately' escalation when symptom duration >= 5-6 days
- Dual Voice & Text Input with Web Speech Recognition & Audio Synthesis
- Focused on the Best-in-Class XGBoost Classifier (99.86% Accuracy)
- Complete Mathematical Derivations & Formulas for XGBoost
- Actionable Home Remedies, Ayurvedic Guidance, Dietary Advice, & Specialist Recommendations
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
    assess_urgency_comprehensive,
    get_remedies_for_disease,
    predict_clinical_comprehensive,
    get_comorbidity_tailored_precautions,
    get_comorbidity_tailored_precautions_kn,
    detect_kannada_input,
    get_kannada_report_data,
    get_kannada_symptom_display,
    SYMPTOM_NAMES_KN,
    DISEASE_KNOWLEDGE_KN,
    normalize_patient_gender,
    normalize_patient_conditions,
    get_kannada_gender_display,
    get_kannada_conditions_display,
    GENDER_MAP_KN,
    CONDITIONS_MAP_KN
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

# Custom Healthcare Blue & White Combo Theme Styling
st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
        
        * {
            font-family: 'Plus Jakarta Sans', sans-serif;
        }

        /* Hide Streamlit Deploy Button, Toolbar, and Default Header */
        .stDeployButton,
        div[data-testid="stToolbar"],
        #MainMenu,
        footer,
        header[data-testid="stHeader"] {
            display: none !important;
            visibility: hidden !important;
            height: 0 !important;
            padding: 0 !important;
            margin: 0 !important;
        }

        /* Blue & Navy Theme Background */
        .stApp {
            background: linear-gradient(135deg, #090d16 0%, #0f172a 50%, #1e1b4b 100%) !important;
            color: #f8fafc !important;
        }

        /* Sidebar: Sleek Dark Navy with 100% White Visible Text */
        section[data-testid="stSidebar"],
        div[data-testid="stSidebar"],
        div[data-testid="stSidebarContent"],
        div[data-testid="stSidebarUserContent"],
        div[data-testid="stSidebarHeader"] {
            background-color: #0b1329 !important;
            background: linear-gradient(180deg, #0b1329 0%, #0f172a 100%) !important;
            border-right: 1.5px solid rgba(255, 255, 255, 0.12) !important;
            color: #ffffff !important;
        }

        /* Sidebar Headers */
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3,
        section[data-testid="stSidebar"] h4 {
            color: #38bdf8 !important;
            font-weight: 800 !important;
        }

        /* Sidebar Labels (Patient Age, Gender, Severity, etc.) */
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] label p,
        section[data-testid="stSidebar"] div[data-testid="stWidgetLabel"] p,
        section[data-testid="stSidebar"] div[data-testid="stWidgetLabel"] label,
        section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p {
            color: #ffffff !important;
            font-weight: 700 !important;
            font-size: 0.95rem !important;
            opacity: 1 !important;
        }

        /* Sidebar Inputs: White/Cream Boxes */
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
            background-color: #ffffff !important;
            color: #0f172a !important;
            border: 1.5px solid #cbd5e1 !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
        }

        section[data-testid="stSidebar"] div[data-baseweb="select"] * {
            color: #0f172a !important;
            font-weight: 600 !important;
        }

        /* Sidebar Sliders Text & Values */
        section[data-testid="stSidebar"] div[data-testid="stSlider"] p,
        section[data-testid="stSidebar"] div[data-testid="stSlider"] span {
            color: #38bdf8 !important;
            font-weight: 700 !important;
        }

        /* High-Contrast Headings & Typography */
        h1, h2, h3, h4, h5, h6 {
            color: #ffffff !important;
            font-weight: 800 !important;
        }

        p, span, label, div {
            color: #f1f5f9;
        }

        /* Glass Cards */
        .glass-card {
            background: rgba(15, 23, 42, 0.85) !important;
            backdrop-filter: blur(12px) !important;
            border: 1.5px solid rgba(255, 255, 255, 0.12) !important;
            border-radius: 16px !important;
            padding: 1.5rem !important;
            margin-bottom: 1.25rem !important;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.3) !important;
        }

        .hero-banner {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.25) 0%, rgba(168, 85, 247, 0.25) 100%) !important;
            border: 1.5px solid rgba(168, 85, 247, 0.4) !important;
            border-radius: 20px !important;
            padding: 1.6rem 2rem !important;
            margin-bottom: 1.5rem !important;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3) !important;
        }

        /* THE CRITICAL WHITE INPUT BOX PATTERN (For high typing visibility) */
        div[data-testid="stTextArea"] textarea {
            background: #ffffff !important;
            color: #0f172a !important;
            font-size: 1.05rem !important;
            font-weight: 600 !important;
            border: 2px solid #38bdf8 !important;
            border-radius: 12px !important;
            padding: 14px 16px !important;
            box-shadow: 0 4px 14px rgba(0, 0, 0, 0.2) !important;
        }

        div[data-testid="stTextArea"] textarea:focus {
            border-color: #818cf8 !important;
            box-shadow: 0 0 0 3px rgba(56, 189, 248, 0.35) !important;
        }

        div[data-testid="stTextArea"] label p {
            color: #ffffff !important;
            font-weight: 700 !important;
            font-size: 1.05rem !important;
        }

        /* Dropdown / Selectbox White Box Pattern */
        div[data-baseweb="select"] > div {
            background: #ffffff !important;
            border: 1.5px solid #cbd5e1 !important;
            border-radius: 10px !important;
        }

        div[data-baseweb="select"] * {
            color: #0f172a !important;
            font-weight: 600 !important;
        }

        /* Action Buttons (Primary) */
        button[kind="primary"],
        div[data-testid="stButton"] > button[kind="primary"],
        div[data-testid="stButton"] > button[data-testid="baseButton-primary"],
        button[data-testid="baseButton-primary"] {
            background: linear-gradient(135deg, #ef4444 0%, #f43f5e 100%) !important;
            color: #ffffff !important;
            border: none !important;
            border-radius: 12px !important;
            padding: 12px 26px !important;
            font-weight: 800 !important;
            font-size: 1.05rem !important;
            box-shadow: 0 4px 20px rgba(239, 68, 68, 0.4) !important;
            transition: all 0.2s ease !important;
        }

        button[kind="primary"] *,
        button[data-testid="baseButton-primary"] *,
        button[kind="primary"] p,
        button[data-testid="baseButton-primary"] p {
            color: #ffffff !important;
            font-weight: 800 !important;
        }

        button[kind="primary"]:hover,
        button[data-testid="baseButton-primary"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(239, 68, 68, 0.6) !important;
        }

        /* Urgency Badges */
        .urgency-badge {
            display: inline-block;
            padding: 6px 18px;
            border-radius: 20px;
            font-weight: 800;
            font-size: 0.95rem;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        .badge-emergency {
            background: rgba(239, 68, 68, 0.25) !important;
            color: #f87171 !important;
            border: 1.5px solid #ef4444 !important;
        }

        .badge-doctor-soon {
            background: rgba(245, 158, 11, 0.25) !important;
            color: #fbbf24 !important;
            border: 1.5px solid #f59e0b !important;
        }

        .badge-monitor {
            background: rgba(56, 189, 248, 0.25) !important;
            color: #38bdf8 !important;
            border: 1.5px solid #0284c7 !important;
        }

        .badge-self-care {
            background: rgba(16, 185, 129, 0.25) !important;
            color: #34d399 !important;
            border: 1.5px solid #10b981 !important;
        }

        .symptom-tag {
            display: inline-block;
            background: rgba(56, 189, 248, 0.2);
            color: #7dd3fc;
            border: 1px solid #0284c7;
            border-radius: 20px;
            padding: 4px 14px;
            margin: 4px;
            font-size: 0.88rem;
            font-weight: 600;
        }

        .formula-box {
            background: rgba(15, 23, 42, 0.85) !important;
            border-left: 4px solid #38bdf8 !important;
            padding: 1rem 1.25rem !important;
            border-radius: 8px !important;
            margin: 0.75rem 0 !important;
            color: #e2e8f0 !important;
        }

        /* Radio Buttons Custom High-Contrast Styling */
        div[data-testid="stRadio"] {
            background: rgba(15, 23, 42, 0.92) !important;
            border: 1.5px solid rgba(56, 189, 248, 0.35) !important;
            border-radius: 16px !important;
            padding: 16px 20px !important;
            margin: 10px 0 18px 0 !important;
            box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3) !important;
        }

        div[data-testid="stRadio"] > label {
            color: #38bdf8 !important;
            font-size: 1.05rem !important;
            font-weight: 800 !important;
            margin-bottom: 10px !important;
            display: block !important;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] {
            gap: 8px !important;
            display: flex !important;
            flex-direction: column !important;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] label {
            background: rgba(30, 41, 59, 0.75) !important;
            border: 1px solid rgba(255, 255, 255, 0.12) !important;
            border-radius: 10px !important;
            padding: 10px 16px !important;
            transition: all 0.2s ease !important;
        }

        div[data-testid="stRadio"] div[role="radiogroup"] label:hover {
            background: rgba(56, 189, 248, 0.2) !important;
            border-color: #38bdf8 !important;
            transform: translateX(4px);
        }

        div[data-testid="stRadio"] div[role="radiogroup"] label p {
            color: #ffffff !important;
            font-weight: 700 !important;
            font-size: 0.98rem !important;
            opacity: 1 !important;
            margin: 0 !important;
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
    label_enc = joblib.load(label_path)
    features = joblib.load(feature_path)
    df = pd.read_csv(dataset_path)

    metrics = None
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

    return model, label_enc, features, df, metrics


try:
    xgb_model, label_encoder, feature_cols, dataset, metrics_data = load_assets()
except Exception as e:
    st.error(f"Error loading system models: {e}")
    st.stop()


def render_voice_input_widget():
    """
    Renders a high-reliability dual-engine Voice Input widget:
    1. Primary: Direct HTML5/JS Web Speech Recognition (Zero lag, live interim transcription in Kannada kn-IN or English)
    2. Fallback: Python Audio Streamlit Mic Recorder & Audio Note Uploader
    """
    st.markdown(
        """
        <div style="background:rgba(15,23,42,0.92);border:1.5px solid rgba(56,189,248,0.45);
                    border-radius:14px;padding:12px 18px;margin:4px 0 12px 0;
                    box-shadow:0 8px 24px rgba(0,0,0,0.35);">
            <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:8px;">
                <div style="font-weight:800;font-size:1.02rem;color:#38bdf8;display:flex;align-items:center;gap:8px;">
                    <span>🎙️ Live Kannada & English Voice Input</span>
                    <span style="font-size:0.75rem;background:rgba(56,189,248,0.2);color:#7dd3fc;padding:2px 10px;border-radius:12px;border:1px solid rgba(56,189,248,0.4);">
                        Google Speech Engine
                    </span>
                </div>
                <span style="font-size:0.82rem;color:#94a3b8;">Click mic below to speak symptoms in Kannada or English</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    v_c1, v_c2 = st.columns([1.3, 1.1])
    with v_c1:
        lang_choice = st.selectbox(
            "🌐 Spoken Language:",
            ["Kannada (ಕನ್ನಡ)", "English (India)", "English (US / Global)"],
            index=0,
            key="voice_lang_choice",
            help="Select the language you will speak in (Kannada or English) for optimal speech recognition."
        )
        lang_map = {
            "Kannada (ಕನ್ನಡ)": "kn-IN",
            "English (India)": "en-IN",
            "English (US / Global)": "en-US"
        }
        selected_lang = lang_map.get(lang_choice, "kn-IN")

    with v_c2:
        auto_predict = st.checkbox(
            "⚡ Auto-predict disease on speech",
            value=True,
            help="When checked, disease prediction will run automatically as soon as you finish speaking.",
            key="voice_auto_predict_toggle"
        )

    # --------------------------------------------------------------------
    # PRIMARY ENGINE: Direct Native Web Speech API Component (Chrome/Edge/Brave/Mobile)
    # --------------------------------------------------------------------
    lang_display = "Kannada (ಕನ್ನಡ)" if selected_lang == "kn-IN" else "English"
    web_mic_html = f"""
    <div style="margin:4px 0 10px 0;">
      <style>
        @keyframes pulse-ring {{
          0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0.7); }}
          70% {{ transform: scale(1.02); box-shadow: 0 0 0 12px rgba(239, 68, 68, 0); }}
          100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(239, 68, 68, 0); }}
        }}
        .mic-btn-idle {{
          background: linear-gradient(135deg, #0284c7 0%, #38bdf8 100%);
          color: #ffffff;
          border: none;
          border-radius: 12px;
          padding: 12px 22px;
          font-weight: 800;
          font-size: 15px;
          cursor: pointer;
          display: flex;
          align-items: center;
          justify-content: center;
          gap: 10px;
          width: 100%;
          box-shadow: 0 4px 18px rgba(56, 189, 248, 0.35);
          transition: all 0.2s ease;
        }}
        .mic-btn-idle:hover {{
          transform: translateY(-2px);
          box-shadow: 0 6px 22px rgba(56, 189, 248, 0.55);
        }}
        .mic-btn-recording {{
          background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%) !important;
          color: #ffffff !important;
          animation: pulse-ring 1.4s infinite cubic-bezier(0.4, 0, 0.6, 1);
        }}
      </style>

      <button id="web-mic-btn" class="mic-btn-idle" type="button">
        <span style="font-size:22px;">🎙️</span>
        <span id="mic-btn-label">Click to Speak Symptoms ({lang_display})</span>
      </button>

      <div id="mic-status-box" style="margin-top:8px;font-size:13px;color:#94a3b8;min-height:22px;padding:2px 4px;">
        Click button above and speak symptoms clearly in {lang_display}.
      </div>
      <div id="mic-live-transcript" style="display:none;margin-top:6px;background:rgba(30,41,59,0.9);border:1px solid #38bdf8;padding:8px 12px;border-radius:8px;font-size:14px;color:#7dd3fc;"></div>

      <script>
        (function() {{
          const btn = document.getElementById('web-mic-btn');
          const label = document.getElementById('mic-btn-label');
          const statusBox = document.getElementById('mic-status-box');
          const liveBox = document.getElementById('mic-live-transcript');
          const selectedLang = "{selected_lang}";
          const autoPred = {str(auto_predict).lower()};

          const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

          if (!SpeechRecognition) {{
            statusBox.innerHTML = '<span style="color:#fbbf24;">⚠️ Live browser speech is best supported in Google Chrome, Microsoft Edge, or Brave. You can also use the Audio Note Uploader below.</span>';
            return;
          }}

          let recognition = null;
          let isListening = false;
          let finalSpokenText = '';

          try {{
            recognition = new SpeechRecognition();
            recognition.continuous = false;
            recognition.interimResults = true;
            recognition.lang = selectedLang;
            recognition.maxAlternatives = 1;
          }} catch(e) {{
            statusBox.innerHTML = '<span style="color:#f87171;">Speech recognition initialization error.</span>';
            return;
          }}

          recognition.onstart = function() {{
            isListening = true;
            btn.classList.add('mic-btn-recording');
            label.innerText = '🔴 Listening... Click to Stop & Process';
            statusBox.innerHTML = '<span style="color:#34d399;font-weight:700;">🎙️ Listening in ' + (selectedLang === 'kn-IN' ? 'Kannada (ಕನ್ನಡ)' : 'English') + '... Speak symptoms clearly!</span>';
            if (liveBox) {{
              liveBox.style.display = 'none';
              liveBox.innerHTML = '';
            }}
          }};

          recognition.onresult = function(event) {{
            let interimTranscript = '';
            for (let i = event.resultIndex; i < event.results.length; ++i) {{
              if (event.results[i].isFinal) {{
                finalSpokenText += event.results[i][0].transcript;
              }} else {{
                interimTranscript += event.results[i][0].transcript;
              }}
            }}
            const displayText = finalSpokenText || interimTranscript;
            if (displayText && liveBox) {{
              liveBox.style.display = 'block';
              liveBox.innerHTML = '<b>Heard:</b> \"' + displayText + '\"';
            }}
          }};

          recognition.onerror = function(event) {{
            isListening = false;
            btn.classList.remove('mic-btn-recording');
            label.innerText = 'Click to Speak Symptoms (' + (selectedLang === 'kn-IN' ? 'Kannada' : 'English') + ')';
            
            if (event.error === 'not-allowed' || event.error === 'permission-denied') {{
              statusBox.innerHTML = '<span style="color:#f87171;font-weight:700;">🚫 Microphone access blocked. Please click the lock / camera-mic icon in your browser address bar and enable microphone permissions.</span>';
            }} else if (event.error === 'no-speech') {{
              statusBox.innerHTML = '<span style="color:#fbbf24;font-weight:600;">⚠️ No speech detected. Please click the button and speak into your microphone.</span>';
            }} else if (event.error === 'network') {{
              statusBox.innerHTML = '<span style="color:#fbbf24;">⚠️ Network issue with speech service. Please ensure internet is connected.</span>';
            }} else {{
              statusBox.innerHTML = '<span style="color:#f87171;">⚠️ Speech error: ' + event.error + '</span>';
            }}
          }};

          recognition.onend = function() {{
            isListening = false;
            btn.classList.remove('mic-btn-recording');
            label.innerText = 'Click to Speak Symptoms (' + (selectedLang === 'kn-IN' ? 'Kannada' : 'English') + ')';

            if (finalSpokenText && finalSpokenText.trim().length > 0) {{
              const captured = finalSpokenText.trim();
              statusBox.innerHTML = '<span style="color:#34d399;font-weight:700;">✅ Captured: \"' + captured + '\" — Transferring to symptoms & analyzing...</span>';
              
              // 1. Direct real-time DOM injection into Streamlit textarea
              try {{
                const parentDoc = window.parent.document;
                const textAreas = parentDoc.querySelectorAll('textarea');
                for (let i = 0; i < textAreas.length; i++) {{
                  const ta = textAreas[i];
                  const nativeSetter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value')?.set;
                  if (nativeSetter) {{
                    nativeSetter.call(ta, captured);
                  }} else {{
                    ta.value = captured;
                  }}
                  ta.dispatchEvent(new Event('input', {{ bubbles: true }}));
                  ta.dispatchEvent(new Event('change', {{ bubbles: true }}));
                }}

                // If auto-predict enabled, also click the predict button
                if (autoPred) {{
                  setTimeout(function() {{
                    const btns = parentDoc.querySelectorAll('button');
                    for (let b of btns) {{
                      if (b.innerText && b.innerText.includes('Predict Disease')) {{
                        b.click();
                        break;
                      }}
                    }}
                  }}, 500);
                }}
              }} catch(domErr) {{
                console.log('DOM sync notice:', domErr);
              }}

              // 2. Reliable URL parameter sync to guarantee Python state update
              setTimeout(function() {{
                try {{
                  const encoded = encodeURIComponent(captured);
                  window.parent.location.search = '?voice_query=' + encoded + '&voice_predict=' + (autoPred ? '1' : '0');
                }} catch(err) {{
                  console.error(err);
                }}
              }}, 400);
            }}
          }};

          btn.addEventListener('click', function() {{
            if (isListening) {{
              recognition.stop();
            }} else {{
              finalSpokenText = '';
              try {{
                recognition.lang = selectedLang;
                recognition.start();
              }} catch(err) {{
                try {{
                  recognition.stop();
                  setTimeout(function() {{ recognition.start(); }}, 100);
                }} catch(e) {{
                  statusBox.innerHTML = '<span style="color:#f87171;">Microphone is already active or busy.</span>';
                }}
              }}
            }}
          }});
        }})();
      </script>
    </div>
    """
    components.html(web_mic_html, height=130)

    # --------------------------------------------------------------------
    # FALLBACK / BACKUP: Python Audio Recorder & Audio Note Uploader
    # --------------------------------------------------------------------
    with st.expander("🛠️ Alternative Voice Recorder & Audio Note File Uploader (.wav, .mp3, .m4a)", expanded=False):
        st.caption("If your browser restricts live Web Speech, record raw audio below or upload a saved audio voice note:")
        raw_audio = mic_recorder(
            start_prompt="🎙️ Record Audio Clip",
            stop_prompt="⏹️ Stop & Process Audio Clip",
            just_once=False,
            use_container_width=True,
            format="wav",
            key=f"python_mic_recorder_{selected_lang}"
        )
        if raw_audio and "bytes" in raw_audio:
            with st.spinner("Processing audio through Google Speech Engine..."):
                try:
                    r = sr.Recognizer()
                    audio_data = sr.AudioData(raw_audio["bytes"], raw_audio["sample_rate"], raw_audio["sample_width"])
                    transcribed = r.recognize_google(audio_data, language=selected_lang)
                    if transcribed and transcribed.strip():
                        st.session_state["patient_symptoms_text_box"] = transcribed.strip()
                        st.session_state["input_text"] = transcribed.strip()
                        st.session_state["voice_banner_msg"] = transcribed.strip()
                        if auto_predict:
                            st.session_state["trigger_predict_now"] = True
                        st.success(f"✅ Transcribed: \"{transcribed}\"")
                        st.rerun()
                except sr.UnknownValueError:
                    st.warning("⚠️ Could not recognize speech from audio. Please speak louder or closer to the microphone.")
                except Exception as ex:
                    st.error(f"⚠️ Speech recognition error: {ex}")

        st.markdown("---")
        uploaded_audio = st.file_uploader(
            "Or upload an audio recording file of symptoms:",
            type=["wav", "mp3", "m4a", "ogg", "flac"],
            key="voice_audio_file_uploader"
        )
        if uploaded_audio is not None:
            if st.button("🎧 Transcribe & Predict Audio File", key="btn_transcribe_audio_file"):
                with st.spinner("Processing audio recording file..."):
                    try:
                        r = sr.Recognizer()
                        with sr.AudioFile(uploaded_audio) as source:
                            audio_data = r.record(source)
                            transcribed = r.recognize_google(audio_data, language=selected_lang)
                            if transcribed and transcribed.strip():
                                st.session_state["patient_symptoms_text_box"] = transcribed.strip()
                                st.session_state["input_text"] = transcribed.strip()
                                st.session_state["voice_banner_msg"] = transcribed.strip()
                                if auto_predict:
                                    st.session_state["trigger_predict_now"] = True
                                st.success(f"✅ Transcribed: \"{transcribed}\"")
                                st.rerun()
                    except sr.UnknownValueError:
                        st.warning("⚠️ Audio was unclear or no speech detected in the file.")
                    except Exception as ex:
                        st.error(f"⚠️ Audio file error: {ex}")


def render_tts_button(text_to_speak: str, lang: str = "en-IN", button_label: str = None, stop_label: str = None):
    """
    Multilingual AI Voice Assistant using browser SpeechSynthesis.
    Supports Kannada (kn-IN) and Indian English (en-IN) voices for clear, natural clinical readouts.
    """
    clean_text = text_to_speak.replace('*', '').replace('#', '').replace('_', '').replace('•', '').replace('⚠️', 'Warning: ').replace('🚨', 'Emergency alert: ')
    clean_text = clean_text.replace('"', '\\"').replace("\n", " ").replace("  ", " ").strip()
    
    if not button_label:
        button_label = "🎙️ Listen to Diagnosis & Care Plan (Voice Assistant)" if lang != "kn-IN" else "🎙️ ರೋಗ ನಿರ್ಣಯ ಮತ್ತು ಆರೈಕೆ ವರದಿಯನ್ನು ಆಲಿಸಿ (ಕನ್ನಡ ಧ್ವನಿ)"
    if not stop_label:
        stop_label = "⏹️ Speaking Voice Assistant... (Click to Stop)" if lang != "kn-IN" else "⏹️ ಕನ್ನಡದಲ್ಲಿ ಓದಲಾಗುತ್ತಿದೆ... (ನಿಲ್ಲಿಸಲು ಕ್ಲಿಕ್ ಮಾಡಿ)"

    tts_html = f"""
    <div style="margin:12px 0 16px 0;">
      <button id="tts-btn" style="background:rgba(56,189,248,0.18);color:#38bdf8;border:2px solid #38bdf8;padding:10px 22px;border-radius:26px;cursor:pointer;font-weight:700;font-size:14px;display:inline-flex;align-items:center;gap:10px;box-shadow:0 3px 12px rgba(56,189,248,0.25);transition:all 0.2s ease;">
        <span style="font-size:18px;">🎙️</span> <b>{button_label}</b>
      </button>
      <script>
        (function() {{
          let isSpeaking = false;
          const btn = document.getElementById('tts-btn');
          if (!btn) return;

          btn.addEventListener('click', () => {{
            if (!('speechSynthesis' in window)) {{
              alert('Speech synthesis is not supported in this browser.');
              return;
            }}

            if (window.speechSynthesis.speaking && isSpeaking) {{
              window.speechSynthesis.cancel();
              isSpeaking = false;
              btn.innerHTML = '<span style="font-size:18px;">🎙️</span> <b>{button_label}</b>';
              btn.style.background = 'rgba(56,189,248,0.18)';
              btn.style.borderColor = '#38bdf8';
              btn.style.color = '#38bdf8';
              return;
            }}

            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance("{clean_text}");
            utterance.rate = 0.90;
            utterance.pitch = 1.0;
            utterance.lang = '{lang}';

            // Find matching voice if available in browser
            const targetLang = '{lang}'.toLowerCase();
            const voices = window.speechSynthesis.getVoices();
            const matchedVoice = voices.find(v => 
              v.lang.toLowerCase() === targetLang ||
              v.lang.toLowerCase().startsWith(targetLang.split('-')[0]) ||
              (targetLang.includes('kn') && (v.name.toLowerCase().includes('kannada') || v.name.toLowerCase().includes('kn'))) ||
              (targetLang.includes('en') && (v.name.toLowerCase().includes('india') || v.name.toLowerCase().includes('heera') || v.name.toLowerCase().includes('ravi') || v.name.toLowerCase().includes('neerja')))
            );
            if (matchedVoice) {{
              utterance.voice = matchedVoice;
            }}

            utterance.onend = () => {{
              isSpeaking = false;
              btn.innerHTML = '<span style="font-size:18px;">🎙️</span> <b>{button_label}</b>';
              btn.style.background = 'rgba(56,189,248,0.18)';
              btn.style.borderColor = '#38bdf8';
              btn.style.color = '#38bdf8';
            }};

            utterance.onerror = () => {{
              isSpeaking = false;
              btn.innerHTML = '<span style="font-size:18px;">🎙️</span> <b>{button_label}</b>';
            }};

            isSpeaking = true;
            btn.innerHTML = '<span style="font-size:18px;">⏹️</span> <b>{stop_label}</b>';
            btn.style.background = 'rgba(239,68,68,0.25)';
            btn.style.borderColor = '#ef4444';
            btn.style.color = '#f87171';
            window.speechSynthesis.speak(utterance);
          }});
        }})();
      </script>
    </div>
    """
    components.html(tts_html, height=55)


# ---------------------------------------------------------
# Sidebar: Patient Details & Triage Inputs (Bilingual: Kannada & English)
# ---------------------------------------------------------
with st.sidebar:
    # Language Selector Header with Icon
    st.markdown(
        """
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:6px;">
            <span style="font-weight:800;color:#f8fafc;font-size:1.05rem;display:flex;align-items:center;gap:6px;">
                <span>🌐</span> <span>ಮಾಹಿತಿ ಭಾಷೆ / Language:</span>
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    if "sidebar_lang" not in st.session_state:
        st.session_state["sidebar_lang"] = "english"

    sb_lang_idx = 1 if st.session_state.get("sidebar_lang") == "kannada" else 0
    selected_sb_lang = st.radio(
        "Select Information Language:",
        ["English (EN)", "ಕನ್ನಡ (KN)"],
        index=sb_lang_idx,
        key="sidebar_lang_selector_radio",
        horizontal=True,
        label_visibility="collapsed"
    )
    is_sb_kn = "ಕನ್ನಡ" in selected_sb_lang or "KN" in selected_sb_lang
    st.session_state["sidebar_lang"] = "kannada" if is_sb_kn else "english"

    st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.1);margin:8px 0 12px 0;'>", unsafe_allow_html=True)

    # Initialize shared values in session state if not set
    if "patient_age_val" not in st.session_state:
        st.session_state["patient_age_val"] = 28
    if "patient_gender_val" not in st.session_state:
        st.session_state["patient_gender_val"] = "Male"
    if "patient_days_val" not in st.session_state:
        st.session_state["patient_days_val"] = 3
    if "patient_severity_val" not in st.session_state:
        st.session_state["patient_severity_val"] = 5
    if "patient_conditions_val" not in st.session_state:
        st.session_state["patient_conditions_val"] = ["None"]

    if is_sb_kn:
        st.markdown(
            """
            <div style="background:rgba(56,189,248,0.12);border:1px solid rgba(56,189,248,0.35);padding:10px 14px;border-radius:10px;margin-bottom:12px;">
                <div style="font-weight:800;color:#38bdf8;font-size:1.05rem;display:flex;align-items:center;gap:6px;">
                    <span>🧑‍⚕️</span> <span>ರೋಗಿಯ ಮಾಹಿತಿ (Patient Info)</span>
                </div>
                <div style="font-size:0.8rem;color:#cbd5e1;margin-top:4px;line-height:1.4;">
                    ನಿಖರ ರೋಗ ನಿರ್ಣಯ ಮತ್ತು ಆರೈಕೆಗಾಗಿ ರೋಗಿಯ ವಿವರಗಳನ್ನು ನಮೂದಿಸಿ
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        age = st.number_input(
            "🎂 ರೋಗಿಯ ವಯಸ್ಸು (Patient Age):",
            min_value=1,
            max_value=120,
            value=int(st.session_state["patient_age_val"]),
            step=1,
            key="sb_age_kn"
        )
        st.session_state["patient_age_val"] = age

        gender_options_kn = ["ಪುರುಷ (Male)", "ಮಹಿಳೆ (Female)", "ಇತರ (Other)", "ಹೇಳಲು ಇಷ್ಟವಿಲ್ಲ (Prefer not to say)"]
        curr_g = st.session_state["patient_gender_val"]
        g_idx = 0 if curr_g == "Male" else (1 if curr_g == "Female" else (2 if curr_g == "Other" else 3))
        gender_kn_choice = st.selectbox(
            "⚧️ ಲಿಂಗ (Gender):",
            gender_options_kn,
            index=g_idx,
            key="sb_gender_kn"
        )
        gender = normalize_patient_gender(gender_kn_choice)
        st.session_state["patient_gender_val"] = gender

        days = st.slider(
            "⏱️ ರೋಗಲಕ್ಷಣಗಳ ಅವಧಿ (ದಿನಗಳಲ್ಲಿ) - Duration (Days):",
            min_value=1,
            max_value=60,
            value=int(st.session_state["patient_days_val"]),
            key="sb_days_kn"
        )
        st.session_state["patient_days_val"] = days

        severity = st.slider(
            "⚡ ತೊಂದರೆಯ ತೀವ್ರತೆ (Discomfort Severity 1-10):",
            min_value=1,
            max_value=10,
            value=int(st.session_state["patient_severity_val"]),
            key="sb_sev_kn"
        )
        st.session_state["patient_severity_val"] = severity

        cond_options_kn = [
            "ಯಾವುದೂ ಇಲ್ಲ (None)",
            "ಮಧುಮೇಹ / ಶುಗರ್ (Diabetes)",
            "ರಕ್ತದೊತ್ತಡ / ಬಿಪಿ (Hypertension)",
            "ಉಬ್ಬಸ / ಅಸ್ತಮಾ (Asthma)",
            "ಹೃದ್ರೋಗ (Heart Disease)",
            "ಮೂತ್ರಪಿಂಡ ಕಾಯಿಲೆ (Kidney Disease)",
            "ಥೈರಾಯ್ಡ್ ಸಮಸ್ಯೆ (Thyroid Disorder)"
        ]
        curr_conds = st.session_state.get("patient_conditions_val", ["None"])
        kn_defaults = [CONDITIONS_MAP_KN.get(c, "ಯಾವುದೂ ಇಲ್ಲ (None)") for c in curr_conds if c in CONDITIONS_MAP_KN] or ["ಯಾವುದೂ ಇಲ್ಲ (None)"]

        selected_conds_kn = st.multiselect(
            "🩺 ಮೊದಲೇ ಇರುವ ಆರೋಗ್ಯ ಸಮಸ್ಯೆಗಳು (Pre-existing Conditions):",
            cond_options_kn,
            default=kn_defaults,
            key="sb_conds_kn"
        )
        existing_cond = normalize_patient_conditions(selected_conds_kn)
        st.session_state["patient_conditions_val"] = existing_cond

        # Dynamic Alert inside sidebar if symptom duration >= 5 days
        if days >= 5:
            st.error(f"🚨 **ದೀರ್ಘಕಾಲೀನ ಎಚ್ಚರಿಕೆ:** ರೋಗಲಕ್ಷಣಗಳು **{days} ದಿನಗಳಿಂದ** ಮುಂದುವರಿದಿವೆ (5 ದಿನಗಳ ಮಿತಿ ಮೀರಿದೆ). ತಕ್ಷಣ ವೈದ್ಯರನ್ನು ಭೇಟಿ ಮಾಡಿ.")

        st.markdown("---")
        st.markdown("### 🤖 ಎಐ ಮಾದರಿ ಮಾಹಿತಿ (Model Architecture)")
        st.markdown(
            """
            <div style="background:rgba(15,23,42,0.95);border:1.5px solid #6366f1;border-radius:12px;padding:14px;box-shadow:0 4px 15px rgba(0,0,0,0.4);">
                <div style="font-weight:800;color:#38bdf8;font-size:1rem;display:flex;align-items:center;gap:6px;">
                    <span>⚡</span> <span>XGBoost ಕ್ಲಿನಿಕಲ್ ಎಐ ಮಾದರಿ</span>
                </div>
                <div style="font-size:0.85rem;color:#e2e8f0;margin-top:6px;line-height:1.4;">
                    ಮಲ್ಟಿ-ಕ್ಲಾಸ್ ಸಂಭವನೀಯತೆಯೊಂದಿಗೆ ಅತ್ಯಾಧುನಿಕ ಗ್ರೇಡಿಯೆಂಟ್ ಬೂಸ್ಟಿಂಗ್
                </div>
                <div style="margin-top:10px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.12);font-size:0.88rem;color:#34d399;font-weight:700;line-height:1.5;">
                    ✓ ಪರೀಕ್ಷಾ ನಿಖರತೆ: 99.86%<br>
                    ✓ 5-ಹಂತದ CV: 99.78% (±0.11%)
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("---")
        st.caption("🔒 ಶೈಕ್ಷಣಿಕ ಕ್ಲಿನಿಕಲ್ ನಿರ್ಧಾರ ಬೆಂಬಲ ವ್ಯವಸ್ಥೆ • ವೇಗ ಮತ್ತು ಸುರಕ್ಷಿತ")

    else:
        st.markdown("### 🧑‍⚕️ Patient Information")
        age = st.number_input(
            "🎂 Patient Age:",
            min_value=1,
            max_value=120,
            value=int(st.session_state["patient_age_val"]),
            step=1,
            key="sb_age_en"
        )
        st.session_state["patient_age_val"] = age

        gender_options_en = ["Male", "Female", "Other", "Prefer not to say"]
        curr_g = st.session_state["patient_gender_val"]
        g_idx = 0 if curr_g == "Male" else (1 if curr_g == "Female" else (2 if curr_g == "Other" else 3))
        gender_en_choice = st.selectbox(
            "⚧️ Gender:",
            gender_options_en,
            index=g_idx,
            key="sb_gender_en"
        )
        gender = normalize_patient_gender(gender_en_choice)
        st.session_state["patient_gender_val"] = gender

        days = st.slider(
            "⏱️ Duration of Symptoms (Days):",
            min_value=1,
            max_value=60,
            value=int(st.session_state["patient_days_val"]),
            key="sb_days_en"
        )
        st.session_state["patient_days_val"] = days

        severity = st.slider(
            "⚡ Discomfort Severity (1-10):",
            min_value=1,
            max_value=10,
            value=int(st.session_state["patient_severity_val"]),
            key="sb_sev_en"
        )
        st.session_state["patient_severity_val"] = severity

        cond_options_en = ["None", "Diabetes", "Hypertension", "Asthma", "Heart Disease", "Kidney Disease", "Thyroid Disorder"]
        curr_conds = st.session_state.get("patient_conditions_val", ["None"])
        en_defaults = [c for c in curr_conds if c in cond_options_en] or ["None"]

        selected_conds_en = st.multiselect(
            "🩺 Pre-existing Conditions:",
            cond_options_en,
            default=en_defaults,
            key="sb_conds_en"
        )
        existing_cond = normalize_patient_conditions(selected_conds_en)
        st.session_state["patient_conditions_val"] = existing_cond

        # Dynamic Alert inside sidebar if symptom duration >= 5 days
        if days >= 5:
            st.warning(f"⚠️ **Prolonged Duration Notice:** Symptoms persisting for **{days} days** (>= 5-6 days) require in-person medical attention.")

        st.markdown("---")
        st.markdown("### 🤖 Model Architecture")
        st.markdown(
            """
            <div style="background:rgba(15,23,42,0.95);border:1.5px solid #6366f1;border-radius:12px;padding:14px;box-shadow:0 4px 15px rgba(0,0,0,0.4);">
                <div style="font-weight:800;color:#38bdf8;font-size:1rem;display:flex;align-items:center;gap:6px;">
                    <span>⚡</span> <span>XGBoost Classifier</span>
                </div>
                <div style="font-size:0.85rem;color:#e2e8f0;margin-top:6px;line-height:1.4;">
                    Extreme Gradient Boosting with Softmax Multi-Class Probability
                </div>
                <div style="margin-top:10px;padding-top:8px;border-top:1px solid rgba(255,255,255,0.12);font-size:0.88rem;color:#34d399;font-weight:700;line-height:1.5;">
                    ✓ Test Accuracy: 99.86%<br>
                    ✓ 5-Fold CV: 99.78% (±0.11%)
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("---")
        st.caption("🔒 Educational Clinical Decision Support • Fast & Scalable")


# ---------------------------------------------------------
# Page State Manager (Home vs. Dedicated Results Report)
# ---------------------------------------------------------
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "home"


# =========================================================
# ROUTE A: DEDICATED FULL-PAGE DIAGNOSTIC REPORT (Page 2)
# =========================================================
if st.session_state.get("current_page") == "results" and "report_data" in st.session_state:
    p_data = st.session_state.get("report_data", {})
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

    duration_days = snap.get("days", 3)
    sev_num = snap.get("severity", 5)
    active_conds_str = ", ".join([c for c in snap.get("conditions", []) if c != "None"]) or "None (Healthy Baseline)"
    active_conds_str_kn = get_kannada_conditions_display(snap.get("conditions", []))
    gender_kn = get_kannada_gender_display(snap.get("gender", "Male"))

    # Top Navigation Bar
    col_nav1, col_nav2 = st.columns([1.2, 2.8])
    with col_nav1:
        if st.button("⬅️ Back to Symptom Input", type="primary", use_container_width=True, key="btn_top_back_home"):
            st.session_state["current_page"] = "home"
            st.rerun()

    with col_nav2:
        st.markdown(
            """
            <div style="display:flex;align-items:center;justify-content:flex-end;height:100%;gap:10px;padding-top:4px;">
                <span style="background:rgba(16,185,129,0.15);color:#34d399;border:1.5px solid #10b981;font-weight:700;font-size:0.9rem;padding:6px 16px;border-radius:20px;">
                    ✓ AI Diagnosis Verified (99.86% XGBoost Engine)
                </span>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Hero Banner
    st.markdown(
        """
        <div class="hero-banner" style="margin-top:10px;padding:1.4rem 1.8rem;">
            <div style="display:flex;align-items:center;gap:14px;">
                <span style="font-size:2.2rem;">📊</span>
                <div>
                    <h2 style="margin:0;font-size:1.85rem;font-weight:800;background:linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                        Clinical Disease Assessment Results
                    </h2>
                    <p style="margin:4px 0 0 0;color:#cbd5e1;font-size:0.95rem;font-weight:500;">
                        Choose between the <b>Quick Medical Summary Report</b> or the <b>Detailed Comprehensive Medical Report</b> below.
                    </p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Retrieve or compile Kannada report payload
    is_kannada_input = p_data.get("is_kannada_input", False)
    kn_data = p_data.get("kn_report_data")
    if not kn_data:
        kn_data = get_kannada_report_data(
            top_disease,
            matched_keys,
            urgency_level,
            days=int(duration_days),
            severity=int(sev_num),
            vuln_score=int(vuln_score)
        )

    # Language Switcher Bar (Supports 1-click Kannada / English toggle)
    col_lang_info, col_lang_toggle = st.columns([2.2, 1.3])
    with col_lang_info:
        if is_kannada_input:
            st.markdown(
                """
                <div style="display:inline-flex;align-items:center;gap:8px;background:rgba(56,189,248,0.12);border:1px solid rgba(56,189,248,0.35);padding:6px 14px;border-radius:18px;margin-top:6px;">
                    <span style="font-size:1.05rem;">🌐</span>
                    <span style="color:#7dd3fc;font-size:0.88rem;font-weight:700;">ಕನ್ನಡ ಇನ್‌ಪುಟ್ ಪತ್ತೆಯಾಗಿದೆ • ವರದಿ ಕನ್ನಡದಲ್ಲಿದೆ (Kannada Input Active)</span>
                </div>
                """,
                unsafe_allow_html=True
            )
    with col_lang_toggle:
        curr_choice = st.session_state.get("report_lang_choice", "kannada" if is_kannada_input else "english")
        lang_idx = 0 if curr_choice == "kannada" else 1
        selected_lang_choice = st.radio(
            "🌐 Report Language / ವರದಿ ಭಾಷೆ:",
            ["ಕನ್ನಡ (Kannada)", "English"],
            index=lang_idx,
            key="report_lang_choice_radio",
            horizontal=True
        )
        is_kannada_active = "Kannada" in selected_lang_choice
        st.session_state["report_lang_choice"] = "kannada" if is_kannada_active else "english"

    # Prominent Two-Sheet Selection Header
    st.markdown("<br>", unsafe_allow_html=True)
    report_format_options = [
        "📄 Sheet 1: Quick Medical Summary Report (Major Disease, Symptoms & Doctor vs. Rest Verdict)" if not is_kannada_active else "📄 ಶೀಟ್ 1: ತ್ವರಿತ ವೈದ್ಯಕೀಯ ಸಾರಾಂಶ ವರದಿ (ಕಾಯಿಲೆ, ರೋಗಲಕ್ಷಣಗಳು & ವೈದ್ಯರ ಭೇಟಿ/ವಿಶ್ರಾಂತಿ ತೀರ್ಪು)",
        "📊 Sheet 2: Detailed Medical Report & Full Clinical Description (Comprehensive Diagnostics, Differential & Care Package)" if not is_kannada_active else "📊 ಶೀಟ್ 2: ವಿವರವಾದ ವೈದ್ಯಕೀಯ ವರದಿ ಮತ್ತು ಪೂರ್ಣ ವಿವರಣೆ (Detailed Report)"
    ]

    selected_format_idx = 1 if st.session_state.get("active_report_tab") == "detailed" else 0

    selected_report_mode = st.radio(
        "Select Report View / Sheet Format:" if not is_kannada_active else "ವರದಿ ಸ್ವರೂಪವನ್ನು ಆಯ್ಕೆಮಾಡಿ:",
        report_format_options,
        index=selected_format_idx,
        key="report_view_mode_radio",
        horizontal=False
    )

    is_quick_mode = ("Sheet 1" in selected_report_mode) or ("ಶೀಟ್ 1" in selected_report_mode)

    # Urgency Badge & Action Styling
    badge_class_map = {
        "EMERGENCY": "badge-emergency",
        "See Doctor Immediately": "badge-emergency",
        "SEE DOCTOR IMMEDIATELY": "badge-emergency",
        "See Doctor Soon": "badge-doctor-soon",
        "CONSULT DOCTOR SOON": "badge-doctor-soon",
        "Monitor 2-3 Days": "badge-monitor",
        "MONITOR 2-3 DAYS": "badge-monitor",
        "Self-Care": "badge-self-care",
        "SELF-CARE & HOME REMEDIES": "badge-self-care"
    }
    badge_class = badge_class_map.get(urgency_level, "badge-monitor")

    # Determine Doctor vs Rest Verdict
    requires_doctor = (duration_days >= 5) or ("EMERGENCY" in urgency_level.upper()) or ("SEE DOCTOR" in urgency_level.upper()) or ("CONSULT DOCTOR" in urgency_level.upper())

    # =========================================================================
    # SHEET 1: QUICK MEDICAL SUMMARY REPORT (Concise 1-Page Summary)
    # =========================================================================
    if is_quick_mode:
        st.session_state["active_report_tab"] = "quick"
        st.markdown("<hr style='border:none;border-top:1.5px solid rgba(56,189,248,0.3);margin:16px 0 20px 0;'>", unsafe_allow_html=True)

        if is_kannada_active:
            # -------------------------------------------------------------
            # KANNADA QUICK MEDICAL SUMMARY REPORT
            # -------------------------------------------------------------
            st.markdown(
                """
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;">
                    <h3 style="margin:0;color:#38bdf8;font-size:1.4rem;font-weight:800;display:flex;align-items:center;gap:10px;">
                        <span>📄</span> <span>ಶೀಟ್ 1: ತ್ವರಿತ ವೈದ್ಯಕೀಯ ಸಾರಾಂಶ ವರದಿ (Quick Medical Report)</span>
                    </h3>
                    <span style="font-size:0.85rem;background:rgba(56,189,248,0.15);color:#7dd3fc;padding:4px 12px;border-radius:12px;border:1px solid rgba(56,189,248,0.4);">
                        ಕನ್ನಡ ವೈದ್ಯಕೀಯ ಸಾರಾಂಶ ನೋಟ
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

            # 1. Main Predicted Disease & Symptoms Grid in Kannada
            col_q1, col_q2 = st.columns([1.6, 1.4])

            with col_q1:
                st.markdown(
                    f"""
                    <div class="glass-card" style="border:2px solid #38bdf8;box-shadow:0 0 25px rgba(56,189,248,0.2);padding:22px;">
                        <div style="font-size:0.85rem;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;">
                            🩺 ಪ್ರಮುಖ ಗುರುತಿಸಲಾದ ಕಾಯಿಲೆ (Predicted Major Disease)
                        </div>
                        <div style="font-size:1.85rem;font-weight:800;color:#f8fafc;margin:8px 0 10px 0;line-height:1.3;">
                            {kn_data['display_name_kn']}
                        </div>
                        <div style="margin-bottom:14px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
                            <span class="urgency-badge {badge_class}">{kn_data['urgency_kn']}</span>
                            <span style="background:rgba(56,189,248,0.2);color:#38bdf8;padding:4px 12px;border-radius:14px;font-weight:700;font-size:0.9rem;">
                                {top_confidence:.1f}% ಎಐ ಖಚಿತತೆ (AI Confidence)
                            </span>
                        </div>
                        <div style="color:#cbd5e1;font-size:0.95rem;line-height:1.6;margin-bottom:16px;">
                            {kn_data['description_kn']}
                        </div>
                        <div style="background:rgba(15,23,42,0.9);border-radius:10px;padding:12px 16px;border:1px solid rgba(56,189,248,0.3);">
                            <span style="font-size:0.85rem;color:#94a3b8;">ಸಲಹೆ ನೀಡಲಾದ ತಜ್ಞ ವೈದ್ಯರು (Recommended Specialist):</span><br>
                            <span style="font-size:1.1rem;font-weight:800;color:#38bdf8;">🧑‍⚕️ {kn_data['specialist_kn']}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col_q2:
                st.markdown(
                    f"""
                    <div class="glass-card" style="padding:22px;height:100%;">
                        <div style="font-size:1.15rem;font-weight:700;color:#f8fafc;margin-bottom:12px;display:flex;align-items:center;gap:8px;">
                            <span>📋</span> <span>ಗುರುತಿಸಲಾದ ರೋಗಲಕ್ಷಣಗಳು ({len(matched_keys)} Symptoms)</span>
                        </div>
                        <div style="margin-bottom:16px;">
                            {''.join([f'<span class="symptom-tag" style="margin-bottom:6px;display:inline-block;">✓ {s}</span>' for s in kn_data['translated_symptoms']])}
                        </div>
                        <div style="background:rgba(15,23,42,0.8);border-left:4px solid #38bdf8;padding:12px 16px;border-radius:8px;font-size:0.9rem;color:#cbd5e1;line-height:1.5;">
                            <span style="font-size:0.82rem;color:#94a3b8;font-weight:600;">ರೋಗಿಯ ವಿವರ (Patient Context):</span><br>
                            <b>ವಯಸ್ಸು:</b> {snap.get('age', 28)} ವರ್ಷ | <b>ಲಿಂಗ:</b> {gender_kn} | <b>ಅವಧಿ:</b> {duration_days} ದಿನಗಳು | <b>ತೀವ್ರತೆ:</b> {sev_num}/10<br>
                            <b>ಇತರ ಆರೋಗ್ಯ ಸಮಸ್ಯೆ:</b> {active_conds_str_kn}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # 2. PROMINENT CLINICAL VERDICT: WHETHER TO MEET DOCTOR OR TAKE REST (KANNADA)
            st.markdown("<br>", unsafe_allow_html=True)
            if kn_data['requires_doctor']:
                st.markdown(
                    f"""
                    <div class="glass-card" style="border:2px solid #ef4444;background:linear-gradient(135deg, rgba(239,68,68,0.18), rgba(15,23,42,0.95));box-shadow:0 0 30px rgba(239,68,68,0.25);padding:24px;">
                        <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
                            <span style="font-size:2rem;">🏥</span>
                            <div>
                                <h3 style="margin:0;color:#f87171;font-size:1.35rem;font-weight:800;">
                                    {kn_data['verdict_headline_kn']}
                                </h3>
                                <span style="font-size:0.88rem;color:#fca5a5;font-weight:600;">
                                    ವೈದ್ಯರ ನೇರ ತಪಾಸಣೆ ಅತ್ಯಗತ್ಯ (In-Person Medical Evaluation Needed)
                                </span>
                            </div>
                        </div>
                        <div style="font-size:1rem;color:#f1f5f9;line-height:1.6;margin-bottom:16px;">
                            <b>ವೈದ್ಯಕೀಯ ವಿವರಣೆ:</b> {kn_data['verdict_reason_kn']}
                        </div>
                        <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(240px, 1fr));gap:14px;">
                            <div style="background:rgba(0,0,0,0.4);padding:14px;border-radius:10px;border:1px solid rgba(239,68,68,0.4);">
                                <b style="color:#f87171;">🧑‍⚕️ ಭೇಟಿ ನೀಡಬೇಕಾದ ತಜ್ಞರು:</b><br>
                                <span style="color:#f8fafc;font-size:0.95rem;">{kn_data['specialist_kn']}</span>
                            </div>
                            <div style="background:rgba(0,0,0,0.4);padding:14px;border-radius:10px;border:1px solid rgba(239,68,68,0.4);">
                                <b style="color:#fbbf24;">⚡ ತಕ್ಷಣದ ಕ್ರಮ:</b><br>
                                <span style="color:#f8fafc;font-size:0.95rem;">{kn_data['immediate_action_kn']}</span>
                            </div>
                            <div style="background:rgba(0,0,0,0.4);padding:14px;border-radius:10px;border:1px solid rgba(239,68,68,0.4);">
                                <b style="color:#38bdf8;">🛡️ ಪ್ರಮುಖ ಸುರಕ್ಷತಾ ನಿಯಮ:</b><br>
                                <span style="color:#f8fafc;font-size:0.95rem;">{kn_data['precautions_kn']}</span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="glass-card" style="border:2px solid #10b981;background:linear-gradient(135deg, rgba(16,185,129,0.18), rgba(15,23,42,0.95));box-shadow:0 0 30px rgba(16,185,129,0.25);padding:24px;">
                        <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
                            <span style="font-size:2rem;">🏡</span>
                            <div>
                                <h3 style="margin:0;color:#34d399;font-size:1.35rem;font-weight:800;">
                                    {kn_data['verdict_headline_kn']}
                                </h3>
                                <span style="font-size:0.88rem;color:#6ee7b7;font-weight:600;">
                                    ಸೌಮ್ಯ ಹಂತ • ಮನೆಯಲ್ಲೇ ಸೂಕ್ತ ಆರೈಕೆ ಮತ್ತು ವಿಶ್ರಾಂತಿ ಸಾಕು
                                </span>
                            </div>
                        </div>
                        <div style="font-size:1rem;color:#f1f5f9;line-height:1.6;margin-bottom:16px;">
                            <b>ಆರೈಕೆ ವಿವರಣೆ:</b> {kn_data['verdict_reason_kn']}
                        </div>
                        <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(240px, 1fr));gap:14px;">
                            <div style="background:rgba(0,0,0,0.4);padding:14px;border-radius:10px;border:1px solid rgba(16,185,129,0.4);">
                                <b style="color:#34d399;">🛌 ವಿಶ್ರಾಂತಿ ಕ್ರಮ:</b><br>
                                <span style="color:#f8fafc;font-size:0.95rem;">8-9 ಗಂಟೆಗಳ ಕಾಲ ಚೆನ್ನಾಗಿ ನಿದ್ರೆ ಮಾಡಿ; ಅನಗತ್ಯ ದೈಹಿಕ ಶ್ರಮ ತಪ್ಪಿಸಿ.</span>
                            </div>
                            <div style="background:rgba(0,0,0,0.4);padding:14px;border-radius:10px;border:1px solid rgba(16,185,129,0.4);">
                                <b style="color:#38bdf8;">💧 ನೀರು ಮತ್ತು ಆಹಾರ:</b><br>
                                <span style="color:#f8fafc;font-size:0.95rem;">{kn_data['diet_kn']}</span>
                            </div>
                            <div style="background:rgba(0,0,0,0.4);padding:14px;border-radius:10px;border:1px solid rgba(16,185,129,0.4);">
                                <b style="color:#fbbf24;">⏱️ ಗಮನಿಸಬೇಕಾದ ಸಮಯ:</b><br>
                                <span style="color:#f8fafc;font-size:0.95rem;">2-3 ದಿನಗಳಲ್ಲಿ ಗುಣವಾಗದಿದ್ದರೆ ಅಥವಾ 5 ದಿನ ಮೀರಿದರೆ ವೈದ್ಯರನ್ನು ಭೇಟಿ ಮಾಡಿ.</span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # 3. Quick Home Care & Remedies Card in Kannada
            st.markdown("<br>", unsafe_allow_html=True)
            remedies_html = "".join([f"<li style='margin-bottom:8px;'>{r}</li>" for r in kn_data['home_remedies_kn']])
            st.markdown(
                f"""
                <div class="glass-card" style="border:1.5px solid #38bdf8;padding:20px;">
                    <div style="font-size:1.15rem;font-weight:700;color:#38bdf8;margin-bottom:12px;display:flex;align-items:center;gap:8px;">
                        <span>🌿</span> <span>ಮನೆ ಮದ್ದು & ತ್ವರಿತ ಆರೈಕೆ ಸಲಹೆಗಳು (Quick Home Care Guidance)</span>
                    </div>
                    <ul style="color:#e2e8f0;font-size:0.95rem;line-height:1.6;margin:0 0 12px 20px;padding:0;">
                        {remedies_html}
                    </ul>
                    <div style="font-size:0.88rem;color:#94a3b8;border-top:1px solid rgba(255,255,255,0.1);padding-top:10px;">
                        <b>ಆಹಾರ ನಿಯಮ:</b> {kn_data['diet_kn']}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # 4. Audio Voice Readout in Kannada
            render_tts_button(
                kn_data['tts_text_kn'],
                lang="kn-IN",
                button_label="🎙️ ರೋಗ ನಿರ್ಣಯ ಮತ್ತು ಆರೈಕೆ ವರದಿಯನ್ನು ಆಲಿಸಿ (ಕನ್ನಡ ಧ್ವನಿ)",
                stop_label="⏹️ ಕನ್ನಡದಲ್ಲಿ ಓದಲಾಗುತ್ತಿದೆ... (ನಿಲ್ಲಿಸಲು ಕ್ಲಿಕ್ ಮಾಡಿ)"
            )

        else:
            # -------------------------------------------------------------
            # ENGLISH QUICK MEDICAL SUMMARY REPORT
            # -------------------------------------------------------------
            st.markdown(
                """
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;">
                    <h3 style="margin:0;color:#38bdf8;font-size:1.4rem;font-weight:800;display:flex;align-items:center;gap:10px;">
                        <span>📄</span> <span>Sheet 1: Quick Medical Summary Report</span>
                    </h3>
                    <span style="font-size:0.85rem;background:rgba(56,189,248,0.15);color:#7dd3fc;padding:4px 12px;border-radius:12px;border:1px solid rgba(56,189,248,0.4);">
                        Doctor & Patient Summary View
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

            # 1. Main Predicted Disease & Symptoms Grid
            col_q1, col_q2 = st.columns([1.6, 1.4])

            with col_q1:
                st.markdown(
                    f"""
                    <div class="glass-card" style="border:2px solid #38bdf8;box-shadow:0 0 25px rgba(56,189,248,0.2);padding:22px;">
                        <div style="font-size:0.85rem;color:#94a3b8;font-weight:700;text-transform:uppercase;letter-spacing:0.5px;">
                            🩺 Major Predicted Disease
                        </div>
                        <div style="font-size:1.95rem;font-weight:800;color:#f8fafc;margin:8px 0 10px 0;">
                            {remedy_data.get('display_name', top_disease)}
                        </div>
                        <div style="margin-bottom:14px;display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
                            <span class="urgency-badge {badge_class}">{urgency_level}</span>
                            <span style="background:rgba(56,189,248,0.2);color:#38bdf8;padding:4px 12px;border-radius:14px;font-weight:700;font-size:0.9rem;">
                                {top_confidence:.1f}% AI Confidence
                            </span>
                        </div>
                        <div style="color:#cbd5e1;font-size:0.95rem;line-height:1.55;margin-bottom:16px;">
                            {remedy_data.get('description', 'Clinical analysis based on matched symptom profile and Bayesian risk weighting.')}
                        </div>
                        <div style="background:rgba(15,23,42,0.9);border-radius:10px;padding:12px 16px;border:1px solid rgba(56,189,248,0.3);">
                            <span style="font-size:0.85rem;color:#94a3b8;">Recommended Specialist:</span><br>
                            <span style="font-size:1.1rem;font-weight:800;color:#38bdf8;">🧑‍⚕️ {remedy_data.get('specialist', 'General Physician')}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col_q2:
                st.markdown(
                    f"""
                    <div class="glass-card" style="padding:22px;height:100%;">
                        <div style="font-size:1.15rem;font-weight:700;color:#f8fafc;margin-bottom:12px;display:flex;align-items:center;gap:8px;">
                            <span>📋</span> <span>Identified Patient Symptoms ({len(matched_keys)})</span>
                        </div>
                        <div style="margin-bottom:16px;">
                            {''.join([f'<span class="symptom-tag" style="margin-bottom:6px;display:inline-block;">✓ {s.replace("_", " ").title()}</span>' for s in matched_keys])}
                        </div>
                        <div style="background:rgba(15,23,42,0.8);border-left:4px solid #38bdf8;padding:12px 16px;border-radius:8px;font-size:0.9rem;color:#cbd5e1;line-height:1.5;">
                            <span style="font-size:0.82rem;color:#94a3b8;font-weight:600;">Patient Context:</span><br>
                            <b>Age:</b> {snap.get('age', 28)} yrs | <b>Duration:</b> {duration_days} Days | <b>Severity:</b> {sev_num}/10<br>
                            <b>Conditions:</b> {active_conds_str}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # 2. PROMINENT CLINICAL VERDICT: WHETHER TO MEET DOCTOR OR TAKE REST
            st.markdown("<br>", unsafe_allow_html=True)
            if requires_doctor:
                if duration_days >= 5:
                    doc_headline = "🚨 PROMPT MEDICAL ATTENTION RECOMMENDED (MEET DOCTOR)"
                    doc_reason = f"Patient symptoms have persisted for <b>{duration_days} days</b> (exceeding the standard 5-6 day clinical threshold) which indicates non-resolving or progressing pathology."
                else:
                    doc_headline = "🚨 MEDICAL CONSULTATION RECOMMENDED (MEET DOCTOR)"
                    doc_reason = f"Clinical risk triage identified significant symptom severity (Level {sev_num}/10) or diagnostic indicators for {remedy_data.get('display_name', top_disease)}."

                st.markdown(
                    f"""
                    <div class="glass-card" style="border:2px solid #ef4444;background:linear-gradient(135deg, rgba(239,68,68,0.18), rgba(15,23,42,0.95));box-shadow:0 0 30px rgba(239,68,68,0.25);padding:24px;">
                        <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
                            <span style="font-size:2rem;">🏥</span>
                            <div>
                                <h3 style="margin:0;color:#f87171;font-size:1.35rem;font-weight:800;">
                                    {doc_headline}
                                </h3>
                                <span style="font-size:0.88rem;color:#fca5a5;font-weight:600;">
                                    In-Person Clinical Assessment Advised
                                </span>
                            </div>
                        </div>
                        <div style="font-size:1rem;color:#f1f5f9;line-height:1.6;margin-bottom:16px;">
                            <b>Clinical Justification:</b> {doc_reason}
                        </div>
                        <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(240px, 1fr));gap:14px;">
                            <div style="background:rgba(0,0,0,0.4);padding:14px;border-radius:10px;border:1px solid rgba(239,68,68,0.4);">
                                <b style="color:#f87171;">🧑‍⚕️ Specialist to Visit:</b><br>
                                <span style="color:#f8fafc;font-size:0.95rem;">{remedy_data.get('specialist', 'General Physician')}</span>
                            </div>
                            <div style="background:rgba(0,0,0,0.4);padding:14px;border-radius:10px;border:1px solid rgba(239,68,68,0.4);">
                                <b style="color:#fbbf24;">⚡ Immediate Action:</b><br>
                                <span style="color:#f8fafc;font-size:0.95rem;">{action_advice}</span>
                            </div>
                            <div style="background:rgba(0,0,0,0.4);padding:14px;border-radius:10px;border:1px solid rgba(239,68,68,0.4);">
                                <b style="color:#38bdf8;">🛡️ Key Safety Rule:</b><br>
                                <span style="color:#f8fafc;font-size:0.95rem;">Avoid unprescribed self-medication; keep medical history and symptom timeline ready.</span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="glass-card" style="border:2px solid #10b981;background:linear-gradient(135deg, rgba(16,185,129,0.18), rgba(15,23,42,0.95));box-shadow:0 0 30px rgba(16,185,129,0.25);padding:24px;">
                        <div style="display:flex;align-items:center;gap:12px;margin-bottom:12px;">
                            <span style="font-size:2rem;">🏡</span>
                            <div>
                                <h3 style="margin:0;color:#34d399;font-size:1.35rem;font-weight:800;">
                                    🏡 HOME REST & SUPPORTIVE CARE RECOMMENDED (TAKE REST)
                                </h3>
                                <span style="font-size:0.88rem;color:#6ee7b7;font-weight:600;">
                                    Acute Mild Profile • Supportive Recovery at Home
                                </span>
                            </div>
                        </div>
                        <div style="font-size:1rem;color:#f1f5f9;line-height:1.6;margin-bottom:16px;">
                            <b>Clinical Justification:</b> Symptoms are acute ({duration_days} days) with mild severity ({sev_num}/10) and low vulnerability score ({vuln_score}/100). No emergency red flags detected.
                        </div>
                        <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(240px, 1fr));gap:14px;">
                            <div style="background:rgba(0,0,0,0.4);padding:14px;border-radius:10px;border:1px solid rgba(16,185,129,0.4);">
                                <b style="color:#34d399;">🛌 Rest & Sleep Protocol:</b><br>
                                <span style="color:#f8fafc;font-size:0.95rem;">Ensure 8-9 hours of sound restorative sleep; avoid physical strain.</span>
                            </div>
                            <div style="background:rgba(0,0,0,0.4);padding:14px;border-radius:10px;border:1px solid rgba(16,185,129,0.4);">
                                <b style="color:#38bdf8;">💧 Hydration & Nutrition:</b><br>
                                <span style="color:#f8fafc;font-size:0.95rem;">Drink 2.5-3 liters of warm water, herbal teas, or electrolyte fluids; eat light warm meals.</span>
                            </div>
                            <div style="background:rgba(0,0,0,0.4);padding:14px;border-radius:10px;border:1px solid rgba(16,185,129,0.4);">
                                <b style="color:#fbbf24;">⏱️ Monitoring Threshold:</b><br>
                                <span style="color:#f8fafc;font-size:0.95rem;">If symptoms do not improve within 48-72 hours or exceed 5 days, consult a physician.</span>
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # 3. Audio Voice Readout for Quick Summary (English)
            speech_summary = f"Quick Medical Report: Major predicted condition is {remedy_data.get('display_name', top_disease)} with {top_confidence:.0f} percent confidence. Identified symptoms include {', '.join(matched_keys)}. Urgency status is {urgency_level}. {'You should seek doctor consultation.' if requires_doctor else 'Supportive home rest is recommended.'}"
            render_tts_button(
                speech_summary,
                lang="en-IN",
                button_label="🔊 Listen to Quick Summary Audio Report",
                stop_label="⏹️ Speaking... (Click to Stop)"
            )

            # 4. Navigation switch button to Sheet 2 (Detailed Report)
            st.markdown("<br>", unsafe_allow_html=True)
            col_btn_det, col_btn_home = st.columns(2)
            with col_btn_det:
                if st.button("📊 View Sheet 2: Detailed Clinical Report & Care Plan", type="primary", use_container_width=True, key="btn_switch_to_detailed_en"):
                    st.session_state["active_report_tab"] = "detailed"
                    st.rerun()
            with col_btn_home:
                if st.button("⬅️ Enter New Symptoms", type="secondary", use_container_width=True, key="btn_quick_back_home_en"):
                    st.session_state["current_page"] = "home"
                    st.rerun()

    # =========================================================================
    # SHEET 2: DETAILED MEDICAL REPORT & FULL CLINICAL DESCRIPTION
    # =========================================================================
    
    else:
        st.session_state["active_report_tab"] = "detailed"
        st.markdown("<hr style='border:none;border-top:1.5px solid rgba(129,140,248,0.3);margin:16px 0 20px 0;'>", unsafe_allow_html=True)

        if is_kannada_active:
            # -------------------------------------------------------------
            # KANNADA DETAILED MEDICAL REPORT (Sheet 2)
            # -------------------------------------------------------------
            st.markdown(
                """
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;">
                    <h3 style="margin:0;color:#a5b4fc;font-size:1.4rem;font-weight:800;display:flex;align-items:center;gap:10px;">
                        <span>📊</span> <span>ಶೀಟ್ 2: ವಿವರವಾದ ವೈದ್ಯಕೀಯ ವರದಿ ಮತ್ತು ಪೂರ್ಣ ವಿವರಣೆ (Detailed Report)</span>
                    </h3>
                    <span style="font-size:0.85rem;background:rgba(129,140,248,0.15);color:#c7d2fe;padding:4px 12px;border-radius:12px;border:1px solid rgba(129,140,248,0.4);">
                        ಸಮಗ್ರ ವಿಶ್ಲೇಷಣೆ & ಆರೈಕೆ ಯೋಜನೆ (Care Plan View)
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

            # 1. Evaluated Patient Clinical Context Card (Kannada)
            if duration_days >= 5:
                phase_str_kn = f"🚨 ದೀರ್ಘಕಾಲೀನ ಹಂತ ({duration_days} ದಿನಗಳು >= 5 ದಿನಗಳ ಮಿತಿ)"
                phase_color = "#f87171"
            elif duration_days <= 3:
                phase_str_kn = f"ತೀವ್ರ ಆರಂಭಿಕ ಹಂತ ({duration_days} ದಿನಗಳು)"
                phase_color = "#38bdf8"
            elif duration_days <= 14:
                phase_str_kn = f"ಮಧ್ಯಮ ಹಂತ ({duration_days} ದಿನಗಳು)"
                phase_color = "#fbbf24"
            else:
                phase_str_kn = f"ದೀರ್ಘಾವಧಿ ಹಂತ ({duration_days} ದಿನಗಳು)"
                phase_color = "#c084fc"

            vuln_tier_map = {
                "Low Vulnerability": "ಕಡಿಮೆ ಅಪಾಯ",
                "Moderate Vulnerability": "ಮಧ್ಯಮ ಅಪಾಯ",
                "High Vulnerability": "ಹೆಚ್ಚಿನ ಅಪಾಯ",
                "Critical Vulnerability": "ಗಂಭೀರ ಅಪಾಯ"
            }
            vuln_tier_kn = vuln_tier_map.get(vuln_tier, vuln_tier)
            vuln_color = "#ef4444" if vuln_score >= 75 else ("#f59e0b" if vuln_score >= 50 else ("#38bdf8" if vuln_score >= 30 else "#34d399"))

            st.markdown(
                f"""
                <div class="glass-card" style="border:1px solid rgba(56,189,248,0.35);background:rgba(15,23,42,0.9);margin-bottom:18px;">
                    <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;margin-bottom:12px;border-bottom:1px solid rgba(255,255,255,0.08);padding-bottom:8px;">
                        <div style="font-weight:800;font-size:1.05rem;color:#f8fafc;display:flex;align-items:center;gap:8px;">
                            <span>🧑‍⚕️ ರೋಗಿಯ ಕ್ಲಿನಿಕಲ್ ವಿವರ (Patient Clinical Context):</span>
                            <span style="font-size:0.85rem;color:#94a3b8;font-weight:500;">(XGBoost ಮಾದರಿ ವಿಶ್ಲೇಷಣೆ)</span>
                        </div>
                        <div style="display:flex;align-items:center;gap:8px;">
                            <span style="font-size:0.82rem;color:#94a3b8;font-weight:600;">ರೋಗಿಯ ದುರ್ಬಲತೆಯ ಸೂಚ್ಯಂಕ:</span>
                            <span style="background:{vuln_color}22;color:{vuln_color};border:1px solid {vuln_color}66;font-weight:800;font-size:0.88rem;padding:2px 10px;border-radius:12px;">
                                {vuln_score}/100 • {vuln_tier_kn}
                            </span>
                        </div>
                    </div>
                    <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(180px, 1fr));gap:12px;">
                        <div>
                            <span style="font-size:0.8rem;color:#94a3b8;">ವ್ಯಕ್ತಿಗತ ವಿವರ (Demographics)</span><br>
                            <span style="font-weight:700;color:#f8fafc;font-size:0.95rem;">{snap.get('age', 28)} ವರ್ಷ • {gender_kn}</span>
                        </div>
                        <div>
                            <span style="font-size:0.8rem;color:#94a3b8;">ಸಮಯ & ಹಂತ (Timeline)</span><br>
                            <span style="font-weight:700;color:{phase_color};font-size:0.95rem;">⏱️ {phase_str_kn}</span>
                        </div>
                        <div>
                            <span style="font-size:0.8rem;color:#94a3b8;">ತೊಂದರೆ ತೀವ್ರತೆ (Severity)</span><br>
                            <span style="font-weight:700;color:{'#f87171' if sev_num>=7 else ('#fbbf24' if sev_num>=4 else '#34d399')};font-size:0.95rem;">
                                ⚡ ಮಟ್ಟ {sev_num}/10
                            </span>
                        </div>
                        <div>
                            <span style="font-size:0.8rem;color:#94a3b8;">ಪೂರ್ವಭಾವಿ ಆರೋಗ್ಯ ಸ್ಥಿತಿ</span><br>
                            <span style="font-weight:700;color:#c084fc;font-size:0.95rem;">🩺 {active_conds_str_kn}</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            # 2. Hero Primary Diagnosis Card & Confirmed Symptoms (Kannada)
            res_col1, res_col2 = st.columns([1.6, 1.4])

            with res_col1:
                st.markdown(
                    f"""
                    <div class="glass-card" style="border:2px solid #38bdf8;box-shadow:0 0 25px rgba(56,189,248,0.15);">
                        <div style="font-size:0.85rem;color:#94a3b8;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;">
                            ಮುಖ್ಯ ರೋಗ ನಿರ್ಣಯ (Primary Clinical Prediction)
                        </div>
                        <div style="font-size:1.85rem;font-weight:800;color:#f8fafc;margin:6px 0 10px 0;">
                            {kn_data['display_name_kn']}
                        </div>
                        <div style="margin-bottom:12px;">
                            <span class="urgency-badge {badge_class}">{kn_data['urgency_kn']}</span>
                        </div>
                        <div style="color:#cbd5e1;font-size:0.92rem;line-height:1.5;margin-bottom:14px;">
                            {kn_data['description_kn']}
                        </div>
                        <div style="background:rgba(15,23,42,0.9);border-radius:10px;padding:10px 14px;border:1px solid rgba(56,189,248,0.3);">
                            <div style="font-size:0.8rem;color:#94a3b8;">ಕ್ಲಿನಿಕಲ್ ಎಐ ಖಚಿತತೆ (AI Confidence)</div>
                            <div style="font-size:1.6rem;font-weight:800;color:#38bdf8;">
                                {top_confidence:.1f}%
                            </div>
                        </div>
                        <div style="margin-top:14px;padding-top:12px;border-top:1px solid rgba(255,255,255,0.08);font-size:0.9rem;color:#cbd5e1;">
                            <b>ಸಲಹೆ ನೀಡಲಾದ ತಜ್ಞ ವೈದ್ಯರು:</b> <span style="color:#38bdf8;font-weight:700;">🧑‍⚕️ {kn_data['specialist_kn']}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with res_col2:
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <div style="font-size:1.1rem;font-weight:700;color:#f8fafc;margin-bottom:10px;display:flex;align-items:center;gap:8px;">
                            <span>📋</span> <span>ದೃಢಪಟ್ಟ ರೋಗಲಕ್ಷಣಗಳು ({len(matched_keys)})</span>
                        </div>
                        <div style="margin-bottom:14px;">
                            {''.join([f'<span class="symptom-tag">✓ {s}</span>' for s in kn_data['translated_symptoms']])}
                        </div>
                        <div style="background:rgba(15,23,42,0.6);border-left:4px solid #38bdf8;padding:10px 14px;border-radius:6px;font-size:0.88rem;color:#cbd5e1;line-height:1.5;">
                            <b>ತಕ್ಷಣದ ಕ್ರಮ:</b> {kn_data['immediate_action_kn']}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # 3. Multi-Parameter Clinical Reasoning & Emergency Alerts (Kannada)
            if duration_days >= 5:
                st.error(f"🚨 **ತುರ್ತು ವೈದ್ಯರ ಭೇಟಿ ಅಗತ್ಯ:** ರೋಗಲಕ್ಷಣಗಳು **{duration_days} ದಿನಗಳಿಂದ** ಮುಂದುವರಿದಿದ್ದು (5 ದಿನಗಳ ಮಿತಿ ಮೀರಿದೆ). ನಿಖರ ತಪಾಸಣೆಗಾಗಿ ತಕ್ಷಣವೇ ವೈದ್ಯರನ್ನು ಭೇಟಿ ಮಾಡಿ.")

            if clinical_reasons:
                reasons_html = "".join([f'<li style="margin-bottom:6px;color:#cbd5e1;">{r}</li>' for r in clinical_reasons])
                st.markdown(
                    f"""
                    <div class="glass-card" style="border-left:5px solid #818cf8;padding:16px 20px;">
                        <div style="font-weight:700;color:#a5b4fc;font-size:1.05rem;margin-bottom:8px;display:flex;align-items:center;gap:8px;">
                            <span>🧠</span> <span>ಬಹು-ಅಂಶಗಳ ಕ್ಲಿನಿಕಲ್ ವಿಶ್ಲೇಷಣೆ ಮತ್ತು ಕಾರಣಗಳು (Clinical Reasoning)</span>
                        </div>
                        <ul style="padding-left:20px;margin-bottom:0;">
                            {reasons_html}
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # 4. Voice Assistant Audio Readout in Kannada
            render_tts_button(
                kn_data['tts_text_kn'],
                lang="kn-IN",
                button_label="🎙️ ರೋಗ ನಿರ್ಣಯ ಮತ್ತು ಸಮಗ್ರ ವರದಿಯನ್ನು ಆಲಿಸಿ (ಕನ್ನಡ ಧ್ವನಿ)",
                stop_label="⏹️ ಕನ್ನಡದಲ್ಲಿ ಓದಲಾಗುತ್ತಿದೆ... (ನಿಲ್ಲಿಸಲು ಕ್ಲಿಕ್ ಮಾಡಿ)"
            )

            # 5. Differential Diagnoses & Probability Breakdown (Kannada)
            st.markdown("### 🔬 ಇತರ ಸಂಭವನೀಯ ರೋಗಗಳು & ಸಂಭವನೀಯತೆಯ ವಿಶ್ಲೇಷಣೆ (Differential Diagnoses)")
            diff_cols = st.columns(min(len(predictions), 3))
            for idx, (d_name, d_prob) in enumerate(predictions[:3]):
                d_std_name = DISEASE_NAME_MAP.get(d_name, d_name)
                d_kn_title = DISEASE_KNOWLEDGE_KN.get(d_std_name, {}).get("display_name_kn", d_std_name)
                with diff_cols[idx]:
                    st.markdown(
                        f"""
                        <div class="glass-card" style="padding:14px;border:1px solid rgba(255,255,255,0.08);">
                            <div style="font-size:0.8rem;color:#94a3b8;">ಸಂಭವನೀಯತೆ #{idx+1}</div>
                            <div style="font-weight:700;color:#f8fafc;font-size:0.95rem;margin:4px 0 8px 0;height:48px;display:flex;align-items:center;">
                                {d_kn_title}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.progress(min(float(d_prob) / 100.0, 1.0), text=f"{d_prob:.1f}%")

            # 6. Interactive Care Plan Filter (Kannada)
            st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.12);margin:28px 0 16px 0;'>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="glass-card" style="border:1.5px solid #38bdf8;background:rgba(15,23,42,0.92);padding:18px 22px;margin-bottom:12px;">
                    <div style="font-weight:800;font-size:1.15rem;color:#38bdf8;display:flex;align-items:center;gap:8px;">
                        <span>🌿</span> <span>ಮನೆ ಮದ್ದು & ಸಮಗ್ರ ಆರೈಕೆ ಸಮಾಲೋಚನೆ (Integrative Care Plan)</span>
                    </div>
                    <div style="font-size:0.95rem;color:#e2e8f0;margin-top:6px;">
                        <b>{kn_data['display_name_kn']}</b> ಕಾಯಿಲೆಗೆ ಸಾಕ್ಷ್ಯಾಧಾರಿತ <b>ಮನೆ ಮದ್ದು</b>, <b>ಆಯುರ್ವೇದ ಕಷಾಯ</b>, <b>ಆಹಾರ ಪದ್ಧತಿ</b>, ಮತ್ತು <b>ತಯಾರಿಕಾ ವಿಧಾನಗಳನ್ನು</b> ವೀಕ್ಷಿಸಿ.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            remedy_view_choice_kn = st.radio(
                "ಆರೈಕೆ ಪ್ಯಾಕೇಜ್ ವೀಕ್ಷಣೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ:",
                [
                    "🌿 ಹೌದು, ಸಂಪೂರ್ಣ ಆರೈಕೆ ಪ್ಯಾಕೇಜ್ ವೀಕ್ಷಿಸಿ (ಮನೆಮದ್ದು, ಆಯುರ್ವೇದ, ಆಹಾರ & ಪಾಕವಿಧಾನಗಳು)",
                    "🏡 ಮನೆ ಮದ್ದು & ಆಯುರ್ವೇದ ಕಷಾಯ ಮಾತ್ರ",
                    "🥗 ಆಹಾರ ನಿಯಮಗಳು & ಪೋಷಕಾಂಶ ಪದ್ಧತಿ ಮಾತ್ರ",
                    "🍲 ಹಂತ-ಹಂತದ ತಯಾರಿಕಾ ಪಾಕವಿಧಾನಗಳು & ಸಲಹೆಗಳು ಮಾತ್ರ",
                    "🛡️ ಸುರಕ್ಷತಾ ಮುನ್ನೆಚ್ಚರಿಕೆಗಳು & ಎಚ್ಚರಿಕೆಗಳು ಮಾತ್ರ",
                    "❌ ಕೇವಲ ರೋಗನಿರ್ಣಯ ವೀಕ್ಷಿಸಿ (ಆರೈಕೆ ಸಲಹೆ ಮರೆಮಾಡಿ)"
                ],
                index=0,
                key="remedy_view_choice_radio_kn"
            )

            home_rems_kn = kn_data.get("home_remedies_kn", [])
            ayur_rems_kn = kn_data.get("ayurvedic_kn", [])
            diet_dos_kn = kn_data.get("diet_dos_kn", [])
            diet_donts_kn = kn_data.get("diet_donts_kn", [])
            precautions_kn_list = kn_data.get("precautions_list_kn", [])
            comorb_precautions_kn = get_comorbidity_tailored_precautions_kn(snap.get("conditions", []), top_disease)

            # Matching condition guide or relevant recipes
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

            # RENDER SELECTED CARE PLAN IN KANNADA
            if "ಸಂಪೂರ್ಣ ಆರೈಕೆ ಪ್ಯಾಕೇಜ್" in remedy_view_choice_kn:
                st.markdown("### 🌿 ಸಮಗ್ರ ಆರೈಕೆ ಪ್ಯಾಕೇಜ್ (Complete Care Package)")
                rem_col1, rem_col2 = st.columns(2)
                with rem_col1:
                    home_items_html = "".join([f'<li style="margin-bottom:10px;color:#cbd5e1;line-height:1.5;">{r}</li>' for r in home_rems_kn])
                    st.markdown(
                        f"""
                        <div class="glass-card" style="border-left:5px solid #10b981;padding:18px;">
                            <h4 style="margin-top:0;color:#34d399;font-size:1.15rem;display:flex;align-items:center;gap:8px;">
                                <span>🏡 ಸಾಕ್ಷ್ಯಾಧಾರಿತ ಮನೆ ಮದ್ದುಗಳು (Home Remedies)</span>
                            </h4>
                            <ul style="padding-left:20px;margin-bottom:0;">
                                {home_items_html}
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    ayur_items_html = "".join([f'<li style="margin-bottom:10px;color:#cbd5e1;line-height:1.5;">{a}</li>' for a in ayur_rems_kn])
                    st.markdown(
                        f"""
                        <div class="glass-card" style="border-left:5px solid #f59e0b;padding:18px;margin-top:14px;">
                            <h4 style="margin-top:0;color:#fbbf24;font-size:1.15rem;display:flex;align-items:center;gap:8px;">
                                <span>🍵 ಸಾಂಪ್ರದಾಯಿಕ ಆಯುರ್ವೇದ ಕಷಾಯ & ಚಿಕಿತ್ಸೆ (Ayurvedic Kadha)</span>
                            </h4>
                            <ul style="padding-left:20px;margin-bottom:0;">
                                {ayur_items_html}
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                with rem_col2:
                    diet_do_html = "".join([f'<div style="margin-bottom:8px;color:#7dd3fc;font-weight:600;font-size:0.95rem;line-height:1.4;">✓ {d}</div>' for d in diet_dos_kn])
                    diet_dont_html = "".join([f'<div style="margin-bottom:8px;color:#fca5a5;font-weight:600;font-size:0.95rem;line-height:1.4;">✗ {d}</div>' for d in diet_donts_kn])
                    st.markdown(
                        f"""
                        <div class="glass-card" style="border-left:5px solid #38bdf8;padding:18px;">
                            <h4 style="margin-top:0;color:#38bdf8;font-size:1.15rem;display:flex;align-items:center;gap:8px;">
                                <span>🥗 ಆಹಾರ ನಿಯಮಗಳು - ಸೇವಿಸಬೇಕಾದ & ತ್ಯಜಿಸಬೇಕಾದ ಆಹಾರ</span>
                            </h4>
                            <div style="font-weight:700;color:#34d399;margin-bottom:6px;font-size:0.92rem;">ಸೇವಿಸಲು ಶಿಫಾರಸು ಮಾಡಿದ ಆಹಾರಗಳು:</div>
                            {diet_do_html}
                            <div style="font-weight:700;color:#f87171;margin:12px 0 6px 0;font-size:0.92rem;">ಸಂಪೂರ್ಣವಾಗಿ ತ್ಯಜಿಸಬೇಕಾದ ಆಹಾರಗಳು:</div>
                            {diet_dont_html}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    precautions_html = "".join([f'<li style="margin-bottom:8px;color:#fca5a5;line-height:1.4;">⚠️ {p}</li>' for p in precautions_kn_list])
                    st.markdown(
                        f"""
                        <div class="glass-card" style="border-left:5px solid #ef4444;padding:18px;margin-top:14px;">
                            <h4 style="margin-top:0;color:#f87171;font-size:1.15rem;display:flex;align-items:center;gap:8px;">
                                <span>🛡️ ಪ್ರಮುಖ ಸುರಕ್ಷತಾ ನಿಯಮಗಳು & ಮುನ್ನೆಚ್ಚರಿಕೆಗಳು</span>
                            </h4>
                            <ul style="padding-left:20px;margin-bottom:0;">
                                {precautions_html}
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Comorbidity tailored precautions in Kannada
                if comorb_precautions_kn:
                    comorb_items = "".join([f'<li style="margin-bottom:8px;color:#e9d5ff;line-height:1.5;">{cp}</li>' for cp in comorb_precautions_kn])
                    st.markdown(
                        f"""
                        <div class="glass-card" style="border-left:5px solid #c084fc;padding:18px;margin-top:14px;">
                            <h4 style="margin-top:0;color:#c084fc;font-size:1.15rem;">
                                🩺 ರೋಗಿಯ ಆರೋಗ್ಯ ಸ್ಥಿತಿಗೆ ತಕ್ಕ ಮುನ್ನೆಚ್ಚರಿಕೆಗಳು ({active_conds_str_kn})
                            </h4>
                            <ul style="padding-left:20px;margin-bottom:0;">
                                {comorb_items}
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Relevant Preparation Recipes
                if relevant_recipes:
                    st.markdown("### 🍲 ಹಂತ-ಹಂತದ ತಯಾರಿಕಾ ಪಾಕವಿಧಾನಗಳು & ವಿಡಿಯೋ ಮಾರ್ಗದರ್ಶಿ")
                    for recipe in relevant_recipes:
                        r_name = recipe.get("name", "Home Remedy")
                        r_icon = recipe.get("icon", "🍵")
                        r_make = recipe.get("make", "")
                        r_why = recipe.get("why", "")
                        r_vid = recipe.get("video", "")
                        r_vlabel = recipe.get("videoLabel", "Medical Tutorial")
                        st.markdown(
                            f"""
                            <div class="glass-card" style="border:1.5px solid rgba(245,158,11,0.35);padding:20px;margin-bottom:14px;">
                                <div style="font-weight:800;color:#fbbf24;font-size:1.15rem;display:flex;align-items:center;gap:10px;">
                                    <span style="font-size:1.3rem;">{r_icon}</span> <span>{r_name}</span>
                                </div>
                                <div style="margin-top:10px;font-size:0.92rem;color:#cbd5e1;line-height:1.6;">
                                    <b style="color:#38bdf8;">ತಯಾರಿಸುವ ವಿಧಾನ (How to Prepare):</b> {r_make}
                                </div>
                                <div style="margin-top:10px;font-size:0.88rem;color:#94a3b8;line-height:1.5;">
                                    <b style="color:#34d399;">ಆರೋಗ್ಯ ಪ್ರಯೋಜನ (Health Benefit):</b> {r_why}
                                </div>
                                {f'<div style="margin-top:12px;"><a href="{r_vid}" target="_blank" style="background:rgba(239,68,68,0.2);color:#f87171;border:1px solid #ef4444;padding:6px 14px;border-radius:18px;font-weight:700;font-size:0.85rem;text-decoration:none;display:inline-block;">▶ ಯೂಟ್ಯೂಬ್ ವಿಡಿಯೋ ವೀಕ್ಷಿಸಿ: {r_vlabel}</a></div>' if r_vid else ''}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            elif "ಮನೆ ಮದ್ದು" in remedy_view_choice_kn:
                st.markdown("### 🏡 ಮನೆ ಮದ್ದು & ಆಯುರ್ವೇದ ಕಷಾಯ (Home Remedies & Ayurveda)")
                rem_col1, rem_col2 = st.columns(2)
                with rem_col1:
                    home_items_html = "".join([f'<li style="margin-bottom:10px;color:#cbd5e1;line-height:1.5;">{r}</li>' for r in home_rems_kn])
                    st.markdown(
                        f"""
                        <div class="glass-card" style="border-left:5px solid #10b981;padding:18px;">
                            <h4 style="margin-top:0;color:#34d399;font-size:1.15rem;">🏡 ಸಾಕ್ಷ್ಯಾಧಾರಿತ ಮನೆ ಮದ್ದುಗಳು</h4>
                            <ul style="padding-left:20px;margin-bottom:0;">
                                {home_items_html}
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with rem_col2:
                    ayur_items_html = "".join([f'<li style="margin-bottom:10px;color:#cbd5e1;line-height:1.5;">{a}</li>' for a in ayur_rems_kn])
                    st.markdown(
                        f"""
                        <div class="glass-card" style="border-left:5px solid #f59e0b;padding:18px;">
                            <h4 style="margin-top:0;color:#fbbf24;font-size:1.15rem;">🍵 ಸಾಂಪ್ರದಾಯಿಕ ಆಯುರ್ವೇದ ಕಷಾಯ</h4>
                            <ul style="padding-left:20px;margin-bottom:0;">
                                {ayur_items_html}
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            elif "ಆಹಾರ ನಿಯಮಗಳು" in remedy_view_choice_kn:
                st.markdown("### 🥗 ಆಹಾರ ನಿಯಮಗಳು & ಪೋಷಕಾಂಶ ಪದ್ಧತಿ (Dietary Protocol)")
                diet_do_html = "".join([f'<div style="margin-bottom:8px;color:#7dd3fc;font-weight:600;font-size:0.95rem;line-height:1.4;">✓ {d}</div>' for d in diet_dos_kn])
                diet_dont_html = "".join([f'<div style="margin-bottom:8px;color:#fca5a5;font-weight:600;font-size:0.95rem;line-height:1.4;">✗ {d}</div>' for d in diet_donts_kn])
                col_d1, col_d2 = st.columns(2)
                with col_d1:
                    st.markdown(
                        f"""
                        <div class="glass-card" style="border-left:5px solid #10b981;padding:18px;">
                            <h4 style="margin-top:0;color:#34d399;font-size:1.15rem;">✓ ಸೇವಿಸಲು ಶಿಫಾರಸು ಮಾಡಿದ ಆಹಾರ</h4>
                            {diet_do_html}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                with col_d2:
                    st.markdown(
                        f"""
                        <div class="glass-card" style="border-left:5px solid #ef4444;padding:18px;">
                            <h4 style="margin-top:0;color:#f87171;font-size:1.15rem;">✗ ತ್ಯಜಿಸಬೇಕಾದ ಆಹಾರಗಳು</h4>
                            {diet_dont_html}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            elif "ಹಂತ-ಹಂತದ" in remedy_view_choice_kn:
                st.markdown("### 🍲 ಹಂತ-ಹಂತದ ತಯಾರಿಕಾ ಪಾಕವಿಧಾನಗಳು (Preparation Recipes)")
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
                            <div class="glass-card" style="border:1.5px solid rgba(245,158,11,0.35);padding:20px;margin-bottom:14px;">
                                <div style="font-weight:800;color:#fbbf24;font-size:1.15rem;display:flex;align-items:center;gap:10px;">
                                    <span style="font-size:1.3rem;">{r_icon}</span> <span>{r_name}</span>
                                </div>
                                <div style="margin-top:10px;font-size:0.92rem;color:#cbd5e1;line-height:1.6;">
                                    <b style="color:#38bdf8;">ತಯಾರಿಸುವ ವಿಧಾನ:</b> {r_make}
                                </div>
                                <div style="margin-top:10px;font-size:0.88rem;color:#94a3b8;line-height:1.5;">
                                    <b style="color:#34d399;">ಆರೋಗ್ಯ ಪ್ರಯೋಜನ:</b> {r_why}
                                </div>
                                {f'<div style="margin-top:12px;"><a href="{r_vid}" target="_blank" style="background:rgba(239,68,68,0.2);color:#f87171;border:1px solid #ef4444;padding:6px 14px;border-radius:18px;font-weight:700;font-size:0.85rem;text-decoration:none;display:inline-block;">▶ ವಿಡಿಯೋ ವೀಕ್ಷಿಸಿ: {r_vlabel}</a></div>' if r_vid else ''}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.info(f"{kn_data['display_name_kn']} ಕಾಯಿಲೆಗೆ ಸಾಮಾನ್ಯ ವಿಶ್ರಾಂತಿ ಮತ್ತು ಬಿಸಿ ನೀರು ಕುಡಿಯುವುದು ಸೂಕ್ತ.")

            elif "ಮುನ್ನೆಚ್ಚರಿಕೆಗಳು" in remedy_view_choice_kn:
                st.markdown("### 🛡️ ಕ್ಲಿನಿಕಲ್ ಮುನ್ನೆಚ್ಚರಿಕೆಗಳು & ಸುರಕ್ಷತಾ ನಿಯಮಗಳು")
                precautions_html = "".join([f'<li style="margin-bottom:10px;color:#fca5a5;line-height:1.5;">⚠️ {p}</li>' for p in precautions_kn_list])
                st.markdown(
                    f"""
                    <div class="glass-card" style="border-left:5px solid #ef4444;padding:20px;">
                        <h4 style="margin-top:0;color:#f87171;font-size:1.15rem;">
                            🛡️ ಸುರಕ್ಷತಾ ನಿಯಮಗಳು & ಎಚ್ಚರಿಕೆಗಳು
                        </h4>
                        <ul style="padding-left:20px;margin-bottom:0;">
                            {precautions_html}
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:
                st.info("ℹ️ ನಿಮ್ಮ ಆಯ್ಕೆಯಂತೆ ಮನೆಮದ್ದುಗಳನ್ನು ಮರೆಮಾಡಲಾಗಿದೆ. ನೀವು ಕ್ಲಿನಿಕಲ್ ರೋಗನಿರ್ಣಯವನ್ನು ವೀಕ್ಷಿಸುತ್ತಿದ್ದೀರಿ.")

            # Navigation switch back to Quick Summary or Home (Kannada)
            st.markdown("<br>", unsafe_allow_html=True)
            col_btn_quick, col_btn_back2 = st.columns(2)
            with col_btn_quick:
                if st.button("📄 ಶೀಟ್ 1 ಕ್ಕೆ ಬದಲಿಸಿ: ತ್ವರಿತ ವೈದ್ಯಕೀಯ ಸಾರಾಂಶ ವರದಿ", type="secondary", use_container_width=True, key="btn_switch_to_quick"):
                    st.session_state["active_report_tab"] = "quick"
                    st.rerun()

            with col_btn_back2:
                if st.button("⬅️ ರೋಗಲಕ್ಷಣಗಳ ಪುಟಕ್ಕೆ ಹಿಂತಿರುಗಿ", type="primary", use_container_width=True, key="btn_detailed_back_home"):
                    st.session_state["current_page"] = "home"
                    st.rerun()

        else:
            # -------------------------------------------------------------
            # ENGLISH DETAILED MEDICAL REPORT (Sheet 2)
            # -------------------------------------------------------------
            st.markdown(
                """
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px;">
                    <h3 style="margin:0;color:#a5b4fc;font-size:1.4rem;font-weight:800;display:flex;align-items:center;gap:10px;">
                        <span>📊</span> <span>Sheet 2: Detailed Medical Report & Full Clinical Description</span>
                    </h3>
                    <span style="font-size:0.85rem;background:rgba(129,140,248,0.15);color:#c7d2fe;padding:4px 12px;border-radius:12px;border:1px solid rgba(129,140,248,0.4);">
                        Comprehensive Multi-Class & Care Plan View
                    </span>
                </div>
                """,
                unsafe_allow_html=True
            )

            # 1. Evaluated Patient Clinical Context Card
            if duration_days >= 5:
                phase_str = f"🚨 Prolonged Phase ({duration_days} Days >= 5 Days Threshold)"
                phase_color = "#f87171"
            elif duration_days <= 3:
                phase_str = f"Acute Phase ({duration_days} Days)"
                phase_color = "#38bdf8"
            elif duration_days <= 14:
                phase_str = f"Subacute Phase ({duration_days} Days)"
                phase_color = "#fbbf24"
            else:
                phase_str = f"Chronic / Persistent ({duration_days} Days)"
                phase_color = "#c084fc"

            vuln_color = "#ef4444" if vuln_score >= 75 else ("#f59e0b" if vuln_score >= 50 else ("#38bdf8" if vuln_score >= 30 else "#34d399"))

            st.markdown(
                f"""
                <div class="glass-card" style="border:1px solid rgba(56,189,248,0.35);background:rgba(15,23,42,0.9);margin-bottom:18px;">
                    <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;margin-bottom:12px;border-bottom:1px solid rgba(255,255,255,0.08);padding-bottom:8px;">
                        <div style="font-weight:800;font-size:1.05rem;color:#f8fafc;display:flex;align-items:center;gap:8px;">
                            <span>🧑‍⚕️ Evaluated Patient Clinical Context:</span>
                            <span style="font-size:0.85rem;color:#94a3b8;font-weight:500;">(Integrated with XGBoost Bayesian Prior)</span>
                        </div>
                        <div style="display:flex;align-items:center;gap:8px;">
                            <span style="font-size:0.82rem;color:#94a3b8;font-weight:600;">Patient Vulnerability Index:</span>
                            <span style="background:{vuln_color}22;color:{vuln_color};border:1px solid {vuln_color}66;font-weight:800;font-size:0.88rem;padding:2px 10px;border-radius:12px;">
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
                            <span style="font-weight:700;color:{phase_color};font-size:0.95rem;">⏱️ {phase_str}</span>
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

            # 2. Hero Primary Diagnosis Card & Confirmed Symptoms
            res_col1, res_col2 = st.columns([1.6, 1.4])

            with res_col1:
                st.markdown(
                    f"""
                    <div class="glass-card" style="border:2px solid #38bdf8;box-shadow:0 0 25px rgba(56,189,248,0.15);">
                        <div style="font-size:0.85rem;color:#94a3b8;font-weight:600;text-transform:uppercase;letter-spacing:0.5px;">
                            Primary Clinical Prediction
                        </div>
                        <div style="font-size:1.85rem;font-weight:800;color:#f8fafc;margin:6px 0 10px 0;">
                            {remedy_data.get('display_name', top_disease)}
                        </div>
                        <div style="margin-bottom:12px;">
                            <span class="urgency-badge {badge_class}">{urgency_level}</span>
                        </div>
                        <div style="color:#cbd5e1;font-size:0.92rem;line-height:1.5;margin-bottom:14px;">
                            {remedy_data.get('description', 'Clinical analysis based on matched symptom profile and Bayesian risk weighting.')}
                        </div>
                        <div style="background:rgba(15,23,42,0.9);border-radius:10px;padding:10px 14px;border:1px solid rgba(56,189,248,0.3);">
                            <div style="font-size:0.8rem;color:#94a3b8;">Calibrated Clinical Confidence</div>
                            <div style="font-size:1.6rem;font-weight:800;color:#38bdf8;">
                                {top_confidence:.1f}%
                            </div>
                        </div>
                        <div style="margin-top:14px;padding-top:12px;border-top:1px solid rgba(255,255,255,0.08);font-size:0.9rem;color:#cbd5e1;">
                            <b>Recommended Specialist:</b> <span style="color:#38bdf8;font-weight:700;">🧑‍⚕️ {remedy_data.get('specialist', 'General Physician')}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with res_col2:
                st.markdown(
                    f"""
                    <div class="glass-card">
                        <div style="font-size:1.1rem;font-weight:700;color:#f8fafc;margin-bottom:10px;display:flex;align-items:center;gap:8px;">
                            <span>📋</span> <span>Confirmed Symptoms ({len(matched_keys)})</span>
                        </div>
                        <div style="margin-bottom:14px;">
                            {''.join([f'<span class="symptom-tag">✓ {s.replace("_", " ").title()}</span>' for s in matched_keys])}
                        </div>
                        <div style="background:rgba(15,23,42,0.6);border-left:4px solid #38bdf8;padding:10px 14px;border-radius:6px;font-size:0.88rem;color:#cbd5e1;line-height:1.5;">
                            <b>Triage Action:</b> {action_advice}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # 3. Multi-Parameter Clinical Reasoning & Emergency Alerts
            if duration_days >= 5:
                st.error(f"🚨 **Urgent Physician Consultation Required:** Patient symptoms have persisted for **{duration_days} days** (>= 5-6 days threshold). Please consult a doctor immediately for diagnostic evaluation.")

            if clinical_reasons:
                reasons_html = "".join([f'<li style="margin-bottom:6px;color:#cbd5e1;">{r}</li>' for r in clinical_reasons])
                st.markdown(
                    f"""
                    <div class="glass-card" style="border-left:5px solid #818cf8;padding:16px 20px;">
                        <div style="font-weight:700;color:#a5b4fc;font-size:1.05rem;margin-bottom:8px;display:flex;align-items:center;gap:8px;">
                            <span>🧠</span> <span>Multi-Parameter Clinical Reasoning</span>
                        </div>
                        <ul style="padding-left:20px;margin-bottom:0;">
                            {reasons_html}
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            # 4. Voice Assistant Audio Readout
            speech_script = f"Clinical Assessment: The symptoms strongly match {remedy_data.get('display_name', top_disease)} with {top_confidence:.0f} percent confidence. Recommended specialist is {remedy_data.get('specialist', 'General Physician')}. Urgency level is {urgency_level}. {action_advice}."
            render_tts_button(speech_script, lang="en-IN")

            # 5. Differential Diagnoses & Probability Breakdown
            st.markdown("### 🔬 Differential Diagnoses & Probability Breakdown")
            diff_cols = st.columns(min(len(predictions), 3))
            for idx, (d_name, d_prob) in enumerate(predictions[:3]):
                with diff_cols[idx]:
                    st.markdown(
                        f"""
                        <div class="glass-card" style="padding:14px;border:1px solid rgba(255,255,255,0.08);">
                            <div style="font-size:0.8rem;color:#94a3b8;">Differential #{idx+1}</div>
                            <div style="font-weight:700;color:#f8fafc;font-size:1.05rem;margin:4px 0 8px 0;height:48px;display:flex;align-items:center;">
                                {DISEASE_NAME_MAP.get(d_name, d_name)}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    st.progress(min(float(d_prob) / 100.0, 1.0), text=f"{d_prob:.1f}%")

            # 6. Interactive Care Plan Filter
            st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.12);margin:28px 0 16px 0;'>", unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="glass-card" style="border:1.5px solid #38bdf8;background:rgba(15,23,42,0.92);padding:18px 22px;margin-bottom:12px;">
                    <div style="font-weight:800;font-size:1.15rem;color:#38bdf8;display:flex;align-items:center;gap:8px;">
                        <span>🌿</span> <span>Home Remedies & Integrative Care Consultation</span>
                    </div>
                    <div style="font-size:0.95rem;color:#e2e8f0;margin-top:6px;">
                        Explore evidence-based <b>Home Remedies</b>, <b>Ayurvedic Formulations</b>, <b>Diet Guidelines</b>, and <b>Preparation Recipes</b> for <b>{remedy_data.get('display_name', top_disease)}</b>.
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            remedy_view_choice = st.radio(
                "Filter Care Package View:",
                [
                    "🌿 Yes, show Complete Care Package (Home Remedies, Ayurvedic Kadha, Diet & Recipes)",
                    "🏡 Home Remedies & Ayurvedic Formulations Only",
                    "🥗 Dietary Guidelines & Nutrition Protocol Only",
                    "🍲 Step-by-Step Preparation Recipes & Video Guides Only",
                    "🛡️ Clinical Precautions & Safety Guidelines Only",
                    "❌ No, show Clinical Diagnosis Only (Hide Remedies)"
                ],
                index=0,
                key="remedy_view_choice_radio"
            )

            home_rems = remedy_data.get("home_remedies", [])
            ayur_rems = remedy_data.get("ayurvedic", [])
            diet_dos = remedy_data.get("diet_do", [])
            diet_donts = remedy_data.get("diet_dont", [])
            precautions = remedy_data.get("precautions", [])

            # Matching condition guide or relevant recipes
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

            # RENDER SELECTED CARE PLAN ACCORDING TO USER'S PREFERENCE
            if "Complete Care Package" in remedy_view_choice:
                st.markdown("### 🌿 Complete Integrative Care Package")
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
                    diet_do_html = "".join([f'<div style="margin-bottom:8px;color:#7dd3fc;font-weight:600;font-size:0.95rem;line-height:1.4;">✓ {d}</div>' for d in diet_dos])
                    diet_dont_html = "".join([f'<div style="margin-bottom:8px;color:#fca5a5;font-weight:600;font-size:0.95rem;line-height:1.4;">✗ {d}</div>' for d in diet_donts])
                    st.markdown(
                        f"""
                        <div class="glass-card" style="border-left:5px solid #38bdf8;padding:18px;">
                            <h4 style="margin-top:0;color:#38bdf8;font-size:1.15rem;display:flex;align-items:center;gap:8px;">
                                <span>🥗 Dietary Protocol (Do's & Don'ts)</span>
                            </h4>
                            <div style="font-weight:700;color:#34d399;margin-bottom:6px;font-size:0.92rem;">Recommended Foods:</div>
                            {diet_do_html}
                            <div style="font-weight:700;color:#f87171;margin:12px 0 6px 0;font-size:0.92rem;">Foods to Avoid:</div>
                            {diet_dont_html}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    precautions_html = "".join([f'<li style="margin-bottom:8px;color:#fca5a5;line-height:1.4;">⚠️ {p}</li>' for p in precautions])
                    st.markdown(
                        f"""
                        <div class="glass-card" style="border-left:5px solid #ef4444;padding:18px;margin-top:14px;">
                            <h4 style="margin-top:0;color:#f87171;font-size:1.15rem;display:flex;align-items:center;gap:8px;">
                                <span>🛡️ Clinical Precautions & Red Flags</span>
                            </h4>
                            <ul style="padding-left:20px;margin-bottom:0;">
                                {precautions_html}
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Comorbidity tailored precautions if present
                if comorbidity_precautions:
                    comorb_items = "".join([f'<li style="margin-bottom:8px;color:#e9d5ff;line-height:1.5;">🩺 {cp}</li>' for cp in comorbidity_precautions])
                    st.markdown(
                        f"""
                        <div class="glass-card" style="border-left:5px solid #c084fc;padding:18px;margin-top:14px;">
                            <h4 style="margin-top:0;color:#c084fc;font-size:1.15rem;">
                                🩺 Patient Condition-Tailored Precautions ({active_conds_str})
                            </h4>
                            <ul style="padding-left:20px;margin-bottom:0;">
                                {comorb_items}
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # Relevant Preparation Recipes
                if relevant_recipes:
                    st.markdown("### 🍲 Step-by-Step Preparation Recipes & Video Guides")
                    for recipe in relevant_recipes:
                        r_name = recipe.get("name", "Home Remedy")
                        r_icon = recipe.get("icon", "🍵")
                        r_make = recipe.get("make", "")
                        r_why = recipe.get("why", "")
                        r_vid = recipe.get("video", "")
                        r_vlabel = recipe.get("videoLabel", "Medical Tutorial")
                        st.markdown(
                            f"""
                            <div class="glass-card" style="border:1.5px solid rgba(245,158,11,0.35);padding:20px;margin-bottom:14px;">
                                <div style="font-weight:800;color:#fbbf24;font-size:1.15rem;display:flex;align-items:center;gap:10px;">
                                    <span style="font-size:1.3rem;">{r_icon}</span> <span>{r_name}</span>
                                </div>
                                <div style="margin-top:10px;font-size:0.92rem;color:#cbd5e1;line-height:1.6;">
                                    <b style="color:#38bdf8;">How to Prepare:</b> {r_make}
                                </div>
                                <div style="margin-top:10px;font-size:0.88rem;color:#94a3b8;line-height:1.5;">
                                    <b style="color:#34d399;">Biological Mechanism:</b> {r_why}
                                </div>
                                {f'<div style="margin-top:12px;"><a href="{r_vid}" target="_blank" style="background:rgba(239,68,68,0.2);color:#f87171;border:1px solid #ef4444;padding:6px 14px;border-radius:18px;font-weight:700;font-size:0.85rem;text-decoration:none;display:inline-block;">▶ Watch YouTube Video Tutorial: {r_vlabel}</a></div>' if r_vid else ''}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            elif "Home Remedies & Ayurvedic" in remedy_view_choice:
                st.markdown("### 🏡 Home Remedies & Ayurvedic Formulations")
                rem_col1, rem_col2 = st.columns(2)
                with rem_col1:
                    home_items_html = "".join([f'<li style="margin-bottom:10px;color:#cbd5e1;line-height:1.5;">{r}</li>' for r in home_rems])
                    st.markdown(
                        f"""
                        <div class="glass-card" style="border-left:5px solid #10b981;padding:18px;">
                            <h4 style="margin-top:0;color:#34d399;font-size:1.15rem;">🏡 Evidence-Based Home Remedies</h4>
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
                            <h4 style="margin-top:0;color:#fbbf24;font-size:1.15rem;">🍵 Traditional Ayurvedic Formulations</h4>
                            <ul style="padding-left:20px;margin-bottom:0;">
                                {ayur_items_html}
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            elif "Dietary Guidelines" in remedy_view_choice:
                st.markdown("### 🥗 Dietary Guidelines & Nutrition Protocol")
                diet_do_html = "".join([f'<div style="margin-bottom:8px;color:#7dd3fc;font-weight:600;font-size:0.95rem;line-height:1.4;">✓ {d}</div>' for d in diet_dos])
                diet_dont_html = "".join([f'<div style="margin-bottom:8px;color:#fca5a5;font-weight:600;font-size:0.95rem;line-height:1.4;">✗ {d}</div>' for d in diet_donts])
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

            elif "Step-by-Step" in remedy_view_choice:
                st.markdown("### 🍲 Step-by-Step Preparation Recipes & Medical Video Guides")
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
                            <div class="glass-card" style="border:1.5px solid rgba(245,158,11,0.35);padding:20px;margin-bottom:14px;">
                                <div style="font-weight:800;color:#fbbf24;font-size:1.15rem;display:flex;align-items:center;gap:10px;">
                                    <span style="font-size:1.3rem;">{r_icon}</span> <span>{r_name}</span>
                                </div>
                                <div style="margin-top:10px;font-size:0.92rem;color:#cbd5e1;line-height:1.6;">
                                    <b style="color:#38bdf8;">How to Prepare:</b> {r_make}
                                </div>
                                <div style="margin-top:10px;font-size:0.88rem;color:#94a3b8;line-height:1.5;">
                                    <b style="color:#34d399;">Biological Mechanism:</b> {r_why}
                                </div>
                                {f'<div style="margin-top:12px;"><a href="{r_vid}" target="_blank" style="background:rgba(239,68,68,0.2);color:#f87171;border:1px solid #ef4444;padding:6px 14px;border-radius:18px;font-weight:700;font-size:0.85rem;text-decoration:none;display:inline-block;">▶ Watch YouTube Video Tutorial: {r_vlabel}</a></div>' if r_vid else ''}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                else:
                    st.info(f"General hydration and rest are recommended for {remedy_data.get('display_name', top_disease)}.")

            elif "Clinical Precautions" in remedy_view_choice:
                st.markdown("### 🛡️ Clinical Precautions & Safety Guidelines")
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
                st.info("ℹ️ Home Remedies are hidden as per your preference. You are viewing the clinical diagnostic assessment.")

            # Navigation switch back to Quick Summary or Home (English)
            st.markdown("<br>", unsafe_allow_html=True)
            col_btn_quick, col_btn_back2 = st.columns(2)
            with col_btn_quick:
                if st.button("📄 Switch to Sheet 1: Quick Medical Summary Report", type="secondary", use_container_width=True, key="btn_switch_to_quick_en"):
                    st.session_state["active_report_tab"] = "quick"
                    st.rerun()

            with col_btn_back2:
                if st.button("⬅️ Back to Symptom Input", type="primary", use_container_width=True, key="btn_detailed_back_home_en"):
                    st.session_state["current_page"] = "home"
                    st.rerun()


# =========================================================
# ROUTE B: HOME PAGE (Input Console & Full Tabs)
# =========================================================
else:
    # Main Page Hero Header
    st.markdown(
        """
        <div class="hero-banner">
            <div style="display:flex;align-items:center;gap:15px;">
                <span style="font-size:2.4rem;">🩺</span>
                <div>
                    <h1 style="margin:0;font-size:2rem;font-weight:800;background:linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">
                        Clinical Disease AI Prediction System
                    </h1>
                    <p style="margin:4px 0 0 0;color:#cbd5e1;font-size:1rem;font-weight:500;">
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

    with tab1:
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
            st.query_params.clear()

        # Show banner if voice query was just captured
        if st.session_state.get("voice_banner_msg"):
            st.success(f"🎙️ **Spoken Symptoms Transferred:** \"{st.session_state['voice_banner_msg']}\"")

        col_input, col_presets = st.columns([2.2, 1.1])

        with col_presets:
            st.markdown("#### ⚡ Quick Presets")
            presets = {
                "Select a preset...": "",
                "Typhoid / Cold: Fever, cough, cold, body pain": "fever, cough, cold and bodypain with headache",
                "[ಕನ್ನಡ] ಜ್ವರ, ಕೆಮ್ಮು, ನೆಗಡಿ ಮತ್ತು ಮೈಕೈ ನೋವು": "ಜ್ವರ, ಕೆಮ್ಮು, ನೆಗಡಿ, ಮೈಕೈ ನೋವು ಮತ್ತು ತಲೆನೋವು",
                "Food Poisoning: Vomiting & loose motion": "severe vomiting, dehydration and loose motion",
                "[ಕನ್ನಡ] ವಾಂತಿ ಮತ್ತು ಭೇದಿ (Food Poisoning)": "ವಿಪರೀತ ವಾಂತಿ, ಭೇದಿ ಮತ್ತು ನಿರ್ಜಲೀಕರಣ",
                "Infection: High fever with chills & shivering": "high fever, violent shivering, chills and sweating",
                "[ಕನ್ನಡ] ಕಾಮಾಲೆ: ಹಳದಿ ಚರ್ಮ ಮತ್ತು ಕಣ್ಣುಗಳು": "ಹಳದಿ ಚರ್ಮ, ಗಾಢ ಹಳದಿ ಮೂತ್ರ, ಹಳದಿ ಕಣ್ಣು ಮತ್ತು ಸುಸ್ತು",
                "Skin: Itching & red skin rash": "itching, skin rash and nodal skin eruptions",
                "[ಕನ್ನಡ] ಚರ್ಮದ ತುರಿಕೆ ಮತ್ತು ದದ್ದುಗಳು": "ತುರಿಕೆ, ಚರ್ಮದ ದದ್ದುಗಳು ಮತ್ತು ಗಂಟು ಗುಳ್ಳೆಗಳು",
                "UTI: Burning urination & bladder pain": "burning urination, foul smell of urine and bladder discomfort",
                "[ಕನ್ನಡ] ಉರಿ ಮೂತ್ರ ಮತ್ತು ಹೊಟ್ಟೆ ನೋವು (UTI)": "ಉರಿ ಮೂತ್ರ, ಮೂತ್ರಕೋಶದ ನೋವು ಮತ್ತು ಅಸ್ವಸ್ಥತೆ",
                "Migraine: Throbbing headache & aura": "throbbing headache, visual disturbances and blurred vision",
                "[ಕನ್ನಡ] ಮೈಗ್ರೇನ್: ತಲೆನೋವು ಮತ್ತು ಕಣ್ಣು ಮಸುಕು": "ವಿಪರೀತ ತಲೆನೋವು, ಕಣ್ಣು ಮಸುಕು ಮತ್ತು ವಾಕರಿಕೆ",
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
            st.markdown("#### 🗣️ Enter or Speak Symptoms (ಕನ್ನಡ / English)")

            # Render Voice Input Widget (Microphone)
            render_voice_input_widget()

            # Initialize session state for text box if not present
            if "patient_symptoms_text_box" not in st.session_state:
                st.session_state["patient_symptoms_text_box"] = st.session_state.get("input_text", "fever, cough, cold and bodypain")

            symptom_query = st.text_area(
                "Patient Symptoms (Kannada / English • Spoken or Typed)",
                height=90,
                placeholder="E.g., I have fever, cough / ನನಗೆ ಜ್ವರ, ಕೆಮ್ಮು, ನೆಗಡಿ ಇದೆ / jwara, kemmu, tale novu...",
                help="Speak via the microphone button above in Kannada or English, or type symptoms here.",
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
                    {''.join([f'<span class="symptom-tag">✓ {s.replace("_", " ").title()}</span>' for s in matched_keys]) if matched_keys else '<span style="color:#94a3b8;font-size:0.9rem;">No clinical symptoms detected yet. Speak or type symptoms above (e.g. fever, headache, vomiting, loose motion, cold).</span>'}
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

        # Single Clean Predict Button (No extra stacked buttons)
        btn_predict = st.button("🔍 Predict Disease (XGBoost Analysis)", type="primary", use_container_width=True, key="btn_run_prediction_main")

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

                    # Assess Comprehensive Urgency (Enforces >=5-6 days See Doctor rule)
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

                    # Detect Kannada input (script / transliterated / voice) and compile Kannada report
                    is_kannada_input = detect_kannada_input(combined_query) or (st.session_state.get("voice_lang") == "kn-IN")
                    kn_report_data = get_kannada_report_data(
                        top_disease,
                        matched_keys,
                        urgency_level,
                        days=int(days),
                        severity=int(severity),
                        vuln_score=int(vuln_score)
                    )

                    # Save full report data and navigate to dedicated Results Page (Page 2)
                    st.session_state["report_data"] = {
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
                        "is_kannada_input": is_kannada_input,
                        "kn_report_data": kn_report_data,
                        "patient_snapshot": {
                            "age": int(age),
                            "gender": gender,
                            "days": int(days),
                            "severity": int(severity),
                            "conditions": existing_cond
                        }
                    }
                    st.session_state["report_lang_choice"] = "kannada" if is_kannada_input else "english"
                    st.session_state["current_page"] = "results"
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
            "Explore evidence-based home remedies, step-by-step recipe preparations, verified video tutorials (WebMD, Mayo Clinic, WHO), and clinical profiles across all major body systems and 30 clinical diseases."
        )

        # ---------------------------------------------------------
        # SECTION 1: ALL YOUTUBE VIDEO GUIDES AVAILABLE AT THE BEGINNING
        # ---------------------------------------------------------
        st.markdown(
            f"""
            <div style="background:linear-gradient(135deg, rgba(220,38,38,0.2) 0%, rgba(99,102,241,0.2) 100%);border:1.5px solid #ef4444;border-radius:16px;padding:16px 20px;margin:12px 0 16px 0;box-shadow:0 6px 20px rgba(0,0,0,0.3);">
                <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;">
                    <div>
                        <h3 style="margin:0;color:#f87171 !important;font-size:1.3rem;font-weight:800;display:flex;align-items:center;gap:10px;">
                            <span>▶️</span> <span>Verified Medical YouTube Video Guides & Remedy Index</span>
                        </h3>
                        <p style="margin:4px 0 0 0;color:#cbd5e1;font-size:0.92rem;">
                            Step-by-step video preparations from WebMD, Mayo Clinic, WHO, and Certified Doctors across {len(REMEDY_RECIPES)} clinical remedy recipes.
                        </p>
                    </div>
                    <span style="background:#dc2626;color:#ffffff;padding:4px 14px;border-radius:14px;font-size:0.82rem;font-weight:800;letter-spacing:0.5px;">
                        {len(REMEDY_RECIPES)} Verified YouTube Guides
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        yt_col1, yt_col2 = st.columns([2.2, 1.2])
        with yt_col1:
            yt_search_q = st.text_input(
                "🔍 Search YouTube Video Guides (by disease, symptom, recipe, or ingredient):",
                placeholder="E.g. Ginger Tea, Golden Milk, ORS, Cough, Cold, Headache, Nausea, Digestion...",
                key="yt_remedy_search_box"
            )
        with yt_col2:
            all_cats = ["All Categories"] + sorted(list(set(r.get("category", "") for r in REMEDY_RECIPES)))
            yt_cat_filter = st.selectbox("Filter Videos by Category:", all_cats, index=0, key="yt_remedy_category_filter")

        # Filter YouTube Recipes
        filtered_yt_recipes = []
        for r in REMEDY_RECIPES:
            match_cat = (yt_cat_filter == "All Categories") or (r.get("category") == yt_cat_filter)
            match_text = True
            if yt_search_q.strip():
                q = yt_search_q.lower()
                match_text = (
                    q in r.get("name", "").lower() or
                    q in r.get("category", "").lower() or
                    q in r.get("make", "").lower() or
                    q in r.get("why", "").lower() or
                    any(q in u.lower() for u in r.get("usedFor", []))
                )
            if match_cat and match_text:
                filtered_yt_recipes.append(r)

        st.caption(f"Showing **{len(filtered_yt_recipes)}** YouTube video tutorials:")

        # Render 2-Column Responsive Grid of All YouTube Video Remedies
        yt_grid_cols = st.columns(2)
        for idx, item in enumerate(filtered_yt_recipes):
            col_target = yt_grid_cols[idx % 2]
            with col_target:
                v_name = item.get("name", "Home Remedy")
                v_icon = item.get("icon", "🍵")
                v_cat = item.get("category", "Remedy")
                v_make = item.get("make", "")
                v_why = item.get("why", "")
                v_vid = item.get("video", "https://www.youtube.com")
                v_label = item.get("videoLabel", "Watch Medical Tutorial")
                v_used = item.get("usedFor", [])

                st.markdown(
                    f"""
                    <div class="glass-card" style="padding:16px;margin-bottom:14px;border:1.5px solid rgba(220,38,38,0.4);background:rgba(15,23,42,0.9);">
                        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
                            <div style="font-weight:800;color:#f8fafc;font-size:1.05rem;display:flex;align-items:center;gap:8px;">
                                <span>{v_icon}</span> <span>{v_name}</span>
                            </div>
                            <span style="font-size:0.75rem;background:rgba(56,189,248,0.18);color:#7dd3fc;padding:2px 10px;border-radius:10px;border:1px solid rgba(56,189,248,0.3);font-weight:700;">
                                {v_cat}
                            </span>
                        </div>
                        <div style="font-size:0.86rem;color:#cbd5e1;margin-bottom:6px;line-height:1.5;">
                            <b style="color:#38bdf8;">Preparation Guide:</b> {v_make}
                        </div>
                        <div style="font-size:0.83rem;color:#94a3b8;margin-bottom:10px;line-height:1.4;">
                            <b style="color:#34d399;">Biological Mechanism:</b> {v_why}
                        </div>
                        <div style="margin-bottom:12px;">
                            <span style="font-size:0.78rem;color:#94a3b8;font-weight:600;">Used for:</span>
                            {' '.join([f'<span class="symptom-tag" style="font-size:0.75rem;padding:2px 8px;">{u}</span>' for u in v_used])}
                        </div>
                        <div style="border-top:1px solid rgba(255,255,255,0.08);padding-top:10px;">
                            <a href="{v_vid}" target="_blank" style="background:#dc2626;color:#ffffff;padding:8px 18px;border-radius:20px;font-weight:800;font-size:0.85rem;text-decoration:none;display:inline-flex;align-items:center;gap:8px;box-shadow:0 4px 14px rgba(220,38,38,0.4);">
                                <span>▶️</span> <span>Watch on YouTube: {v_label} ↗</span>
                            </a>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        st.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,0.15);margin:24px 0 20px 0;'>", unsafe_allow_html=True)

        # ---------------------------------------------------------
        # SECTION 2: 30 CLINICAL DISEASES & CONDITION PROFILES
        # ---------------------------------------------------------
        st.markdown("### 📚 Comprehensive 30 Clinical Diseases & Condition Guides")

        guide_mode = st.radio(
            "Select Encyclopedia View:",
            ["🌿 OpenCare Interactive Condition Profiles", "📚 30 Clinical Disease Knowledge Corpus"],
            horizontal=True,
            key="encyclopedia_view_mode_toggle"
        )

        if "OpenCare" in guide_mode:
            cat_choice = st.selectbox("Filter by Body System / Category:", CATEGORIES_LIST, index=0, key="encyclopedia_category_select")

            filtered_guides = CONDITION_GUIDES
            if cat_choice != "All Categories":
                filtered_guides = [c for c in CONDITION_GUIDES if c.get("category") == cat_choice]

            st.caption(f"Showing **{len(filtered_guides)}** clinical condition guides under **{cat_choice}**:")

            for cond in filtered_guides:
                with st.expander(f"{cond.get('icon', '🩺')} {cond.get('name')} ({cond.get('category')})", expanded=False):
                    c_causes = cond.get('causes', 'Underlying physiological mechanisms.')
                    c_symptoms = cond.get('symptoms', 'Characteristic clinical symptoms.')
                    st.markdown(
                        f"""
                        <div style="background:rgba(15,23,42,0.6);border:1px solid rgba(56,189,248,0.2);border-radius:10px;padding:12px 14px;margin-bottom:12px;">
                            <div style="color:#7dd3fc;font-size:0.9rem;margin-bottom:6px;line-height:1.5;">
                                <b style="color:#38bdf8;">🔬 Pathophysiology & Root Causes:</b> {c_causes}
                            </div>
                            <div style="color:#cbd5e1;font-size:0.88rem;line-height:1.5;">
                                <b style="color:#34d399;">📋 Common Clinical Symptoms:</b> {c_symptoms}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                    st.markdown("##### 🌿 Evidence-Based Home Remedies & Video Guides:")
                    for r in cond.get("remedies", []):
                        r_vlink = r.get("video", "")
                        r_vlbl = r.get("videoLabel", "Watch Video Guide")
                        st.markdown(
                            f"""
                            <div class="glass-card" style="padding:12px 16px;margin-bottom:8px;border-left:4px solid #10b981;">
                                <div style="display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:6px;">
                                    <div style="font-weight:700;color:#34d399;font-size:0.95rem;">
                                        {r.get('icon', '🍵')} {r.get('name')}
                                    </div>
                                    {f'<a href="{r_vlink}" target="_blank" style="background:#dc2626;color:#ffffff;padding:4px 12px;border-radius:14px;font-size:0.75rem;font-weight:700;text-decoration:none;">▶️ {r_vlbl} ↗</a>' if r_vlink else ''}
                                </div>
                                <div style="font-size:0.88rem;color:#cbd5e1;margin-top:4px;">
                                    <b>Preparation & Dosage:</b> {r.get('make')}
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

        else:
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
