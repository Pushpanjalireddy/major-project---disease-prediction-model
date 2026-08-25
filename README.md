# 🩺 Clinical Disease AI Prediction & Triage System (XGBoost)

An intelligent, multi-modal clinical decision support system combining **Extreme Gradient Boosting (XGBoost)** with **real-time multilingual voice input NLP**, Bayesian comorbidity risk assessment, 4-tier clinical triage, and comprehensive **evidence-based home remedies & traditional Ayurvedic care plans**.

---

## 🌟 Key Features

- **⚡ High-Accuracy AI Engine**: Powered by an optimized XGBoost Classifier achieving **99.86% test accuracy**, **99.87% Macro F1-score**, and **99.78% (±0.11%) 5-fold cross-validation accuracy** across 30 primary disease classes and 132 validated clinical symptoms.
- **🎙️ Real-Time Voice & Speech Input**: Integrated speech-to-text recognition supporting **English (US/Global)**, **English (India)**, and **Hindi (हिन्दी)** with automatic symptom box population and instant auto-prediction.
- **📁 Audio File Transcription**: Drag-and-drop support for `.wav`, `.mp3`, `.m4a`, `.ogg`, and `.flac` voice notes.
- **🧠 Multi-Factor Bayesian Triage**: Incorporates patient demographics (Age, Gender), symptom timeline (Acute vs. Subacute vs. Chronic), discomfort severity ($1–10$), and pre-existing comorbidities (Diabetes, Hypertension, Asthma, etc.) into a dynamic **Patient Vulnerability Index ($0–100$)**.
- **🚨 Emergency Red Flag Detection**: Automatic alerts and hospital referral triggers for critical warning symptoms (chest pain, severe breathlessness, slurred speech, balance loss).
- **🌿 Evidence-Based Home Remedies & Ayurvedic Care**: Actionable self-care protocols, classical Ayurvedic Kadha and herbal preparations, categorized dietary rules (Foods to Eat vs. Foods to Avoid), and step-by-step preparation recipes with verified medical video tutorials.
- **🔊 Instant Text-to-Speech (TTS)**: One-click audio readout of diagnosis, urgency level, and care plans.

---

## 🏗️ Tech Stack

- **Frontend / UI**: [Streamlit](https://streamlit.io/) (Dark Mode Healthcare Theme)
- **Machine Learning**: [XGBoost](https://xgboost.readthedocs.io/), [Scikit-learn](https://scikit-learn.org/), [Joblib](https://joblib.readthedocs.io/)
- **Voice & Audio Processing**: `streamlit-mic-recorder`, `SpeechRecognition`, `PyAudio`, `gTTS`
- **Data & Visualizations**: [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [Matplotlib](https://matplotlib.org/), [Seaborn](https://seaborn.pydata.org/)

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Pushpanjalireddy/major-project---disease-prediction-model.git
cd major-project---disease-prediction-model
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
python -m streamlit run app.py
```
or on Windows, double-click **`run_app.bat`**.

---

## 📊 Model Performance Metrics

| Metric | Measured Value |
| :--- | :---: |
| **Test Accuracy** | **99.86%** |
| **5-Fold Cross-Validation** | **99.78% ± 0.11%** |
| **Macro Precision** | **99.89%** |
| **Macro Recall** | **99.87%** |
| **Macro F1-Score** | **99.87%** |
| **Multi-Class Log-Loss** | **0.0087** |
| **Inference Latency** | **5.15 ms / 100 queries** |

---

## 📁 Repository Structure

```
├── app.py                                   # Streamlit Web Application & UI
├── clinical_knowledge.py                    # Clinical Knowledge Base, NLP Parser & Triage Engine
├── home_remedies_guide.py                   # Evidence-based remedies, recipes & video tutorials
├── train_model.py                           # Model training & cross-validation script
├── export_disease_guide.py                  # Disease encyclopedia exporter
├── generate_paper_figures.py                # High-res publication figure generator
├── day_to_day_clinical_disease_dataset.csv  # 3,679 multi-parameter clinical diagnostic dataset
├── disease_xgb_model.joblib                 # Trained XGBoost classifier
├── disease_label_encoder.joblib             # Label encoder for 30 disease classes
├── symptom_feature_cols.joblib              # 132-symptom feature columns
├── xgboost_performance_metrics.json         # Performance benchmarks and confusion matrix
├── requirements.txt                         # Python dependencies
├── run_app.bat                              # One-click Windows application launcher
└── paper_figures/                           # High-resolution benchmark figures
```

---

## 🔒 Disclaimer
*This application is an educational and clinical decision-support tool. It is not intended to replace professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified physician for acute medical conditions.*
