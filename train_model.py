"""
XGBoost Clinical Machine Learning Training & Mathematical Benchmarking Pipeline.
Trains the best-in-class XGBoost Classifier on the clinical disease dataset
and calculates comprehensive mathematical metrics, confusion matrices, and feature importances.
"""

import os
import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier

from clinical_knowledge import DISEASE_NAME_MAP

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "day_to_day_clinical_disease_dataset.csv"
METRICS_PATH = BASE_DIR / "xgboost_performance_metrics.json"


def load_and_preprocess_data():
    print(f"[*] Loading dataset from: {DATASET_PATH}")
    df = pd.read_csv(DATASET_PATH)

    # Standardize disease labels
    df["Disease"] = df["Disease"].astype(str).str.strip().map(lambda d: DISEASE_NAME_MAP.get(d, d))

    feature_cols = [c for c in df.columns if c.startswith("has_")]
    print(f"[*] Dataset Shape: {df.shape}")
    print(f"[*] Found {len(feature_cols)} symptom binary features.")
    print(f"[*] Number of unique clinical diseases: {df['Disease'].nunique()}")

    X = df[feature_cols].copy()
    y_raw = df["Disease"].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)

    return df, X, y_enc, le, feature_cols


def train_and_evaluate_xgboost(X, y_enc, le, feature_cols):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    print("\n[*] Training XGBoost Classifier (n_estimators=300, max_depth=6, lr=0.08)...")
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1
    )

    t0 = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    # Predictions
    y_train_pred = model.predict(X_train)
    
    t_infer_start = time.time()
    y_test_pred = model.predict(X_test)
    infer_time_ms = ((time.time() - t_infer_start) / len(X_test)) * 100 * 1000

    y_test_proba = model.predict_proba(X_test)
    loss_val = float(log_loss(y_test, y_test_proba, labels=np.arange(len(le.classes_))))

    train_acc = float(accuracy_score(y_train, y_train_pred))
    test_acc = float(accuracy_score(y_test, y_test_pred))

    prec_macro = float(precision_score(y_test, y_test_pred, average="macro", zero_division=0))
    prec_weighted = float(precision_score(y_test, y_test_pred, average="weighted", zero_division=0))
    rec_macro = float(recall_score(y_test, y_test_pred, average="macro", zero_division=0))
    rec_weighted = float(recall_score(y_test, y_test_pred, average="weighted", zero_division=0))
    f1_macro = float(f1_score(y_test, y_test_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_test_pred, average="weighted", zero_division=0))

    # 5-Fold Stratified Cross-Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y_enc, cv=cv, scoring="accuracy", n_jobs=-1)
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_test_pred).tolist()

    # Classification Report
    clf_report = classification_report(y_test, y_test_pred, target_names=le.classes_, output_dict=True, zero_division=0)

    # Top Feature Importances (Gini Gain)
    importances = model.feature_importances_
    top_indices = np.argsort(importances)[::-1][:20]
    top_features = [
        {"feature": feature_cols[i].replace("has_", "").replace("_", " ").title(), "score": float(importances[i])}
        for i in top_indices if importances[i] > 0
    ]

    metrics = {
        "model_name": "XGBoost Classifier",
        "train_accuracy": round(train_acc * 100, 2),
        "test_accuracy": round(test_acc * 100, 2),
        "precision_macro": round(prec_macro * 100, 2),
        "precision_weighted": round(prec_weighted * 100, 2),
        "recall_macro": round(rec_macro * 100, 2),
        "recall_weighted": round(rec_weighted * 100, 2),
        "f1_macro": round(f1_macro * 100, 2),
        "f1_weighted": round(f1_weighted * 100, 2),
        "cv_5fold_mean": round(cv_mean * 100, 2),
        "cv_5fold_std": round(cv_std * 100, 2),
        "cv_fold_scores": [round(float(s) * 100, 2) for s in cv_scores],
        "log_loss": round(loss_val, 4),
        "train_time_sec": round(train_time, 3),
        "inference_latency_ms_per_100": round(infer_time_ms, 2),
        "top_features": top_features,
        "classes": le.classes_.tolist(),
        "total_samples": len(X),
        "feature_count": len(feature_cols),
        "confusion_matrix": cm,
        "classification_report": clf_report
    }

    print("="*75)
    print(f"XGBoost Test Accuracy: {test_acc*100:.2f}%")
    print(f"5-Fold Cross-Validation: {cv_mean*100:.2f}% ± {cv_std*100:.2f}%")
    print(f"Macro F1-Score: {f1_macro*100:.2f}%")
    print(f"Multi-Class Log Loss: {loss_val:.4f}")
    print("="*75)

    return model, le, feature_cols, metrics


def save_artifacts(model, le, feature_cols, metrics):
    joblib.dump(model, BASE_DIR / "disease_xgb_model.joblib")
    joblib.dump(le, BASE_DIR / "disease_label_encoder.joblib")
    joblib.dump(feature_cols, BASE_DIR / "symptom_feature_cols.joblib")

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[SUCCESS] Exported disease_xgb_model.joblib and {METRICS_PATH.name}")


if __name__ == "__main__":
    df, X, y_enc, le, feature_cols = load_and_preprocess_data()
    model, le, feature_cols, metrics = train_and_evaluate_xgboost(X, y_enc, le, feature_cols)
    save_artifacts(model, le, feature_cols, metrics)
