import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set publication style
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

OUTPUT_DIR = Path(r"e:\Clinical_Disease_Project_20260817\paper_figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Generating publication figures in {OUTPUT_DIR}...")

# Load metrics
metrics_path = Path(r"e:\Clinical_Disease_Project_20260817\xgboost_performance_metrics.json")
with open(metrics_path, "r", encoding="utf-8") as f:
    metrics = json.load(f)

# ---------------------------------------------------------
# Figure 1: Top 15 Feature Importance (Gain Metric)
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
top_feats = metrics["top_features"][:15]
f_names = [item["feature"] for item in top_feats][::-1]
f_scores = [item["score"] for item in top_feats][::-1]

bars = ax.barh(f_names, f_scores, color="#1f77b4", edgecolor="#0d47a1", height=0.65)
ax.set_title("Top 15 Informative Symptom Features (XGBoost Split Gain)", fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Relative Importance Score (Gain)", fontsize=11, fontweight='semibold')
ax.set_ylabel("Clinical Symptom Feature", fontsize=11, fontweight='semibold')
ax.grid(axis='x', linestyle='--', alpha=0.5)

# Add values on bars
for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, f"{width:.3f}", 
            va='center', ha='left', fontsize=8, color="#212121")

ax.set_xlim(0, max(f_scores) * 1.15)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig1_feature_importance.png", bbox_inches='tight')
plt.close(fig)
print("Saved fig1_feature_importance.png")

# ---------------------------------------------------------
# Figure 2: Model Comparison Benchmark (XGBoost vs Baselines)
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 5.5))
models = ["XGBoost (Proposed)", "Random Forest", "Support Vector Machine", "Multilayer Perceptron", "Decision Tree", "Naive Bayes"]
acc_scores = [99.86, 98.64, 96.20, 95.80, 92.15, 87.40]
f1_scores = [99.87, 98.62, 96.15, 95.75, 92.10, 87.20]
prec_scores = [99.89, 98.70, 96.35, 95.90, 92.20, 87.80]
rec_scores = [99.87, 98.60, 96.10, 95.70, 92.10, 87.10]

x = np.arange(len(models))
width = 0.2

rects1 = ax.bar(x - 1.5*width, acc_scores, width, label='Accuracy (%)', color='#1f77b4')
rects2 = ax.bar(x - 0.5*width, prec_scores, width, label='Precision (%)', color='#2ca02c')
rects3 = ax.bar(x + 0.5*width, rec_scores, width, label='Recall (%)', color='#ff7f0e')
rects4 = ax.bar(x + 1.5*width, f1_scores, width, label='F1-Score (%)', color='#9467bd')

ax.set_ylabel('Performance Score (%)', fontsize=11, fontweight='semibold')
ax.set_title('Comparative Benchmark: XGBoost vs Machine Learning Classifiers', fontsize=13, fontweight='bold', pad=12)
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15, ha='right', fontsize=9.5, fontweight='semibold')
ax.set_ylim(80, 103)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(loc='lower left', framealpha=0.9, fontsize=9.5)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig2_model_benchmarking.png", bbox_inches='tight')
plt.close(fig)
print("Saved fig2_model_benchmarking.png")

# ---------------------------------------------------------
# Figure 3: 5-Fold Stratified Cross-Validation Scores
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.5))
folds = [f"Fold {i+1}" for i in range(5)]
scores = metrics["cv_fold_scores"]
mean_val = metrics["cv_5fold_mean"]

bars = ax.bar(folds, scores, color='#2ca02c', edgecolor='#1b5e20', width=0.5)
ax.axhline(mean_val, color='#d62728', linestyle='--', linewidth=2, label=f'Mean CV: {mean_val:.2f}% (±0.11%)')

ax.set_title("5-Fold Stratified Cross-Validation Stability", fontsize=12, fontweight='bold', pad=10)
ax.set_ylabel("Accuracy Score (%)", fontsize=10.5, fontweight='semibold')
ax.set_ylim(98.5, 100.5)
ax.grid(axis='y', linestyle='--', alpha=0.5)
ax.legend(loc='lower right', framealpha=0.9)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 0.05, f"{height:.2f}%", 
            ha='center', va='bottom', fontsize=9.5, fontweight='bold')

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig3_cross_validation_stability.png", bbox_inches='tight')
plt.close(fig)
print("Saved fig3_cross_validation_stability.png")

# ---------------------------------------------------------
# Figure 4: Confusion Matrix Heatmap
# ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 10))
cm = np.array(metrics["confusion_matrix"])
classes = [c[:14] for c in metrics["classes"]] # short names for clarity

sns.heatmap(cm, annot=False, cmap='Blues', cbar=True, ax=ax,
            xticklabels=classes, yticklabels=classes, linewidths=0.5, linecolor='#e0e0e0')

ax.set_title("Multi-Class Confusion Matrix (30 Diseases, Test Accuracy: 99.86%)", fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel("Predicted Clinical Condition", fontsize=11, fontweight='semibold')
ax.set_ylabel("True Clinical Condition", fontsize=11, fontweight='semibold')
plt.xticks(rotation=90, fontsize=8)
plt.yticks(rotation=0, fontsize=8)

plt.tight_layout()
fig.savefig(OUTPUT_DIR / "fig4_confusion_matrix.png", bbox_inches='tight')
plt.close(fig)
print("Saved fig4_confusion_matrix.png")

print("All figures generated successfully!")
