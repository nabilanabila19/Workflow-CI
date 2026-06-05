"""
modelling.py (MLflow Project version — Kriteria 3)
===================================================
Fire Risk Prediction — Training untuk MLflow Project + GitHub Actions CI
Sumatera Selatan, Indonesia (2019–2024)

Model    : Random Forest Classifier
Tracking : MLflow lokal (mlruns/)
Logging  : Manual logging + artefak

Usage:
    python modelling.py
    python modelling.py --data_path fire_risk_dataset_preprocessing/fire_risk_preprocessed.csv
    mlflow run . -P data_path=fire_risk_dataset_preprocessing/fire_risk_preprocessed.csv
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    ConfusionMatrixDisplay
)
from sklearn.model_selection import train_test_split, cross_val_score

import mlflow
import mlflow.sklearn

warnings.filterwarnings('ignore')


# KONFIGURASI
EXPERIMENT_NAME = "sumselfire-fire-risk-ci"
TARGET_COL      = "fire_risk"
RANDOM_STATE    = 42
TEST_SIZE       = 0.2


# LOAD DATA
def load_data(path: str):
    """Load dataset preprocessed dan split fitur/target."""
    df = pd.read_csv(path)
    print(f"\n Dataset loaded: {df.shape[0]} baris, {df.shape[1]} kolom")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"   Train : {len(X_train)} baris")
    print(f"   Test  : {len(X_test)} baris")
    print(f"   Fitur : {X.columns.tolist()}")
    return X_train, X_test, y_train, y_test, X.columns.tolist()


# ARTEFAK 1 — Confusion Matrix Plot
def plot_confusion_matrix(y_test, y_pred, save_path: str):
    """Buat dan simpan confusion matrix sebagai artefak."""
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=['Low Risk', 'High Risk'])
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title('Confusion Matrix — Fire Risk Prediction (CI)',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save_path}")


# ARTEFAK 2 — Feature Importance Plot
def plot_feature_importance(model, feature_names: list, save_path: str):
    """Buat dan simpan feature importance plot sebagai artefak."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ['#D85A30' if imp > np.mean(sorted_importances) else '#5DCAA5'
              for imp in sorted_importances]
    ax.barh(sorted_features[::-1], sorted_importances[::-1],
            color=colors[::-1], edgecolor='white', alpha=0.85)
    ax.set_title('Feature Importance — Random Forest (CI)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Importance Score')
    ax.axvline(np.mean(sorted_importances), color='navy', linestyle='--',
               linewidth=1.2, label=f'Mean ({np.mean(sorted_importances):.3f})')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {save_path}")


# MAIN — Training + MLflow Logging
def train_and_log(data_path: str):
    
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f" MLflow tracking lokal")
    print(f"   Experiment: {EXPERIMENT_NAME}")

    X_train, X_test, y_train, y_test, feature_names = load_data(data_path)

    params = {
        "n_estimators"     : 100,
        "max_depth"        : 3,
        "min_samples_split": 2,
        "min_samples_leaf" : 2,
        "random_state"     : RANDOM_STATE
    }

    print(f"\n Memulai training Random Forest...")
    print(f"   Parameter: {params}")

    run = mlflow.active_run()
    if run is None:
        mlflow.start_run(run_name="rf_ci_run")
        run = mlflow.active_run()

    # Train model
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    # Hitung metrics
    accuracy  = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_pred_prob)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_mean   = cv_scores.mean()
    cv_std    = cv_scores.std()

    print(f"\n Hasil Evaluasi:")
    print(f"   Accuracy   : {accuracy:.4f}")
    print(f"   Precision  : {precision:.4f}")
    print(f"   Recall     : {recall:.4f}")
    print(f"   F1-Score   : {f1:.4f}")
    print(f"   ROC-AUC    : {roc_auc:.4f}")
    print(f"   CV Accuracy: {cv_mean:.4f} ± {cv_std:.4f}")

    # Manual Logging — Parameters
    mlflow.log_param("model_type",         "RandomForestClassifier")
    mlflow.log_param("n_estimators",        params["n_estimators"])
    mlflow.log_param("max_depth",           str(params["max_depth"]))
    mlflow.log_param("min_samples_split",   params["min_samples_split"])
    mlflow.log_param("min_samples_leaf",    params["min_samples_leaf"])
    mlflow.log_param("random_state",        params["random_state"])
    mlflow.log_param("test_size",           TEST_SIZE)
    mlflow.log_param("train_size",          len(X_train))
    mlflow.log_param("test_size_rows",      len(X_test))
    mlflow.log_param("n_features",          len(feature_names))
    mlflow.log_param("data_path",           data_path)

    # Manual Logging — Metrics
    mlflow.log_metric("accuracy",           accuracy)
    mlflow.log_metric("precision",          precision)
    mlflow.log_metric("recall",             recall)
    mlflow.log_metric("f1_score",           f1)
    mlflow.log_metric("roc_auc",            roc_auc)
    mlflow.log_metric("cv_accuracy_mean",   cv_mean)
    mlflow.log_metric("cv_accuracy_std",    cv_std)

    # Tags
    mlflow.set_tag("dataset",    "SumselFire 2019-2024")
    mlflow.set_tag("task",       "fire_risk_classification")
    mlflow.set_tag("model",      "RandomForest")
    mlflow.set_tag("triggered",  "GitHub Actions CI")

    # Artefak 1 — Confusion Matrix
    print("\n Menyimpan artefak...")
    cm_path = "confusion_matrix.png"
    plot_confusion_matrix(y_test, y_pred, cm_path)
    mlflow.log_artifact(cm_path)

    # Artefak 2 — Feature Importance
    fi_path = "feature_importance.png"
    plot_feature_importance(model, feature_names, fi_path)
    mlflow.log_artifact(fi_path)

    # Artefak 3 — Classification Report
    report = classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk'])
    report_path = "classification_report.txt"
    with open(report_path, 'w') as f:
        f.write("=== Classification Report (CI Run) ===\n\n")
        f.write(report)
    mlflow.log_artifact(report_path)

    # Log model
    mlflow.sklearn.log_model(
        sk_model      = model,
        artifact_path = "random_forest_model"
    )

    print(f"\n Training selesai!")
    print(f"   Run ID   : {run.info.run_id}")
    print(f"   Artifact : confusion_matrix.png, feature_importance.png, classification_report.txt")

    return model


# CLI Entry Point
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='SumselFire MLflow Project — CI Training'
    )
    parser.add_argument(
        '--data_path',
        type=str,
        default='fire_risk_dataset_preprocessing/fire_risk_preprocessed.csv',
        help='Path ke file fire_risk_preprocessed.csv'
    )
    args = parser.parse_args()
    train_and_log(data_path=args.data_path)
