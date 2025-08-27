# train.py
# End-to-end training script for credit card fraud detection
# - Loads Kaggle "creditcard.csv" (target column: Class; 1=fraud)
# - Preprocesses (scales), handles imbalance (SMOTE only on training folds)
# - Compares LogisticRegression, RandomForest, XGBoost via GridSearchCV on PR-AUC
# - Chooses a probability threshold on a validation split to maximize F1
# - Evaluates on a held-out test set (ROC-AUC, PR-AUC, confusion matrix)
# - Saves the final pipeline + threshold with joblib

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve, confusion_matrix, classification_report
)
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from xgboost import XGBClassifier

RANDOM_STATE = 42
DATA_PATH = os.path.join("data", "creditcard.csv")
MODEL_PATH = "model.joblib"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    # Kaggle dataset: numeric features only (Time, V1..V28, Amount), target "Class"
    assert "Class" in df.columns, "Target column 'Class' not found"
    X = df.drop(columns=["Class"])
    y = df["Class"].astype(int)
    return X, y

def build_preprocessor(X):
    # All columns numeric in this dataset; RobustScaler is robust to outliers
    numeric_features = list(X.columns)
    preproc = ColumnTransformer(
        transformers=[("num", RobustScaler(), numeric_features)],
        remainder="drop"
    )
    return preproc, numeric_features

def compute_scale_pos_weight(y):
    # Used by XGBoost: ratio of negative / positive in training
    neg = (y == 0).sum()
    pos = (y == 1).sum()
    return max(1.0, neg / max(1, pos))  # avoid div-by-zero

def make_param_grids(scale_pos_weight):
    # Each dict is a separate grid with a distinct clf
    grids = [
        {
            "clf": [LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear")],
            "clf__C": [0.1, 0.5, 1.0, 2.0]
        },
        {
            "clf": [RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)],
            "clf__n_estimators": [300, 600],
            "clf__max_depth": [None, 12, 20],
            "clf__min_samples_leaf": [1, 3]
        },
        {
            "clf": [XGBClassifier(
                objective="binary:logistic",
                eval_metric="aucpr",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                scale_pos_weight=scale_pos_weight
            )],
            "clf__n_estimators": [400, 800],
            "clf__max_depth": [4, 8],
            "clf__learning_rate": [0.05, 0.1],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0],
        }
    ]
    return grids

def choose_threshold(y_true, y_prob, strategy="max_f1"):
    # Tune decision threshold on validation set
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    thresholds = np.append(thresholds, 1.0)  # align lengths
    if strategy == "max_f1":
        f1s = (2 * precisions * recalls) / (precisions + recalls + 1e-12)
        best_idx = np.nanargmax(f1s)
        return float(thresholds[best_idx]), float(precisions[best_idx]), float(recalls[best_idx]), float(f1s[best_idx])
    # fallback: 0.5
    return 0.5, None, None, None

def plot_roc_pr(y_true, y_prob, prefix="test"):
    # ROC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = roc_auc_score(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC AUC={roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve ({prefix})")
    plt.legend(loc="lower right")
    roc_path = os.path.join(PLOTS_DIR, f"roc_{prefix}.png")
    plt.savefig(roc_path, bbox_inches="tight")
    plt.close()

    # PR
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision, label=f"AP={ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall Curve ({prefix})")
    plt.legend(loc="lower left")
    pr_path = os.path.join(PLOTS_DIR, f"pr_{prefix}.png")
    plt.savefig(pr_path, bbox_inches="tight")
    plt.close()
    return roc_path, pr_path

def main():
    print("Loading data...")
    X, y = load_data()

    # 60/20/20 split (train/val/test), stratified for class balance
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=RANDOM_STATE, stratify=y_temp
    )  # => 60/20/20 overall

    print("Building preprocessors...")
    preproc, feature_names = build_preprocessor(X_train)

    print("Preparing pipeline with SMOTE (train-only)...")
    base_pipeline = Pipeline(
        steps=[
            ("preprocess", preproc),
            ("smote", SMOTE(random_state=RANDOM_STATE, sampling_strategy=1.0)),
            ("clf", LogisticRegression())  # placeholder replaced by GridSearch
        ]
    )

    # Scale_pos_weight for XGBoost from *training* distribution only
    spw = compute_scale_pos_weight(y_train)
    param_grids = make_param_grids(spw)

    print("Running model selection with GridSearchCV (scoring=average_precision)...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        estimator=base_pipeline,
        param_grid=param_grids,
        scoring="average_precision",   # PR-AUC better for heavy imbalance
        n_jobs=-1,
        cv=cv,
        verbose=1,
        refit=True,
    )
    grid.fit(X_train, y_train)
    print(f"Best CV PR-AUC: {grid.best_score_:.6f}")
    print("Best params:", grid.best_params_)

    best_pipeline = grid.best_estimator_

    # Validate and choose threshold on the *validation* set to avoid test leakage
    print("Choosing probability threshold on validation set (maximize F1)...")
    val_probs = best_pipeline.predict_proba(X_val)[:, 1]
    thr, p, r, f1 = choose_threshold(y_val, val_probs, "max_f1")
    ap_val = average_precision_score(y_val, val_probs)
    roc_val = roc_auc_score(y_val, val_probs)
    print(f"Val PR-AUC: {ap_val:.6f} | ROC-AUC: {roc_val:.6f}")
    print(f"Chosen threshold: {thr:.4f} (val Precision={p:.4f}, Recall={r:.4f}, F1={f1:.4f})")

    # Retrain on train+val using same hyperparams (not the threshold — that’s external)
    print("Refitting best pipeline on train+val...")
    X_trval = pd.concat([X_train, X_val], axis=0)
    y_trval = pd.concat([y_train, y_val], axis=0)
    best_pipeline.fit(X_trval, y_trval)

    # Final evaluation on the held-out test set
    print("Evaluating on test set...")
    test_probs = best_pipeline.predict_proba(X_test)[:, 1]
    ap_test = average_precision_score(y_test, test_probs)
    roc_test = roc_auc_score(y_test, test_probs)
    print(f"Test PR-AUC: {ap_test:.6f} | ROC-AUC: {roc_test:.6f}")

    # Plot curves
    roc_path, pr_path = plot_roc_pr(y_test, test_probs, prefix="test")
    print(f"Saved plots:\n  {roc_path}\n  {pr_path}")

    # Apply chosen threshold to test and print confusion matrix / report
    test_preds = (test_probs >= thr).astype(int)
    cm = confusion_matrix(y_test, test_preds)
    print("Confusion Matrix (test):\n", cm)
    print("Classification Report (test):\n", classification_report(y_test, test_preds, digits=4))

    # Persist pipeline and threshold
    artifact = {
        "pipeline": best_pipeline,
        "threshold": thr,
        "features": feature_names,
        "metadata": {
            "scoring": "average_precision",
            "random_state": RANDOM_STATE
        }
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
