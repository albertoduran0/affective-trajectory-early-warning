"""SHAP feature attribution analysis for ATEW-XGB model."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

import importlib.util

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from src.data_loader import build_panel
from src.features import build_features
from src.output_utils import get_output_dirs
import torch
import random
import os

_spec = importlib.util.spec_from_file_location("run_sota_comp", ROOT_DIR / "scripts" / "run_sota_comparison.py")
_sota = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sota)
_tune_atew_pipeline = _sota._tune_atew_pipeline

def set_deterministic_seed(seed=2026):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_deterministic_seed(2026)


def clean_feature_name(name):
    if name.startswith("P") and len(name) == 2:
        return f"Practice {name[1]}"
    if name == "test1": return "Test 1"
    if name == "test2": return "Test 2"
    
    parts = name.split("_")
    if parts[0] in ["satisfaction", "effort", "motivation", "difficulty", "workload", "frustration"]:
        base = parts[0].capitalize()
        if "ewta_slope" in name:
            return f"{base} (EWTA Linear Trend)"
        if "ewta_std" in name:
            return f"{base} (EWTA Std)"
        if "ewta_mean" in name:
            return f"{base} (EWTA Mean)"
        if "recency_delta" in name:
            return f"{base} (Recency Shift)"
        if "trend" in name:
            return f"{base} (Linear Trend)"
        if "mean" in name:
            return f"{base} (Mean)"
        if "count" in name:
            if "high" in name:
                return f"{base} (High Count)"
            return f"{base} (Count)"
        if "shift" in name:
            return f"{base} (Shift)"
            
    if name == "attendance_rate": return "Attendance Rate"
    if name == "response_rate": return "Response Rate"
    if name == "weeks_responded": return "Weeks Responded"
    if name == "attendance_total": return "Attendance Total"
    if name == "attendance_decay": return "Attendance Decay"
    if name == "frustration_streak": return "Frustration Streak"
    if name == "diff_mot_gap_mean": return "Diff-Mot Gap (Mean)"
    if name == "diff_mot_gap_trend": return "Diff-Mot Gap (Linear Trend)"
    if name == "diff_mot_gap_ewta": return "Diff-Mot Gap (EWTA Mean)"
    if name == "diff_mot_gap_ewta_slope": return "Diff-Mot Gap (EWTA Linear Trend)"
    if name == "engagement_decay": return "Engagement Decay"
    if name == "engagement_ewta_decay": return "Engagement (EWTA Decay)"
    
    return name

def main():
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("XGBoost not installed. Skipping SHAP analysis.")
        return

    data = build_panel()
    surveys = data["surveys_long"]
    attendance = data["attendance_long"]
    grades = data["grades"]

    # LOCO Fold A: train on Year1 with nested tuning + RFECV, explain on Year2
    feat_df_init = build_features(surveys, attendance, grades, max_week=11)
    feat_df_init = feat_df_init[feat_df_init["final_grade"].notna()].reset_index(drop=True)
    train_mask_series = pd.Series((feat_df_init["year"] == "Year1").values, index=feat_df_init.index)

    best_alpha, best_params, selected_cols = _tune_atew_pipeline(
        surveys, attendance, grades, train_mask_series, max_week=11
    )

    feat_df = build_features(surveys, attendance, grades, max_week=11, ewta_alpha=best_alpha)
    feat_df = feat_df[feat_df["final_grade"].notna()].reset_index(drop=True)

    train_mask = (feat_df["year"] == "Year1").values
    test_mask  = (feat_df["year"] == "Year2").values

    cols = [c for c in selected_cols if c in feat_df.columns]
    cols = [c for c in cols if not feat_df.loc[train_mask, c].isna().all()]

    X = feat_df[cols]
    y = feat_df["final_grade"].values

    imputer = SimpleImputer(strategy="median")
    X_tr = pd.DataFrame(imputer.fit_transform(X[train_mask]), columns=cols)
    X_te = pd.DataFrame(imputer.transform(X[test_mask]), columns=cols)
    y_tr = y[train_mask]

    clean_cols = [clean_feature_name(c) for c in cols]
    X_tr.columns = clean_cols
    X_te.columns = clean_cols

    model = XGBRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=best_params["l1"], reg_lambda=best_params["l2"],
        random_state=2026, verbosity=0
    )
    model.fit(X_tr, y_tr)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_te)

    tables_dir, fig_dir = get_output_dirs(ROOT_DIR)

    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_te, show=False, max_display=15)
    plt.title("SHAP Summary Plot (Top 15 Features — Year2 Test Set)", fontweight="bold")
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_summary.png", dpi=600, bbox_inches="tight")
    plt.close()

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx = np.argsort(mean_abs_shap)[::-1]
    top_features = [clean_cols[i] for i in top_idx[:20]]
    with open(tables_dir / "shap_top_features.txt", "w") as f:
        for feat in top_features:
            f.write(f"{feat}\n")

    print(f"SHAP results saved to {fig_dir} and {tables_dir}")

if __name__ == "__main__":
    main()
