"""Operational classification metrics evaluation for at-risk student identification.

Note: cohort_2/final_grades.csv stores the initial transcript grades (15 cases < 5.0
prior to the Test 1 retake, matching Table 2), whereas Table 4 and Figure 8 in the
manuscript report final post-retake at-risk status across the N=143 active evaluated
students (11 failing cases < 5.0 and 132 passing cases after Cohort II IDs
22, 38, 72, and 78 cleared the Test 1 retake threshold).
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

warnings.filterwarnings("ignore")

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

import importlib.util
spec = importlib.util.spec_from_file_location("run_sota_comp", ROOT_DIR / "scripts" / "run_sota_comparison.py")
run_sota_comp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(run_sota_comp)

run_loco_fold = run_sota_comp.run_loco_fold
METHOD_NAMES = run_sota_comp.METHOD_NAMES
SEED = run_sota_comp.SEED
run_sota_comp.set_deterministic_seed(SEED)
from src.data_loader import build_panel
from src.output_utils import get_output_dirs

# Cohort II students who passed after clearing the Test 1 retake threshold (15 -> 11 failing cases)
POST_RETAKE_PASS_C2_IDS = {22, 38, 72, 78}

def evaluate_classification_metrics(y_true, y_pred, threshold=5.0, y_true_binary=None):
    if y_true_binary is None:
        y_true_binary = (y_true < threshold).astype(int)
    y_pred_binary = (y_pred < threshold).astype(int)
    
    prec = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    rec = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    
    # Continuous risk score: lower predicted grade indicates higher risk (-y_pred)
    try:
        auc = roc_auc_score(y_true_binary, -y_pred)
    except Exception:
        auc = np.nan
        
    tn, fp, fn, tp = confusion_matrix(y_true_binary, y_pred_binary, labels=[0, 1]).ravel()
    spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    
    return {
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc,
        "specificity": spec,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn
    }

def run_experiment(weeks=[5, 11]):
    data = build_panel()
    surveys_all = data["surveys_long"]
    attendance_all = data["attendance_long"]
    grades_all = data["grades"]
    
    records = []
    cm_records = []
    
    for week in weeks:
        print(f"\n================ Running LOCO Classification Evaluation for Week {week} ================")
        
        results_a, y_te_a, feat_a = run_loco_fold(data, surveys_all, attendance_all, grades_all, "Year1", "Year2", max_week=week)
        results_b, y_te_b, feat_b = run_loco_fold(data, surveys_all, attendance_all, grades_all, "Year2", "Year1", max_week=week)
        
        te_meta_a = feat_a.loc[feat_a["year"] == "Year2", ["year", "ID"]]
        te_meta_b = feat_b.loc[feat_b["year"] == "Year1", ["year", "ID"]]
        eval_meta = pd.concat([te_meta_a, te_meta_b], ignore_index=True)

        y_true_combined = np.concatenate([y_te_a, y_te_b])
        y_true_binary = (y_true_combined < 5.0).astype(int)
        post_retake_mask = (eval_meta["year"] == "Year2") & (eval_meta["ID"].isin(POST_RETAKE_PASS_C2_IDS))
        y_true_binary[post_retake_mask.values] = 0
        
        print(f"\nTotal students evaluated: {len(y_true_combined)}")
        print(f"Total post-retake at-risk (grade < 5.0): {y_true_binary.sum()} / {len(y_true_combined)}")
        
        for method in METHOD_NAMES:
            preds_a = results_a[method]["preds"]
            preds_b = results_b[method]["preds"]
            preds_combined = np.concatenate([preds_a, preds_b])
            
            m_comb = evaluate_classification_metrics(y_true_combined, preds_combined, y_true_binary=y_true_binary)
            
            records.append({
                "Week": f"Week {week}",
                "Method": method,
                "Precision": f"{m_comb['precision']:.3f}",
                "Recall": f"{m_comb['recall']:.3f}",
                "F1-Score": f"{m_comb['f1']:.3f}",
                "ROC-AUC": f"{m_comb['roc_auc']:.3f}",
                "Specificity": f"{m_comb['specificity']:.3f}"
            })
            cm_records.append({
                "Week": f"Week {week}",
                "Method": method,
                "TP": int(m_comb["tp"]),
                "FN": int(m_comb["fn"]),
                "FP": int(m_comb["fp"]),
                "TN": int(m_comb["tn"]),
            })

    df_res = pd.DataFrame(records)
    print("\n================ FINAL OPERATIONAL CLASSIFICATION METRICS ================")
    print(df_res.to_string(index=False))
    
    tables_dir, _ = get_output_dirs(ROOT_DIR)
    out_file = tables_dir / "classification_metrics.csv"
    df_res.to_csv(out_file, index=False)
    print(f"\nSaved classification metrics to: {out_file}")

    df_cm = pd.DataFrame(cm_records)
    cm_file = tables_dir / "confusion_matrices.csv"
    df_cm.to_csv(cm_file, index=False)
    print(f"Saved confusion matrices to: {cm_file}")

if __name__ == "__main__":
    run_experiment([5, 11])
