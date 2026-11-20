"""Runtime computational efficiency and latency benchmark.

Measures training runtime (seconds) and full-cohort inference latency (milliseconds)
across all evaluated methodologies at Week 11 (LOCO Fold A: Cohort I -> Cohort II).
"""
from __future__ import annotations
import importlib.util
import sys
import time
import warnings
from pathlib import Path
import pandas as pd
from xgboost import XGBRegressor
import torch
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data_loader import build_panel, ALL_ITEMS
from src.features import (
    build_features,
    get_no_affect_features,
    get_static_flat_features,
    get_neural_symbolic_features,
)
from src.output_utils import get_output_dirs

_spec = importlib.util.spec_from_file_location("run_sota_comp", ROOT / "scripts" / "run_sota_comparison.py")
_sota = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_sota)

SEED = _sota.SEED
DEVICE = _sota.DEVICE
set_deterministic_seed = _sota.set_deterministic_seed
_sync_device = _sota._sync_device
_tune_xgb = _sota._tune_xgb
_tune_atew_pipeline = _sota._tune_atew_pipeline
_build_seq_and_tabular_tensors = _sota._build_seq_and_tabular_tensors
_train_seq_model = _sota._train_seq_model
_predict_seq_model = _sota._predict_seq_model
_build_spm_matrices = _sota._build_spm_matrices
_make_bilstm_model = _sota._make_bilstm_model
_make_neural_symbolic_model = _sota._make_neural_symbolic_model


def _measure_xgb_matrices(X_tr, y_tr, X_te):
    t_tr0 = time.perf_counter()
    params = _tune_xgb(X_tr, y_tr)
    model = XGBRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=params["l1"], reg_lambda=params["l2"],
        random_state=SEED, verbosity=0,
    )
    model.fit(X_tr, y_tr)
    train_sec = time.perf_counter() - t_tr0

    t_inf0 = time.perf_counter()
    _ = model.predict(X_te)
    infer_ms = (time.perf_counter() - t_inf0) * 1000.0
    return train_sec, infer_ms


def _measure_xgb_method(feat_df, cols, train_mask, test_mask):
    imputer = SimpleImputer(strategy="median")
    X = feat_df[cols].values
    y = feat_df["final_grade"].values
    X_tr = imputer.fit_transform(X[train_mask])
    X_te = imputer.transform(X[test_mask])
    return _measure_xgb_matrices(X_tr, y[train_mask], X_te)


def _measure_torch_seq_method(surveys_all, feat_df, feature_cols, model_fn, train_mask, test_mask, max_week=11, reg_fn=None):
    _sync_device(DEVICE)
    t_tr0 = time.perf_counter()
    X_seq_tr, X_seq_te, X_tab_tr, X_tab_te, y_tr_t = _build_seq_and_tabular_tensors(
        surveys_all, feat_df, feature_cols, train_mask, test_mask, max_week=max_week
    )
    model = _train_seq_model(
        model_fn(len(ALL_ITEMS), X_tab_tr.shape[1]),
        X_seq_tr, X_tab_tr, y_tr_t, reg_fn=reg_fn, device=DEVICE,
    )
    _sync_device(DEVICE)
    train_sec = time.perf_counter() - t_tr0

    _sync_device(DEVICE)
    t_inf0 = time.perf_counter()
    _ = _predict_seq_model(model, X_seq_te, X_tab_te, device=DEVICE)
    _sync_device(DEVICE)
    infer_ms = (time.perf_counter() - t_inf0) * 1000.0
    return train_sec, infer_ms


def run_runtime_benchmark():
    set_deterministic_seed(SEED)
    print("=" * 70)
    print("COMPUTATIONAL RUNTIME AND LATENCY BENCHMARK")
    print("=" * 70)

    data = build_panel()
    surveys_all = data["surveys_long"]
    att_all = data["attendance_long"]
    grades_all = data["grades"]

    feat_df = build_features(surveys_all, att_all, grades_all, max_week=11)
    feat_df = feat_df[feat_df["final_grade"].notna()].reset_index(drop=True)
    train_mask = (feat_df["year"] == "Year1").values
    test_mask = (feat_df["year"] == "Year2").values
    y_tr_all = feat_df.loc[train_mask, "final_grade"].values

    records = []

    # Grades-LR (OLS linear regression strictly on available continuous grades)
    g_cols = [c for c in _sota.get_available_grade_cols(11) if c in feat_df.columns]
    imp_lr = SimpleImputer(strategy="median")
    X_lr_tr_raw = feat_df.loc[train_mask, g_cols].values
    X_lr_te_raw = feat_df.loc[test_mask, g_cols].values
    t_lr0 = time.perf_counter()
    X_lr_tr = imp_lr.fit_transform(X_lr_tr_raw)
    lr_model = LinearRegression()
    lr_model.fit(X_lr_tr, y_tr_all)
    lr_train_s = time.perf_counter() - t_lr0
    X_lr_te = imp_lr.transform(X_lr_te_raw)
    t_lr_inf0 = time.perf_counter()
    _ = lr_model.predict(X_lr_te)
    lr_infer_ms = (time.perf_counter() - t_lr_inf0) * 1000.0
    records.append({"Method": "Grades-LR", "Training_Time_sec": lr_train_s, "Inference_Cohort_ms": lr_infer_ms})

    for name, raw_cols in [
        ("Academic-Only", get_no_affect_features()),
        ("Static-XGB", get_static_flat_features()),
    ]:
        cols = [c for c in raw_cols if c in feat_df.columns]
        tr_s, inf_ms = _measure_xgb_method(feat_df, cols, train_mask, test_mask)
        records.append({"Method": name, "Training_Time_sec": tr_s, "Inference_Cohort_ms": inf_ms})

    # Seq-Pattern-Mining (includes frequent step-1 transition mining on training fold)
    t_spm0 = time.perf_counter()
    X_spm_tr, X_spm_te = _build_spm_matrices(surveys_all, feat_df, train_mask, test_mask, max_week=11)
    spm_prep_s = time.perf_counter() - t_spm0
    spm_fit_s, spm_inf_ms = _measure_xgb_matrices(X_spm_tr, y_tr_all, X_spm_te)
    records.append({
        "Method": "Seq-Pattern-Mining",
        "Training_Time_sec": spm_prep_s + spm_fit_s,
        "Inference_Cohort_ms": spm_inf_ms,
    })

    # Neural-Symbolic (Affect-LinguoNet hybrid model)
    ns_tr_s, ns_inf_ms = _measure_torch_seq_method(
        surveys_all, feat_df, get_neural_symbolic_features(),
        lambda in_sz, tab_d: _make_neural_symbolic_model(in_sz, tab_d, emo_dim=16),
        train_mask, test_mask, max_week=11,
        reg_fn=lambda m: 1e-3 * torch.sum(m.aff_mod.weight ** 2),
    )
    records.append({
        "Method": "Neural-Symbolic",
        "Training_Time_sec": ns_tr_s,
        "Inference_Cohort_ms": ns_inf_ms,
    })

    # ATEW-XGB (includes trajectory extraction, nested tuning, and RFECV)
    t_atew0 = time.perf_counter()
    train_mask_series = pd.Series(train_mask, index=feat_df.index)
    best_alpha, best_params, selected = _tune_atew_pipeline(
        surveys_all, att_all, grades_all, train_mask_series, max_week=11
    )
    feat_opt = build_features(surveys_all, att_all, grades_all, max_week=11, ewta_alpha=best_alpha)
    feat_opt = feat_opt[feat_opt["final_grade"].notna()].reset_index(drop=True)
    tr_m = (feat_opt["year"] == "Year1").values
    te_m = (feat_opt["year"] == "Year2").values
    cols = [c for c in selected if c in feat_opt.columns]
    imp = SimpleImputer(strategy="median")
    X_tr = imp.fit_transform(feat_opt.loc[tr_m, cols].values)
    X_te = imp.transform(feat_opt.loc[te_m, cols].values)
    y_tr = feat_opt.loc[tr_m, "final_grade"].values
    atew_model = XGBRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=best_params["l1"], reg_lambda=best_params["l2"],
        random_state=SEED, verbosity=0,
    )
    atew_model.fit(X_tr, y_tr)
    atew_train_s = time.perf_counter() - t_atew0

    t_inf0 = time.perf_counter()
    _ = atew_model.predict(X_te)
    atew_infer_ms = (time.perf_counter() - t_inf0) * 1000.0
    records.append({
        "Method": "ATEW-XGB (Ours)",
        "Training_Time_sec": atew_train_s,
        "Inference_Cohort_ms": atew_infer_ms,
    })

    # BiLSTM-Extractor
    bilstm_tr_s, bilstm_inf_ms = _measure_torch_seq_method(
        surveys_all, feat_df, get_no_affect_features(),
        lambda in_sz, acad_d: _make_bilstm_model(in_sz, acad_d, hidden=32),
        train_mask, test_mask, max_week=11,
    )
    records.append({
        "Method": "BiLSTM-Extractor",
        "Training_Time_sec": bilstm_tr_s,
        "Inference_Cohort_ms": bilstm_inf_ms,
    })

    df_runtime = pd.DataFrame(records)
    print("\nBenchmark Summary (4 Decimals):")
    print(df_runtime.to_string(index=False))

    tables_dir, _ = get_output_dirs(ROOT)
    out_csv = tables_dir / "runtime_analysis.csv"
    df_runtime.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"\nSaved runtime benchmark results to: {out_csv}")
    print("=" * 70)


if __name__ == "__main__":
    run_runtime_benchmark()
