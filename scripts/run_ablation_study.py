"""Ablation study evaluating isolated EWTA, psychological operators, and simpler temporal representations."""
from __future__ import annotations
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import torch, random, os

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data_loader import build_panel, ALL_ITEMS
from src.features import build_features, get_atew_features, get_no_affect_features, EWTA_ALPHAS
from src.output_utils import get_output_dirs

def set_deterministic_seed(seed=2026):
    random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

set_deterministic_seed(2026)
warnings.filterwarnings("ignore")

SEED = 2026
PSYCH_FEATS = [
    "frustration_count", "frustration_high_count", "frustration_streak",
    "diff_mot_gap_mean", "diff_mot_gap_trend",
    "diff_mot_gap_ewta", "diff_mot_gap_ewta_slope",
    "engagement_decay", "engagement_ewta_decay",
]


def _train_xgb(X_tr, y_tr, X_te, **params):
    m = XGBRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=params.get("l1", 1.0), reg_lambda=params.get("l2", 5.0),
        random_state=SEED, verbosity=0
    )
    m.fit(X_tr, y_tr)
    return m.predict(X_te)


def compute_bootstrap_metrics(y_true, y_pred, n_bootstrap=1000, seed=SEED):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 3:
        return {k: np.nan for k in ["MSE_mean","MSE_std","MAE_mean","MAE_std","RMSE_mean","RMSE_std","MAPE_mean","MAPE_std","R2_mean","R2_std"]}
    yt, yp = y_true[mask], y_pred[mask]
    n = len(yt)
    rng = np.random.default_rng(seed)
    mses, maes, rmses, mapes, r2s = [], [], [], [], []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yt_b, yp_b = yt[idx], yp[idx]
        mses.append(mean_squared_error(yt_b, yp_b))
        maes.append(mean_absolute_error(yt_b, yp_b))
        rmses.append(np.sqrt(mean_squared_error(yt_b, yp_b)))
        mapes_val = np.mean(np.abs((yt_b - yp_b) / np.clip(np.abs(yt_b), 1e-8, None)))
        mapes.append(mapes_val)
        r2s.append(r2_score(yt_b, yp_b))
    return {
        "MSE_mean": np.mean(mses), "MSE_std": np.std(mses),
        "MAE_mean": np.mean(maes), "MAE_std": np.std(maes),
        "RMSE_mean": np.mean(rmses), "RMSE_std": np.std(rmses),
        "MAPE_mean": np.mean(mapes), "MAPE_std": np.std(mapes),
        "R2_mean": np.mean(r2s), "R2_std": np.std(r2s),
    }


def _tune_ablation(surveys, attendance, grades, train_mask_series,
                   max_week, force_alpha=None, drop_psych=False):
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    imputer = SimpleImputer(strategy="median")
    best_score, best_alpha = -float("inf"), force_alpha if force_alpha else 0.85
    best_params = {"l1": 1.0, "l2": 5.0}
    alphas_to_test = [force_alpha] if force_alpha is not None else EWTA_ALPHAS

    for alpha in alphas_to_test:
        feat_df = build_features(surveys, attendance, grades, max_week=max_week, ewta_alpha=alpha)
        feat_df = feat_df[feat_df["final_grade"].notna()].reset_index(drop=True)
        tr_mask = train_mask_series.reindex(feat_df.index, fill_value=False)
        cols = [c for c in get_atew_features() if c in feat_df.columns]
        if force_alpha == 1.0:
            cols = [c for c in cols if not c.endswith("_recency_delta")]
        if drop_psych:
            cols = [c for c in cols if c not in PSYCH_FEATS]
        X_tr_raw = feat_df.loc[tr_mask, cols].values
        y_tr = feat_df.loc[tr_mask, "final_grade"].values
        for l1 in [0.1, 1.0, 5.0]:
            for l2 in [1.0, 5.0, 10.0]:
                scores = []
                for tr_i, va_i in kf.split(X_tr_raw):
                    imp_cv = SimpleImputer(strategy="median")
                    X_tr_f = imp_cv.fit_transform(X_tr_raw[tr_i])
                    X_va_f = imp_cv.transform(X_tr_raw[va_i])
                    m = XGBRegressor(
                        n_estimators=100, max_depth=3, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=l1, reg_lambda=l2, random_state=SEED, verbosity=0
                    )
                    m.fit(X_tr_f, y_tr[tr_i])
                    scores.append(r2_score(y_tr[va_i], m.predict(X_va_f)))
                s = np.mean(scores)
                if s > best_score:
                    best_score = s; best_alpha = alpha; best_params = {"l1": l1, "l2": l2}

    feat_df = build_features(surveys, attendance, grades, max_week=max_week, ewta_alpha=best_alpha)
    feat_df = feat_df[feat_df["final_grade"].notna()].reset_index(drop=True)
    tr_mask = train_mask_series.reindex(feat_df.index, fill_value=False)
    cols = [c for c in get_atew_features() if c in feat_df.columns]
    if force_alpha == 1.0:
        cols = [c for c in cols if not c.endswith("_recency_delta")]
    if drop_psych:
        cols = [c for c in cols if c not in PSYCH_FEATS]
    X_tr_df = feat_df.loc[tr_mask, cols].copy()
    valid_cols = [c for c in cols if X_tr_df[c].notna().any()]
    X_tr_arr = imputer.fit_transform(X_tr_df[valid_cols])
    y_tr = feat_df.loc[tr_mask, "final_grade"].values
    estimator = XGBRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=best_params["l1"], reg_lambda=best_params["l2"],
        random_state=SEED, verbosity=0
    )
    selector = RFECV(estimator, step=1, cv=kf, scoring="r2", min_features_to_select=8, n_jobs=-1)
    selector.fit(X_tr_arr, y_tr)
    selected = [valid_cols[i] for i in range(len(valid_cols)) if selector.support_[i]]
    return best_alpha, best_params, selected


def run_loco_ablation_fold(surveys_all, att_all, grades_all,
                           train_year, test_year, max_week, methods):
    """Single LOCO fold for ablation."""
    imputer = SimpleImputer(strategy="median")
    feat_df = build_features(surveys_all, att_all, grades_all, max_week=max_week)
    feat_df = feat_df[feat_df["final_grade"].notna()].reset_index(drop=True)
    tr_mask_bool = (feat_df["year"] == train_year).values
    te_mask_bool = (feat_df["year"] == test_year).values
    y_te = feat_df.loc[te_mask_bool, "final_grade"].values
    train_mask_series = pd.Series(tr_mask_bool, index=feat_df.index)

    fold_preds = {}
    for m in methods:
        t0 = time.perf_counter()
        best_alpha, best_params, selected_feats = _tune_ablation(
            surveys_all, att_all, grades_all, train_mask_series,
            max_week, force_alpha=m["force_alpha"], drop_psych=m["drop_psych"]
        )
        feat_df_opt = build_features(surveys_all, att_all, grades_all,
                                     max_week=max_week, ewta_alpha=best_alpha)
        feat_df_opt = feat_df_opt[feat_df_opt["final_grade"].notna()].reset_index(drop=True)
        tr_m = (feat_df_opt["year"] == train_year).values
        te_m = (feat_df_opt["year"] == test_year).values
        cols = [c for c in selected_feats if c in feat_df_opt.columns]
        X_opt = feat_df_opt[cols].values
        X_tr = imputer.fit_transform(X_opt[tr_m])
        X_te = imputer.transform(X_opt[te_m])
        y_tr_opt = feat_df_opt.loc[tr_m, "final_grade"].values
        preds = _train_xgb(X_tr, y_tr_opt, X_te, **best_params)
        elapsed = time.perf_counter() - t0
        fold_preds[m["name"]] = {"preds": preds, "elapsed": elapsed}
        print(f"    [{train_year}->{test_year}] {m['name']:16s}  R2={r2_score(y_te, preds):.4f}  ({elapsed:.1f}s)")

    return fold_preds, y_te


def _tune_xgb(X_tr, y_tr):
    """5-fold CV grid search over L1/L2 penalties for XGBoost baselines."""
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    best_score, best_params = -float("inf"), {"l1": 1.0, "l2": 5.0}
    for l1 in [0.1, 1.0, 5.0]:
        for l2 in [1.0, 5.0, 10.0]:
            scores = []
            for tr_i, va_i in kf.split(X_tr):
                imp = SimpleImputer(strategy="median")
                X_tr_f = imp.fit_transform(X_tr[tr_i])
                X_va_f = imp.transform(X_tr[va_i])
                m = XGBRegressor(
                    n_estimators=100, max_depth=3, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=l1, reg_lambda=l2, random_state=SEED, verbosity=0,
                )
                m.fit(X_tr_f, y_tr[tr_i])
                scores.append(r2_score(y_tr[va_i], m.predict(X_va_f)))
            s = float(np.mean(scores))
            if s > best_score:
                best_score, best_params = s, {"l1": l1, "l2": l2}
    return best_params


def _add_last_observations(feat_df: pd.DataFrame, surveys: pd.DataFrame, max_week: int) -> pd.DataFrame:
    """Attach most recent observed affective rating up to max_week for each student."""
    sv = surveys[surveys["week"] <= max_week].sort_values("week")
    rows = []
    for (yr, sid), grp in sv.groupby(["year", "ID"]):
        rec = {"year": yr, "ID": sid}
        for item in ALL_ITEMS:
            vals = grp[item].dropna().values
            rec[f"{item}_last"] = float(vals[-1]) if len(vals) else np.nan
        rows.append(rec)
    return feat_df.merge(pd.DataFrame(rows), on=["year", "ID"], how="left")


def _tune_standard_ewma(surveys, attendance, grades, train_mask_series, max_week: int):
    """Tune EWMA decay alpha and L1/L2 penalties for Standard-EWMA (ewta_mean + academic)."""
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    cols_ewma = get_no_affect_features() + [f"{item}_ewta_mean" for item in ALL_ITEMS]
    best_score, best_alpha = -float("inf"), 0.85
    best_params = {"l1": 1.0, "l2": 5.0}

    for alpha in EWTA_ALPHAS:
        feat_df = build_features(surveys, attendance, grades, max_week=max_week, ewta_alpha=alpha)
        feat_df = feat_df[feat_df["final_grade"].notna()].reset_index(drop=True)
        tr_mask = train_mask_series.reindex(feat_df.index, fill_value=False)
        cols = [c for c in cols_ewma if c in feat_df.columns]
        X_tr_raw = feat_df.loc[tr_mask, cols].values
        y_tr = feat_df.loc[tr_mask, "final_grade"].values
        for l1 in [0.1, 1.0, 5.0]:
            for l2 in [1.0, 5.0, 10.0]:
                scores = []
                for tr_i, va_i in kf.split(X_tr_raw):
                    imp = SimpleImputer(strategy="median")
                    X_tr_f = imp.fit_transform(X_tr_raw[tr_i])
                    X_va_f = imp.transform(X_tr_raw[va_i])
                    m = XGBRegressor(
                        n_estimators=100, max_depth=3, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.8,
                        reg_alpha=l1, reg_lambda=l2, random_state=SEED, verbosity=0,
                    )
                    m.fit(X_tr_f, y_tr[tr_i])
                    scores.append(r2_score(y_tr[va_i], m.predict(X_va_f)))
                s = float(np.mean(scores))
                if s > best_score:
                    best_score, best_alpha, best_params = s, alpha, {"l1": l1, "l2": l2}
    return best_alpha, best_params, cols_ewma


def run_temporal_baselines_week(surveys_all, att_all, grades_all, max_week: int) -> list[dict]:
    """Evaluate simpler temporal representations (Recent-Obs, Linear-Trends, Standard-EWMA) at max_week."""
    loco_folds = [("Year1", "Year2"), ("Year2", "Year1")]
    variants = ["Recent-Obs", "Linear-Trends", "Standard-EWMA"]
    fold_preds = {v: [] for v in variants}
    fold_y = []
    fold_times = {v: 0.0 for v in variants}

    recent_cols = get_no_affect_features() + [f"{item}_last" for item in ALL_ITEMS]
    linear_cols = get_no_affect_features() + [f"{item}_mean" for item in ALL_ITEMS] + [f"{item}_trend" for item in ALL_ITEMS]

    feat_base = build_features(surveys_all, att_all, grades_all, max_week=max_week)
    feat_base = _add_last_observations(feat_base, surveys_all, max_week=max_week)
    feat_base = feat_base[feat_base["final_grade"].notna()].reset_index(drop=True)

    for tr_yr, te_yr in loco_folds:
        train_mask = (feat_base["year"] == tr_yr).values
        test_mask = (feat_base["year"] == te_yr).values
        y_tr = feat_base.loc[train_mask, "final_grade"].values
        y_te = feat_base.loc[test_mask, "final_grade"].values
        fold_y.append(y_te)
        imputer = SimpleImputer(strategy="median")

        for name, raw_cols in [("Recent-Obs", recent_cols), ("Linear-Trends", linear_cols)]:
            t0 = time.perf_counter()
            cols = [c for c in raw_cols if c in feat_base.columns]
            X = feat_base[cols].values
            X_tr = imputer.fit_transform(X[train_mask])
            X_te = imputer.transform(X[test_mask])
            params = _tune_xgb(X_tr, y_tr)
            fold_preds[name].append(_train_xgb(X_tr, y_tr, X_te, **params))
            fold_times[name] += time.perf_counter() - t0

        t0 = time.perf_counter()
        best_alpha, best_params, cols_ewma = _tune_standard_ewma(
            surveys_all, att_all, grades_all, pd.Series(train_mask, index=feat_base.index), max_week=max_week
        )
        feat_opt = build_features(surveys_all, att_all, grades_all, max_week=max_week, ewta_alpha=best_alpha)
        feat_opt = feat_opt[feat_opt["final_grade"].notna()].reset_index(drop=True)
        tr_m = (feat_opt["year"] == tr_yr).values
        te_m = (feat_opt["year"] == te_yr).values
        cols = [c for c in cols_ewma if c in feat_opt.columns]
        X_opt = feat_opt[cols].values
        X_tr = imputer.fit_transform(X_opt[tr_m])
        X_te = imputer.transform(X_opt[te_m])
        fold_preds["Standard-EWMA"].append(_train_xgb(X_tr, feat_opt.loc[tr_m, "final_grade"].values, X_te, **best_params))
        fold_times["Standard-EWMA"] += time.perf_counter() - t0

    all_y = np.concatenate(fold_y)
    rows = []
    for v in variants:
        met = compute_bootstrap_metrics(all_y, np.concatenate(fold_preds[v]))
        met["Method"] = v
        met["max_week"] = max_week
        met["time"] = fold_times[v]
        rows.append(met)
        print(f"  [W{max_week:02d}] {v:15s}  R2={met['R2_mean']:.3f}+/-{met['R2_std']:.3f}")
    return rows


def main():
    print("Running ATEW ablation study (LOCO protocol)...")

    data = build_panel()
    surveys_all = data["surveys_long"]
    att_all     = data["attendance_long"]
    grades_all  = data["grades"]

    methods = [
        {"name": "ATEW-Full",    "force_alpha": None, "drop_psych": False},
        {"name": "ATEW-No-EWTA", "force_alpha": 1.0,  "drop_psych": False},
        {"name": "ATEW-No-Psych","force_alpha": None,  "drop_psych": True},
    ]

    full_sweep = os.environ.get("RUN_FULL_SWEEP", "0") == "1"
    weeks = list(range(11, 3, -1)) if full_sweep else [11]
    all_results = []
    temporal_results = []

    for w in weeks:
        print(f"\n--- Week {w} ---")
        loco_folds = [("Year1", "Year2"), ("Year2", "Year1")]
        fold_data = []
        for tr_yr, te_yr in loco_folds:
            fp, y_te = run_loco_ablation_fold(
                surveys_all, att_all, grades_all, tr_yr, te_yr, w, methods
            )
            fold_data.append((fp, y_te))

        for m in methods:
            all_preds = np.concatenate([fold_data[f][0][m["name"]]["preds"] for f in range(2)])
            all_y     = np.concatenate([fold_data[f][1] for f in range(2)])
            met = compute_bootstrap_metrics(all_y, all_preds)

            met["Method"] = m["name"]
            met["max_week"] = w
            met["time"] = sum(fold_data[f][0][m["name"]]["elapsed"] for f in range(2))
            all_results.append(met)
            print(f"  [LOCO-avg] {m['name']:16s}  R2={met['R2_mean']:.4f}+/-{met['R2_std']:.4f}")

    df_res = pd.DataFrame(all_results)

    print("\nRunning simpler temporal representation baselines...")
    for w in (range(4, 12) if full_sweep else [11]):
        temporal_results.extend(run_temporal_baselines_week(surveys_all, att_all, grades_all, max_week=w))
    df_temp = pd.DataFrame(temporal_results)[[c for c in df_res.columns]]

    order_map = {
        "Recent-Obs": 0, "Linear-Trends": 1, "Standard-EWMA": 2,
        "ATEW-No-EWTA": 3, "ATEW-No-Psych": 4, "ATEW-Full": 5,
    }
    df_unified = pd.concat([df_temp, df_res], ignore_index=True)
    df_unified["_ord"] = df_unified["Method"].map(order_map)
    df_unified = df_unified.sort_values(["max_week", "_ord"]).drop(columns=["_ord"]).reset_index(drop=True)
    out_dir, _ = get_output_dirs(ROOT)
    out_name = "ablation_summary.csv"
    df_unified.to_csv(out_dir / out_name, index=False, float_format="%.6f")
    print(f"Saved ablation summary to {out_dir / out_name}")
    if not full_sweep:
        print("Weeks 4-10 sweep skipped. (Set RUN_FULL_SWEEP=1 to rerun full Weeks 4-11 ablation).")


if __name__ == "__main__":
    main()
