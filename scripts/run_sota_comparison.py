"""SOTA baseline comparison under Leave-One-Cohort-Out (LOCO) evaluation protocol."""
from __future__ import annotations
import sys, warnings, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV
from xgboost import XGBRegressor
import torch, random, os

warnings.filterwarnings("ignore")
SEED = 2026
np.random.seed(SEED)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data_loader import build_panel, ALL_ITEMS
from src.features import (
    build_features, EWTA_ALPHAS,
    get_atew_features, get_no_affect_features,
    get_seq_pattern_mining_features, get_neural_symbolic_features,
    get_static_flat_features, mine_frequent_step1_patterns,
)
from src.output_utils import get_output_dirs

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

set_deterministic_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _sync_device(device: torch.device = DEVICE):
    """Synchronize CUDA kernels before timing when running on GPU."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)

METHOD_NAMES = [
    "Grades-LR", "Academic-Only", "Static-XGB", "BiLSTM-Extractor",
    "Seq-Pattern-Mining", "Neural-Symbolic", "ATEW-XGB",
]

BASE_SUMMARY_COLS = [
    "MSE_mean", "MSE_std", "MAE_mean", "MAE_std",
    "RMSE_mean", "RMSE_std", "MAPE_mean", "MAPE_std",
    "R2_mean", "R2_std", "Method", "max_week", "time",
]
FULL_SUMMARY_COLS = BASE_SUMMARY_COLS + ["CCC_mean", "CCC_std"]

GRADE_SCHEDULE = [
    (4, "P1"),
    (6, "P2"),
    (8, "test1"),
    (10, "P3"),
    (10, "P4"),
]


def get_available_grade_cols(max_week: int) -> list[str]:
    """Return continuous assessment grade columns released by max_week."""
    return [col for wk, col in GRADE_SCHEDULE if max_week >= wk]


def concordance_correlation_coefficient(y_true, y_pred):
    """
    Lin's Concordance Correlation Coefficient (CCC).
    Lin, L. I. (1989). A concordance correlation coefficient to evaluate reproducibility.
    Biometrics, 255-268.
    """
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    cov = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    denom = var_true + var_pred + (mean_true - mean_pred) ** 2
    if denom == 0:
        return 1.0 if np.allclose(y_true, y_pred) else 0.0
    return (2.0 * cov) / denom


def _train_xgb(X_tr, y_tr, X_te, l1=0.1, l2=1.0):
    model = XGBRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=l1, reg_lambda=l2,
        random_state=SEED, verbosity=0,
    )
    model.fit(X_tr, y_tr)
    return model.predict(X_te)


def _tune_xgb(X_tr, y_tr):
    l1s = [0.1, 1.0, 5.0]
    l2s = [1.0, 5.0, 10.0]
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    best_score, best_params = -float("inf"), {"l1": 1.0, "l2": 5.0}
    for l1 in l1s:
        for l2 in l2s:
            scores = []
            for tr_i, va_i in kf.split(X_tr):
                imp = SimpleImputer(strategy="median")
                X_tr_f = imp.fit_transform(X_tr[tr_i])
                X_va_f = imp.transform(X_tr[va_i])
                m = XGBRegressor(
                    n_estimators=100, max_depth=3, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=l1, reg_lambda=l2,
                    random_state=SEED, verbosity=0
                )
                m.fit(X_tr_f, y_tr[tr_i])
                scores.append(r2_score(y_tr[va_i], m.predict(X_va_f)))
            s = np.mean(scores)
            if s > best_score:
                best_score = s
                best_params = {"l1": l1, "l2": l2}
    return best_params


def _tune_atew_pipeline(surveys, attendance, grades, train_mask_series, max_week):
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    l1s = [0.1, 1.0, 5.0]
    l2s = [1.0, 5.0, 10.0]
    imputer = SimpleImputer(strategy="median")
    best_score, best_alpha = -float("inf"), 0.85
    best_params = {"l1": 1.0, "l2": 5.0}

    for alpha in EWTA_ALPHAS:
        feat_df = build_features(surveys, attendance, grades, max_week=max_week, ewta_alpha=alpha)
        feat_df = feat_df[feat_df["final_grade"].notna()].reset_index(drop=True)
        tr_mask = train_mask_series.reindex(feat_df.index, fill_value=False)
        cols = [c for c in get_atew_features() if c in feat_df.columns]
        X_tr_raw = feat_df.loc[tr_mask, cols].values
        y_tr = feat_df.loc[tr_mask, "final_grade"].values
        for l1 in l1s:
            for l2 in l2s:
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

    # RFECV with best alpha
    feat_df = build_features(surveys, attendance, grades, max_week=max_week, ewta_alpha=best_alpha)
    feat_df = feat_df[feat_df["final_grade"].notna()].reset_index(drop=True)
    tr_mask = train_mask_series.reindex(feat_df.index, fill_value=False)
    cols = [c for c in get_atew_features() if c in feat_df.columns]
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


def _build_seq_and_tabular_tensors(surveys_long, y_df, feature_cols, train_mask, test_mask, max_week=11):
    """Build weekly survey sequences and training-imputed tabular tensors for neural models."""
    n_items = len(ALL_ITEMS)
    all_ids = y_df[["year", "ID"]].values
    sequences = np.zeros((len(all_ids), max_week, n_items), dtype=np.float32)
    for i, (yr, sid) in enumerate(all_ids):
        s = surveys_long[(surveys_long["year"] == yr) & (surveys_long["ID"] == sid)]
        for _, row in s.iterrows():
            w = int(row["week"]) - 1
            if 0 <= w < max_week:
                for j, item in enumerate(ALL_ITEMS):
                    sequences[i, w, j] = float(row[item]) if pd.notna(row[item]) else 0.0

    valid_cols = [c for c in feature_cols if c in y_df.columns and y_df.loc[train_mask, c].notna().any()]
    tab_imputer = SimpleImputer(strategy="median")
    X_tab_tr_arr = tab_imputer.fit_transform(y_df.loc[train_mask, valid_cols])
    X_tab_te_arr = tab_imputer.transform(y_df.loc[test_mask, valid_cols])

    tr_idx = np.where(train_mask)[0]
    te_idx = np.where(test_mask)[0]
    return (
        torch.FloatTensor(sequences[tr_idx]),
        torch.FloatTensor(sequences[te_idx]),
        torch.FloatTensor(X_tab_tr_arr),
        torch.FloatTensor(X_tab_te_arr),
        torch.FloatTensor(y_df.iloc[tr_idx]["final_grade"].values),
    )


def _train_seq_model(model, X_seq_tr, X_tab_tr, y_tr_t, reg_fn=None, device: torch.device = DEVICE):
    """Train a bimodal sequence-tabular PyTorch regressor with early stopping on `device`."""
    import copy
    import torch.nn as nn
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, TensorDataset

    model = model.to(device)
    indices = np.arange(len(X_seq_tr))
    tr_inner, va_inner = train_test_split(indices, test_size=0.15, random_state=SEED)
    Xs_t, Xs_v = X_seq_tr[tr_inner], X_seq_tr[va_inner].to(device)
    Xt_t, Xt_v = X_tab_tr[tr_inner], X_tab_tr[va_inner].to(device)
    y_t, y_v = y_tr_t[tr_inner], y_tr_t[va_inner].to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    dl_t = DataLoader(TensorDataset(Xs_t, Xt_t, y_t), batch_size=16, shuffle=True)
    best_val, patience_counter, best_weights = float("inf"), 0, None

    for _ in range(100):
        model.train()
        for xs_b, xt_b, yb in dl_t:
            xs_b, xt_b, yb = xs_b.to(device), xt_b.to(device), yb.to(device)
            pred = model(xs_b, xt_b)
            loss = loss_fn(pred, yb)
            if reg_fn is not None:
                loss = loss + reg_fn(model)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xs_v, Xt_v), y_v).item()
        if val_loss < best_val:
            best_val = val_loss
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 15:
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model


def _predict_seq_model(model, X_seq_te, X_tab_te, device: torch.device = DEVICE):
    """Run inference for a trained PyTorch sequence-tabular regressor on `device`."""
    model.eval()
    with torch.no_grad():
        return model(X_seq_te.to(device), X_tab_te.to(device)).cpu().numpy()


def _fit_seq_regressor(model, X_seq_tr, X_tab_tr, y_tr_t, X_seq_te, X_tab_te, reg_fn=None, device: torch.device = DEVICE):
    """Train a bimodal sequence-tabular PyTorch regressor with early stopping and return predictions."""
    model = _train_seq_model(model, X_seq_tr, X_tab_tr, y_tr_t, reg_fn=reg_fn, device=device)
    return _predict_seq_model(model, X_seq_te, X_tab_te, device=device)


def _make_bilstm_model(input_size, acad_dim, hidden=32):
    """Create BiLSTM regressor with attention-gated bimodal fusion (Hao, 2026)."""
    import torch.nn as nn

    class LSTMReg(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden, batch_first=True, num_layers=1, bidirectional=True)
            fusion_dim = hidden * 2 + acad_dim
            self.attn_gate = nn.Sequential(nn.Linear(fusion_dim, fusion_dim), nn.Sigmoid())
            self.fc = nn.Linear(fusion_dim, 1)

        def forward(self, x_seq, x_acad):
            _, (h, _) = self.lstm(x_seq)
            h_cat = torch.cat((h[-2], h[-1], x_acad), dim=-1)
            h_fused = h_cat * self.attn_gate(h_cat)
            return self.fc(h_fused).squeeze(-1)

    return LSTMReg()


def _make_neural_symbolic_model(input_size, tab_dim, emo_dim=16):
    """Create hybrid neural-symbolic regressor adapting Affect-LinguoNet (Lu, 2025)."""
    import torch.nn as nn

    class AffectLinguoNetReg(nn.Module):
        def __init__(self):
            super().__init__()
            self.emo_gru = nn.GRU(input_size, emo_dim, batch_first=True)
            self.emo_attn = nn.Linear(emo_dim, 1)
            self.tab_proj = nn.Linear(tab_dim, emo_dim)
            self.joint_norm = nn.LayerNorm(emo_dim * 2)
            self.comp_proj = nn.Linear(emo_dim * 2, emo_dim)
            self.aff_mod = nn.Linear(emo_dim, emo_dim, bias=False)
            self.head = nn.Linear(emo_dim + tab_dim, 1)

        def forward(self, x_seq, x_tab):
            emo_seq, _ = self.emo_gru(x_seq)
            attn_w = torch.softmax(self.emo_attn(emo_seq), dim=1)
            eps = torch.sum(attn_w * emo_seq, dim=1)
            z_tab = self.tab_proj(x_tab)
            h_joint = self.joint_norm(torch.cat((z_tab, eps), dim=-1))
            c_mod = torch.sigmoid(self.comp_proj(h_joint)) + self.aff_mod(eps)
            return self.head(torch.cat((c_mod, x_tab), dim=-1)).squeeze(-1)

    return AffectLinguoNetReg()


def _train_lstm(surveys_long, y_df, train_mask, test_mask, max_week=11):
    """BiLSTM baseline with attention-gated bimodal fusion (Hao, 2026)."""
    X_seq_tr, X_seq_te, X_acad_tr, X_acad_te, y_tr_t = _build_seq_and_tabular_tensors(
        surveys_long, y_df, get_no_affect_features(), train_mask, test_mask, max_week
    )
    model = _make_bilstm_model(len(ALL_ITEMS), X_acad_tr.shape[1], hidden=32)
    return _fit_seq_regressor(model, X_seq_tr, X_acad_tr, y_tr_t, X_seq_te, X_acad_te)


def _train_neural_symbolic(surveys_long, y_df, train_mask, test_mask, max_week=11):
    """Hybrid neural-symbolic predictor adapting Affect-LinguoNet (Lu, 2025)."""
    X_seq_tr, X_seq_te, X_tab_tr, X_tab_te, y_tr_t = _build_seq_and_tabular_tensors(
        surveys_long, y_df, get_neural_symbolic_features(), train_mask, test_mask, max_week
    )
    model = _make_neural_symbolic_model(len(ALL_ITEMS), X_tab_tr.shape[1], emo_dim=16)
    return _fit_seq_regressor(
        model, X_seq_tr, X_tab_tr, y_tr_t, X_seq_te, X_tab_te,
        reg_fn=lambda m: 1e-3 * torch.sum(m.aff_mod.weight ** 2),
    )


def _build_spm_matrices(surveys_long, feat_df, train_mask, test_mask, max_week=11, min_support=0.20):
    """Combine mined step-1 transition frequencies (min_support on training fold) with SPM features."""
    train_ids = set(zip(feat_df.loc[train_mask, "year"].astype(str), feat_df.loc[train_mask, "ID"].astype(int)))
    pat_df = mine_frequent_step1_patterns(surveys_long, train_ids, max_week=max_week, min_support=min_support)
    merged = feat_df.merge(pat_df, on=["year", "ID"], how="left")
    pat_cols = [c for c in pat_df.columns if c.startswith("spm_pat_")]
    if pat_cols:
        merged[pat_cols] = merged[pat_cols].fillna(0)
    cols = [c for c in get_seq_pattern_mining_features() if c in merged.columns] + pat_cols
    X = merged[cols].values
    imputer = SimpleImputer(strategy="median")
    return imputer.fit_transform(X[train_mask]), imputer.transform(X[test_mask])


def compute_bootstrap_metrics(y_true, y_pred, n_bootstrap=1000, seed=SEED):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if mask.sum() < 3:
        return {k: np.nan for k in ["MSE_mean","MSE_std","MAE_mean","MAE_std","RMSE_mean","RMSE_std","MAPE_mean","MAPE_std","R2_mean","R2_std","CCC_mean","CCC_std"]}
    yt, yp = y_true[mask], y_pred[mask]
    n = len(yt)
    rng = np.random.default_rng(seed)
    mses, maes, rmses, mapes, r2s, cccs = [], [], [], [], [], []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yt_b, yp_b = yt[idx], yp[idx]
        mses.append(mean_squared_error(yt_b, yp_b))
        maes.append(mean_absolute_error(yt_b, yp_b))
        rmses.append(np.sqrt(mean_squared_error(yt_b, yp_b)))
        mapes_val = np.mean(np.abs((yt_b - yp_b) / np.clip(np.abs(yt_b), 1e-8, None)))
        mapes.append(mapes_val)
        r2s.append(r2_score(yt_b, yp_b))
        cccs.append(concordance_correlation_coefficient(yt_b, yp_b))
    return {
        "MSE_mean": np.mean(mses), "MSE_std": np.std(mses),
        "MAE_mean": np.mean(maes), "MAE_std": np.std(maes),
        "RMSE_mean": np.mean(rmses), "RMSE_std": np.std(rmses),
        "MAPE_mean": np.mean(mapes), "MAPE_std": np.std(mapes),
        "R2_mean": np.mean(r2s), "R2_std": np.std(r2s),
        "CCC_mean": np.mean(cccs), "CCC_std": np.std(cccs),
    }


def _paired_permutation_test(y_true, preds_a, preds_b, n_perm=5000, n_boot=1000, seed=SEED):
    """Paired Monte Carlo permutation test (Good 2005; Raschka 2018) with paired bootstrap 95% CI for Delta R2."""
    rng = np.random.default_rng(seed)
    err_a = (y_true - preds_a) ** 2
    err_b = (y_true - preds_b) ** 2
    base_diff = np.mean(err_b) - np.mean(err_a)

    count = 0
    for _ in range(n_perm):
        swap = rng.binomial(1, 0.5, size=len(y_true))
        p_err_a = np.where(swap, err_b, err_a)
        p_err_b = np.where(swap, err_a, err_b)
        diff = np.mean(p_err_b) - np.mean(p_err_a)
        if diff >= base_diff:
            count += 1

    p_val = count / n_perm
    r2_a = r2_score(y_true, preds_a)
    r2_b = r2_score(y_true, preds_b)
    delta_r2 = r2_a - r2_b

    # Paired student-level bootstrap 95% confidence interval for Delta R2
    boot_diffs = []
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot_diffs.append(r2_score(y_true[idx], preds_a[idx]) - r2_score(y_true[idx], preds_b[idx]))
    ci_low, ci_high = np.percentile(boot_diffs, [2.5, 97.5])
    return delta_r2, p_val, count, float(ci_low), float(ci_high)


def run_loco_fold(data, surveys_all, attendance_all, grades_all,
                  train_year: str, test_year: str, max_week: int = 11):
    """
    Single LOCO fold: train on train_year, test on test_year.
    Returns a dict {method: predictions_array} and the ground truth y_te.
    """
    # Build features on BOTH years so indices align with the global DataFrame
    surveys_both = surveys_all.reset_index(drop=True)
    att_both = attendance_all.reset_index(drop=True)

    feat_df = build_features(surveys_both, att_both, grades_all, max_week=max_week)
    feat_df = feat_df[feat_df["final_grade"].notna()].reset_index(drop=True)

    train_mask = (feat_df["year"] == train_year).values
    test_mask  = (feat_df["year"] == test_year).values
    y = feat_df["final_grade"].values
    y_tr, y_te = y[train_mask], y[test_mask]

    imputer = SimpleImputer(strategy="median")
    results = {}

    for method in METHOD_NAMES:
        t0 = time.perf_counter()

        if method == "Grades-LR":
            g_cols = [c for c in get_available_grade_cols(max_week) if c in feat_df.columns and feat_df[c].notna().any()]
            if not g_cols:
                preds = np.full(test_mask.sum(), float(np.mean(y_tr)))
            else:
                X = feat_df[g_cols].values
                X_tr = imputer.fit_transform(X[train_mask])
                X_te = imputer.transform(X[test_mask])
                lr = LinearRegression()
                lr.fit(X_tr, y_tr)
                preds = lr.predict(X_te)

        elif method == "Academic-Only":
            cols = [c for c in get_no_affect_features() if c in feat_df.columns]
            X = feat_df[cols].values
            X_tr = imputer.fit_transform(X[train_mask])
            X_te = imputer.transform(X[test_mask])
            params = _tune_xgb(X_tr, y_tr)
            preds = _train_xgb(X_tr, y_tr, X_te, **params)

        elif method == "Static-XGB":
            cols = [c for c in get_static_flat_features() if c in feat_df.columns]
            X = feat_df[cols].values
            X_tr = imputer.fit_transform(X[train_mask])
            X_te = imputer.transform(X[test_mask])
            params = _tune_xgb(X_tr, y_tr)
            preds = _train_xgb(X_tr, y_tr, X_te, **params)

        elif method == "BiLSTM-Extractor":
            preds = _train_lstm(surveys_both, feat_df, train_mask, test_mask, max_week)

        elif method == "Seq-Pattern-Mining":
            X_tr, X_te = _build_spm_matrices(surveys_both, feat_df, train_mask, test_mask, max_week=max_week)
            params = _tune_xgb(X_tr, y_tr)
            preds = _train_xgb(X_tr, y_tr, X_te, **params)

        elif method == "Neural-Symbolic":
            preds = _train_neural_symbolic(surveys_both, feat_df, train_mask, test_mask, max_week)

        elif method == "ATEW-XGB":
            train_mask_series = pd.Series(train_mask, index=feat_df.index)
            best_alpha, best_xgb_params, selected_feats = _tune_atew_pipeline(
                surveys_both, att_both, grades_all, train_mask_series, max_week
            )
            feat_df_opt = build_features(surveys_both, att_both, grades_all,
                                         max_week=max_week, ewta_alpha=best_alpha)
            feat_df_opt = feat_df_opt[feat_df_opt["final_grade"].notna()].reset_index(drop=True)
            tr_m = (feat_df_opt["year"] == train_year).values
            te_m = (feat_df_opt["year"] == test_year).values
            cols = [c for c in selected_feats if c in feat_df_opt.columns]
            X_opt = feat_df_opt[cols].values
            X_tr = imputer.fit_transform(X_opt[tr_m])
            X_te = imputer.transform(X_opt[te_m])
            y_tr_opt = feat_df_opt.loc[tr_m, "final_grade"].values
            preds = _train_xgb(X_tr, y_tr_opt, X_te, **best_xgb_params)

        elapsed = time.perf_counter() - t0
        results[method] = {"preds": preds, "elapsed": elapsed}
        r2_val = r2_score(y_te, preds)
        print(f"    [{train_year}->{test_year}] {method:22s}  R2={r2_val:.4f}  ({elapsed:.1f}s)")

    return results, y_te, feat_df


def run_experiment(max_week: int = 11, include_ccc: bool = True):
    """Run LOCO: both folds, average bootstrap metrics."""
    print(f"\nRunning LOCO evaluation (max_week={max_week})...")

    data = build_panel()
    surveys_all = data["surveys_long"]
    att_all     = data["attendance_long"]
    grades_all  = data["grades"]

    loco_folds = [("Year1", "Year2"), ("Year2", "Year1")]
    fold_results = []  # list of (results_dict, y_te)

    for tr_yr, te_yr in loco_folds:
        res, y_te, _ = run_loco_fold(data, surveys_all, att_all, grades_all,
                                     tr_yr, te_yr, max_week)
        fold_results.append((res, y_te))

    all_rows = []
    for method in METHOD_NAMES:
        all_preds = np.concatenate([fold_results[f][0][method]["preds"] for f in range(2)])
        all_y     = np.concatenate([fold_results[f][1] for f in range(2)])
        met = compute_bootstrap_metrics(all_y, all_preds)

        met["Method"] = method
        met["max_week"] = max_week
        met["time"] = sum(fold_results[f][0][method]["elapsed"] for f in range(2))
        all_rows.append(met)
        print(f"  [LOCO-avg] {method:22s}  R2={met['R2_mean']:.4f}+/-{met['R2_std']:.4f}  MAE={met['MAE_mean']:.4f}+/-{met['MAE_std']:.4f}")

    df = pd.DataFrame(all_rows)
    cols = FULL_SUMMARY_COLS if include_ccc else BASE_SUMMARY_COLS
    return df[cols], fold_results


def main():
    print("Running LOCO comparison across cohorts...")
    out_dir, _ = get_output_dirs(ROOT)

    # Full experiment (week 11)
    results_full, fold_results_11 = run_experiment(max_week=11, include_ccc=True)
    results_full.to_csv(out_dir / "sota_summary.csv", index=False, float_format="%.6f")
    print(f"Saved sota_summary.csv to {out_dir}")

    # Early warning sweep (weeks 4 to 11)
    if os.environ.get("RUN_FULL_SWEEP", "0") == "1":
        print("\nRunning early warning sweep (weeks 4 to 11)...")
        ew_results = []
        for w in range(4, 12):
            res_w, _ = run_experiment(max_week=w, include_ccc=False)
            ew_results.append(res_w)
        ew_df = pd.concat(ew_results, ignore_index=True)
        ew_df.to_csv(out_dir / "sota_early_warning_summary.csv", index=False, float_format="%.6f")
        print(f"Saved sota_early_warning_summary.csv to {out_dir}")
    else:
        print("\nEarly-warning sweep preserved from precomputed results. (Set RUN_FULL_SWEEP=1 to rerun all weeks).")

    # Paired Monte Carlo permutation test across all held-out LOCO predictions
    print("\nEvaluating statistical significance (LOCO held-out predictions)...")
    all_y = np.concatenate([fold_results_11[f][1] for f in range(2)])
    preds_atew = np.concatenate([fold_results_11[f][0]["ATEW-XGB"]["preds"] for f in range(2)])

    baselines = ["Neural-Symbolic", "Seq-Pattern-Mining", "BiLSTM-Extractor", "Static-XGB", "Academic-Only", "Grades-LR"]
    sig_results = []
    n_perm = 5000

    for base in baselines:
        preds_base = np.concatenate([fold_results_11[f][0][base]["preds"] for f in range(2)])
        diff, p_val, exceed_count, ci_low, ci_high = _paired_permutation_test(
            all_y, preds_atew, preds_base, n_perm=n_perm, n_boot=5000
        )
        p_str = f"< {1.0 / n_perm:.4f}" if exceed_count == 0 else f"{p_val:.4f}"
        sig_stars = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "ns"))
        print(
            f"    ATEW-XGB vs {base}:  Delta R2 = {diff:+.3f} "
            f"[{ci_low:+.3f}, {ci_high:+.3f}],  p = {p_str} ({sig_stars})"
        )
        sig_results.append({
            "comparison": f"ATEW-XGB vs {base}",
            "delta_R2": round(diff, 3),
            "ci_95_low": round(ci_low, 3),
            "ci_95_high": round(ci_high, 3),
            "p_value": p_str,
            "significance": sig_stars,
        })

    pd.DataFrame(sig_results).to_csv(out_dir / "significance_test.csv", index=False)
    print(f"Saved significance tests to {out_dir / 'significance_test.csv'}")
    print(f"Outputs written to {out_dir}")

if __name__ == "__main__":
    main()
