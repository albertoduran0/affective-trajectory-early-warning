"""
Feature engineering for ATEW (Affective Trajectory Early Warning).

This module implements the EWTA (Exponentially Weighted Temporal
Aggregation) framework for encoding longitudinal affective survey
data into theory-driven predictive features.

Key operators:
  - EWTA mean/std: exponentially weighted aggregation with learnable decay
  - Frustration cascade: streak-based encoding of persistent negative affect
  - Difficulty-Motivation gap: cross-item trajectory interaction
  - Engagement decay: multi-signal composite trend
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.data_loader import FRUSTRATION_COL, ALL_ITEMS


def _slope(vals):
    """OLS linear trend (slope) over non-NaN values."""
    v = np.array(vals, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 3:
        return np.nan
    return float(np.polyfit(np.arange(len(v)), v, 1)[0])


def _max_streak(vals, threshold=0):
    """Max consecutive weeks above threshold."""
    streak, best = 0, 0
    for v in vals:
        if not np.isnan(v) and v > threshold:
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return best


def _ewta_mean(vals, alpha):
    """Exponentially Weighted Temporal Aggregation — mean.

    Computes:  E_EWTA = sum(alpha^(T-t) * E_t) / sum(alpha^(T-t))

    Lower alpha -> more weight on recent observations (stronger recency bias).
    alpha=1.0 recovers the uniform (arithmetic) mean.
    """
    v = np.array(vals, dtype=float)
    mask = ~np.isnan(v)
    v = v[mask]
    if len(v) < 2:
        return np.mean(v) if len(v) else np.nan
    T = len(v)
    weights = np.array([alpha ** (T - 1 - t) for t in range(T)])
    return float(np.dot(weights, v) / weights.sum())


def _ewta_std(vals, alpha):
    """Exponentially Weighted Temporal Aggregation — standard deviation.

    Captures weighted volatility: high values indicate unstable affective
    trajectories even if the mean looks normal.
    """
    v = np.array(vals, dtype=float)
    mask = ~np.isnan(v)
    v = v[mask]
    if len(v) < 3:
        return np.nan
    T = len(v)
    weights = np.array([alpha ** (T - 1 - t) for t in range(T)])
    w_norm = weights / weights.sum()
    mu = np.dot(w_norm, v)
    var = np.dot(w_norm, (v - mu) ** 2)
    return float(np.sqrt(var))


def _ewta_slope(vals, alpha):
    """Weighted linear slope using EWTA weights.

    Gives more influence to recent (time, value) pairs when estimating
    the affective trajectory direction.
    """
    v = np.array(vals, dtype=float)
    mask = ~np.isnan(v)
    v = v[mask]
    if len(v) < 3:
        return np.nan
    T = len(v)
    t_arr = np.arange(T, dtype=float)
    weights = np.array([alpha ** (T - 1 - t) for t in range(T)])
    w_sum = weights.sum()
    wt = np.dot(weights, t_arr) / w_sum
    wv = np.dot(weights, v) / w_sum
    num = np.dot(weights, (t_arr - wt) * (v - wv))
    den = np.dot(weights, (t_arr - wt) ** 2)
    if abs(den) < 1e-12:
        return 0.0
    return float(num / den)


EWTA_ALPHAS = [0.5, 0.7, 0.85, 0.95]


def build_features(surveys: pd.DataFrame, attendance: pd.DataFrame,
                   grades: pd.DataFrame, max_week: int = 11,
                   ewta_alpha: float = 0.85) -> pd.DataFrame:
    """
    Build student-level feature matrix using data up to `max_week`.

    Parameters
    ----------
    surveys : long-format survey DataFrame
    attendance : long-format attendance DataFrame
    grades : student-level grades DataFrame
    max_week : temporal cutoff for early-warning sweeps
    ewta_alpha : decay parameter for EWTA operators (optimised by CV)
    """
    sv = surveys[surveys["week"] <= max_week].copy()
    att = attendance[attendance["week"] <= max_week].copy()

    rows = []
    for (yr, sid), grp in sv.groupby(["year", "ID"]):
        grp = grp.sort_values("week")
        r = {"year": yr, "ID": sid, "max_week": max_week}

        for item in ALL_ITEMS:
            vals = grp[item].dropna().values
            r[f"{item}_mean"] = np.mean(vals) if len(vals) else np.nan
            r[f"{item}_std"] = np.std(vals) if len(vals) > 1 else np.nan

        for item in ALL_ITEMS:
            vals = grp[item].values
            r[f"{item}_trend"] = _slope(vals)
            r[f"{item}_ewta_mean"] = _ewta_mean(vals, ewta_alpha)
            r[f"{item}_ewta_std"] = _ewta_std(vals, ewta_alpha)
            r[f"{item}_ewta_slope"] = _ewta_slope(vals, ewta_alpha)
            uniform_mean = r[f"{item}_mean"]
            ewta_mean = r[f"{item}_ewta_mean"]
            if pd.notna(uniform_mean) and pd.notna(ewta_mean):
                r[f"{item}_recency_delta"] = ewta_mean - uniform_mean
            else:
                r[f"{item}_recency_delta"] = np.nan

            mid = max_week // 2
            early = grp[grp["week"] <= mid][item].dropna().values
            late = grp[grp["week"] > mid][item].dropna().values
            r[f"{item}_early_mean"] = np.mean(early) if len(early) else np.nan
            r[f"{item}_late_mean"] = np.mean(late) if len(late) else np.nan
            r[f"{item}_shift"] = (
                (np.mean(late) - np.mean(early))
                if len(early) and len(late) else np.nan
            )

        # Reindex over weeks 1..max_week so any missing week becomes NaN and resets the streak
        frust_full = (
            grp.set_index("week")[FRUSTRATION_COL]
            .reindex(range(1, max_week + 1))
            .values
        )
        r["frustration_count"] = int(np.nansum(frust_full > 0))
        r["frustration_high_count"] = int(np.nansum(frust_full == 2))
        r["frustration_streak"] = _max_streak(frust_full, threshold=0)

        if "difficulty" in grp.columns and "motivation" in grp.columns:
            gap = grp["difficulty"].values - grp["motivation"].values
            r["diff_mot_gap_mean"] = float(np.nanmean(gap))
            r["diff_mot_gap_trend"] = _slope(gap)
            r["diff_mot_gap_ewta"] = _ewta_mean(gap, ewta_alpha)
            r["diff_mot_gap_ewta_slope"] = _ewta_slope(gap, ewta_alpha)

        ed_vals = [
            v for v in (
                r.get("effort_trend", np.nan),
                r.get("satisfaction_trend", np.nan),
                r.get("motivation_trend", np.nan),
            )
            if pd.notna(v)
        ]
        r["engagement_decay"] = float(np.mean(ed_vals)) if ed_vals else np.nan

        ed_ewta_vals = [
            v for v in (
                r.get("effort_ewta_slope", np.nan),
                r.get("satisfaction_ewta_slope", np.nan),
                r.get("motivation_ewta_slope", np.nan),
            )
            if pd.notna(v)
        ]
        r["engagement_ewta_decay"] = float(np.mean(ed_ewta_vals)) if ed_ewta_vals else np.nan

        r["weeks_responded"] = len(grp)
        r["response_rate"] = len(grp) / max_week

        s_att = att[(att["year"] == yr) & (att["ID"] == sid)]
        r["attendance_rate"] = s_att["attended"].mean() if len(s_att) else np.nan
        r["attendance_total"] = s_att["attended"].sum() if len(s_att) else 0

        if len(s_att) >= 4:
            mid_a = len(s_att) // 2
            early_a = s_att.iloc[:mid_a]["attended"].mean()
            late_a = s_att.iloc[mid_a:]["attended"].mean()
            r["attendance_decay"] = late_a - early_a
        else:
            r["attendance_decay"] = np.nan

        diff_to_frust = 0
        mot_drop = 0
        if "difficulty" in grp.columns and FRUSTRATION_COL in grp.columns:
            weeks = grp["week"].values
            diff_vals = grp["difficulty"].values
            frust_vals = grp[FRUSTRATION_COL].values
            mot_vals = grp["motivation"].values if "motivation" in grp.columns else np.full_like(diff_vals, np.nan)
            for t in range(len(grp) - 1):
                if int(weeks[t + 1]) - int(weeks[t]) != 1:
                    continue
                if not np.isnan(diff_vals[t]) and not np.isnan(frust_vals[t + 1]):
                    if diff_vals[t] >= 4 and frust_vals[t + 1] > 0:
                        diff_to_frust += 1
                if not np.isnan(mot_vals[t]) and not np.isnan(mot_vals[t + 1]):
                    if (mot_vals[t] - mot_vals[t + 1]) >= 1.0:
                        mot_drop += 1
        r["spm_diff_to_frust"] = diff_to_frust
        r["spm_mot_drop"] = mot_drop

        r["sym_risk_frust"] = 1 if r.get("frustration_streak", 0) >= 2 else 0

        sym_risk_dmg = 0
        if "difficulty" in grp.columns and "motivation" in grp.columns:
            grp_indexed = grp.set_index("week").reindex(range(1, max_week + 1))
            gap_full = grp_indexed["difficulty"].values - grp_indexed["motivation"].values
            # Flag >= 2 consecutive weeks of positive difficulty-motivation gap
            sym_risk_dmg = 1 if _max_streak(gap_full, threshold=0) >= 2 else 0
        r["sym_risk_dmg"] = sym_risk_dmg

        sym_risk_effort = 0
        if "effort" in grp.columns:
            eff_full = (
                grp.set_index("week")["effort"]
                .reindex(range(1, max_week + 1))
                .values
            )
            streak = 0
            for e in eff_full:
                if not np.isnan(e) and e < 3:
                    streak += 1
                    if streak >= 2:
                        sym_risk_effort = 1
                        break
                else:
                    streak = 0
        r["sym_risk_effort"] = sym_risk_effort

        rows.append(r)

    features = pd.DataFrame(rows)

    # Keep every student of the grades panel in the cohort: students without
    # survey responses up to max_week retain their academic/attendance
    # features (affective features NaN). Only students with no survey response
    # and no attended session across the full course leave the panel
    # (see data/README.md).
    answered = set(zip(surveys["year"], surveys["ID"]))
    att_yes = attendance[attendance["attended"] > 0]
    attended_any = set(zip(att_yes["year"], att_yes["ID"]))
    present = set(zip(features["year"], features["ID"]))
    extra = []
    for yr, sid in zip(grades["year"], grades["ID"]):
        if (yr, sid) in present:
            continue
        if (yr, sid) not in answered and (yr, sid) not in attended_any:
            continue
        r = {"year": yr, "ID": sid, "max_week": max_week}
        r["weeks_responded"] = 0
        r["response_rate"] = 0.0
        s_att = att[(att["year"] == yr) & (att["ID"] == sid)]
        r["attendance_rate"] = s_att["attended"].mean() if len(s_att) else np.nan
        r["attendance_total"] = s_att["attended"].sum() if len(s_att) else 0
        if len(s_att) >= 4:
            mid_a = len(s_att) // 2
            early_a = s_att.iloc[:mid_a]["attended"].mean()
            late_a = s_att.iloc[mid_a:]["attended"].mean()
            r["attendance_decay"] = late_a - early_a
        else:
            r["attendance_decay"] = np.nan
        r["spm_diff_to_frust"] = 0
        r["spm_mot_drop"] = 0
        r["sym_risk_frust"] = 0
        r["sym_risk_dmg"] = 0
        r["sym_risk_effort"] = 0
        extra.append(r)
    if extra:
        features = pd.concat([features, pd.DataFrame(extra)], ignore_index=True)

    grade_cols = ["year", "ID", "gender", "P1", "P2",
                  "test1", "P3", "P4", "test2", "final_grade"]
    grade_cols = [c for c in grade_cols if c in grades.columns]
    
    masked_grades = grades[grade_cols].copy()
    
    if "test2" in masked_grades.columns:
        masked_grades["test2"] = np.nan  # test2 is never available during weeks 4-11
    if max_week < 10:
        if "P3" in masked_grades.columns:
            masked_grades["P3"] = np.nan
        if "P4" in masked_grades.columns:
            masked_grades["P4"] = np.nan
    if max_week < 8:
        if "test1" in masked_grades.columns:
            masked_grades["test1"] = np.nan
    if max_week < 6:
        if "P2" in masked_grades.columns:
            masked_grades["P2"] = np.nan
    if max_week < 4:
        if "P1" in masked_grades.columns:
            masked_grades["P1"] = np.nan

    features = features.merge(masked_grades, on=["year", "ID"], how="left")

    return features


def get_ensemble_lit_features():
    """ENSEMBLE_LIT: static means + attendance + partial grades (literature standard)."""
    feats = [f"{item}_mean" for item in ALL_ITEMS]
    feats += ["attendance_rate", "response_rate"]
    feats += ["P1", "P2", "test1", "P3", "P4"]
    return feats


def get_atew_features():
    """ATEW: ENSEMBLE_LIT base + EWTA trajectory features (D = 58 features).

    Includes 8 non-affective features (``P1``, ``P2``, ``test1``, ``P3``, ``P4``,
    ``attendance_rate``, ``attendance_decay``, ``response_rate``)—omitting the
    linearly proportional counts ``attendance_total`` and ``weeks_responded``
    from ``get_no_affect_features()``—together with 50 affective/trajectory
    features:
      - 6 uniform means (``*_mean``) + 6 uniform slopes (``*_trend``) + 6 half-term shifts (``*_shift``)
      - 24 EWTA statistics: ``*_ewta_mean``, ``*_ewta_std``, ``*_ewta_slope``, ``*_recency_delta``
      - 8 psychological trajectory operators: ``frustration_count``, ``frustration_streak``,
        4 ``diff_mot_gap_*`` features, ``engagement_decay``, ``engagement_ewta_decay``
    """
    feats = get_ensemble_lit_features()
    for item in ALL_ITEMS:
        feats += [
            f"{item}_ewta_mean", f"{item}_ewta_std",
            f"{item}_ewta_slope", f"{item}_recency_delta",
        ]
    for item in ALL_ITEMS:
        feats += [f"{item}_trend", f"{item}_shift"]
    feats += [
        "frustration_count", "frustration_streak",
        "diff_mot_gap_mean", "diff_mot_gap_trend",
        "diff_mot_gap_ewta", "diff_mot_gap_ewta_slope",
        "engagement_decay", "engagement_ewta_decay",
        "attendance_decay",
    ]
    seen = set()
    unique = []
    for f in feats:
        if f not in seen:
            seen.add(f)
            unique.append(f)
    return unique


def get_no_affect_features():
    """No-Affect baseline: academic grades + attendance only (no survey data).

    This baseline establishes the predictive floor achievable without
    any affective information, isolating the added value of surveys.
    """
    return [
        "attendance_rate", "attendance_total", "attendance_decay",
        "response_rate", "weeks_responded",
        "P1", "P2", "test1", "P3", "P4",
    ]


def get_static_flat_features():
    """Static-Flat representation (Level 1 in taxonomy).

    Uniform (unweighted) means and std of each affective item,
    plus all academic, attendance, and participation features.
    """
    feats = list(get_no_affect_features())
    feats += [f"{item}_mean" for item in ALL_ITEMS]
    feats += [f"{item}_std" for item in ALL_ITEMS]
    return list(dict.fromkeys(feats))


def _discretize_item_state(item: str, val: float) -> str | None:
    """Map an item rating to an ordinal discrete state label."""
    if pd.isna(val):
        return None
    if item == FRUSTRATION_COL:
        return f"{item}_{int(round(val))}"
    if val <= 2:
        return f"{item}_low"
    if val == 3:
        return f"{item}_mid"
    return f"{item}_high"


def mine_frequent_step1_patterns(
    surveys: pd.DataFrame,
    train_ids: set[tuple[str, int]],
    max_week: int = 11,
    min_support: float = 0.20,
) -> pd.DataFrame:
    """Mine frequent immediate step-1 transitions (gap=1) on training students.

    Adapts Huang et al. (2024) sequential pattern mining: discretizes weekly
    affective items, discovers step-1 transitions s_t -> s_{t+1} whose training
    support is >= min_support, and returns per-student transition counts up to max_week.
    """
    sv = surveys[surveys["week"] <= max_week].copy()
    student_transitions: dict[tuple[str, int], dict[str, int]] = {}
    pattern_student_sets: dict[str, set[tuple[str, int]]] = {}

    for (yr, sid), grp in sv.groupby(["year", "ID"]):
        grp = grp.sort_values("week").reset_index(drop=True)
        key = (str(yr), int(sid))
        counts: dict[str, int] = {}
        in_train = key in train_ids

        for idx in range(len(grp) - 1):
            row_t = grp.iloc[idx]
            row_next = grp.iloc[idx + 1]
            if int(row_next["week"]) - int(row_t["week"]) != 1:
                continue
            for item_a in ALL_ITEMS:
                s_a = _discretize_item_state(item_a, row_t[item_a])
                if s_a is None:
                    continue
                for item_b in ALL_ITEMS:
                    s_b = _discretize_item_state(item_b, row_next[item_b])
                    if s_b is None:
                        continue
                    pat = f"spm_pat_{s_a}_to_{s_b}"
                    counts[pat] = counts.get(pat, 0) + 1
                    if in_train:
                        pattern_student_sets.setdefault(pat, set()).add(key)

        student_transitions[key] = counts

    n_train = max(len(train_ids), 1)
    frequent_patterns = sorted(
        pat for pat, holders in pattern_student_sets.items()
        if (len(holders) / n_train) >= min_support
    )

    rows = []
    for (yr, sid), counts in student_transitions.items():
        row = {"year": yr, "ID": sid}
        for pat in frequent_patterns:
            row[pat] = counts.get(pat, 0)
        rows.append(row)
    return pd.DataFrame(rows)


def get_seq_pattern_mining_features():
    """Methodology 1 (Level 3): Sequential Pattern Mining transitions + academic grades."""
    feats = get_no_affect_features()
    feats += ["spm_diff_to_frust", "spm_mot_drop"]
    # Include uniform means to match Static-XGB baseline capacity
    for item in ALL_ITEMS:
        feats.append(f"{item}_mean")
    return list(dict.fromkeys(feats))


def get_neural_symbolic_features():
    """Methodology 2 (Level 3): Symbolic logical flags (used alongside neural encoder)."""
    feats = get_no_affect_features()
    feats += ["sym_risk_frust", "sym_risk_dmg", "sym_risk_effort"]
    return list(dict.fromkeys(feats))
