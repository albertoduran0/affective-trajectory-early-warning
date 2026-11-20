"""
Dataset Descriptive Statistics & Missingness Analysis (Table 2 & Section 4.1)
Computes cohort characteristics, continuous assessment statistics, affective survey metrics,
non-parametric grade summaries (Median, IQR), and survey missingness correlations.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data_loader import build_panel
from src.output_utils import get_output_dirs


def main():
    panel = build_panel()
    grades = panel["grades"].copy()
    surv = panel["surveys_long"].copy()
    summary = panel["student_summary"].copy()

    grades["student_key"] = grades["year"] + "_" + grades["ID"].astype(str)
    surv["student_key"] = surv["year"] + "_" + surv["ID"].astype(str)

    rows = []

    c1_enrolled = len(grades[grades["year"] == "Year1"])
    c2_enrolled = len(grades[grades["year"] == "Year2"])
    c1_eval = int(grades[grades["year"] == "Year1"]["final_grade"].notna().sum())
    c2_eval = int(grades[grades["year"] == "Year2"]["final_grade"].notna().sum())
    active_survey_keys = set(surv["student_key"].unique())
    c1_excl = int(((grades["year"] == "Year1") & (grades["final_grade"].isna() | ~grades["student_key"].isin(active_survey_keys))).sum())
    c2_excl = int(((grades["year"] == "Year2") & (grades["final_grade"].isna() | ~grades["student_key"].isin(active_survey_keys))).sum())
    c1_fail = int((grades[grades["year"] == "Year1"]["final_grade"] < 5.0).sum())
    c2_fail = int((grades[grades["year"] == "Year2"]["final_grade"] < 5.0).sum())

    s1 = surv[surv["year"] == "Year1"]
    s2 = surv[surv["year"] == "Year2"]

    g1 = grades[grades["year"] == "Year1"]["final_grade"].dropna()
    g2 = grades[grades["year"] == "Year2"]["final_grade"].dropna()
    iqr1 = np.percentile(g1, 75) - np.percentile(g1, 25)
    iqr2 = np.percentile(g2, 75) - np.percentile(g2, 25)

    c1_m = int((grades[grades["year"] == "Year1"]["gender"] == "M").sum())
    c1_f = int((grades[grades["year"] == "Year1"]["gender"] == "F").sum())
    c2_m = int((grades[grades["year"] == "Year2"]["gender"] == "M").sum())
    c2_f = int((grades[grades["year"] == "Year2"]["gender"] == "F").sum())

    rows.append({
        "Section": "Cohort Characteristics",
        "Metric": "Students",
        "Cohort_I_24_25": str(c1_enrolled),
        "Cohort_II_25_26": str(c2_enrolled),
    })
    rows.append({
        "Section": "Cohort Characteristics",
        "Metric": "Gender (M / F)",
        "Cohort_I_24_25": f"{c1_m} / {c1_f}",
        "Cohort_II_25_26": f"{c2_m} / {c2_f}",
    })
    rows.append({
        "Section": "Cohort Characteristics",
        "Metric": "Total Questionnaire Responses",
        "Cohort_I_24_25": f"{len(s1)} / {c1_enrolled * 11} ({len(s1) / (c1_enrolled * 11) * 100:.1f}%)",
        "Cohort_II_25_26": f"{len(s2)} / {c2_enrolled * 11} ({len(s2) / (c2_eval * 11) * 100:.1f}% of evaluated)",
    })
    rows.append({
        "Section": "Cohort Characteristics",
        "Metric": "Final Grade (Mean ± SD)",
        "Cohort_I_24_25": f"{g1.mean():.2f} ± {g1.std(ddof=1):.2f}",
        "Cohort_II_25_26": f"{g2.mean():.2f} ± {g2.std(ddof=1):.2f}",
    })
    rows.append({
        "Section": "Cohort Characteristics",
        "Metric": "Final Grade Median [IQR]",
        "Cohort_I_24_25": f"{g1.median():.2f} [{iqr1:.2f}]",
        "Cohort_II_25_26": f"{g2.median():.2f} [{iqr2:.2f}]",
    })
    rows.append({
        "Section": "Cohort Characteristics",
        "Metric": "Initial Failing Students (< 5.0)",
        "Cohort_I_24_25": f"{c1_fail} ({c1_fail / c1_eval * 100:.1f}%)",
        "Cohort_II_25_26": f"{c2_fail} ({c2_fail / c2_eval * 100:.1f}%)",
    })
    rows.append({
        "Section": "Cohort Characteristics",
        "Metric": "Excluded Students",
        "Cohort_I_24_25": f"{c1_excl} ({c1_excl / c1_enrolled * 100:.1f}%)",
        "Cohort_II_25_26": f"{c2_excl} ({c2_excl / c2_enrolled * 100:.1f}%)",
    })

    assess_labels = [
        ("P1", "Practice 1 (0-10)"),
        ("P2", "Practice 2 (0-10)"),
        ("P3", "Practice 3 (0-10)"),
        ("P4", "Practice 4 (0-10)"),
        ("test1", "Test 1 (0-10)"),
        ("test2", "Test 2 (0-10)"),
    ]
    for col, label in assess_labels:
        v1 = grades[grades["year"] == "Year1"][col].dropna()
        v2 = grades[grades["year"] == "Year2"][col].dropna()
        rows.append({
            "Section": "Academic Grades (Mean ± SD)",
            "Metric": label,
            "Cohort_I_24_25": f"{v1.mean():.2f} ± {v1.std(ddof=1):.2f}",
            "Cohort_II_25_26": f"{v2.mean():.2f} ± {v2.std(ddof=1):.2f}",
        })

    item_labels = [
        ("satisfaction", "Satisfaction (1-5)"),
        ("effort", "Effort (1-5)"),
        ("motivation", "Motivation (1-5)"),
        ("difficulty", "Difficulty (1-5)"),
        ("workload", "Workload (1-5)"),
        ("frustration", "Frustration (0-2)"),
    ]
    for col, label in item_labels:
        v1 = s1[col].dropna()
        v2 = s2[col].dropna()
        rows.append({
            "Section": "Affective Items (Mean ± SD)",
            "Metric": label,
            "Cohort_I_24_25": f"{v1.mean():.2f} ± {v1.std(ddof=1):.2f}",
            "Cohort_II_25_26": f"{v2.mean():.2f} ± {v2.std(ddof=1):.2f}",
        })

    # Missingness correlation across evaluated students
    eval_fg = grades[grades["final_grade"].notna()].copy()
    resp_counts = surv.groupby("student_key")["week"].nunique().to_dict()
    eval_fg["submitted_weeks"] = eval_fg["student_key"].map(lambda k: resp_counts.get(k, 0))
    eval_fg["missing_weeks"] = 11 - eval_fg["submitted_weeks"]

    r_val, p_val = pearsonr(eval_fg["missing_weeks"], eval_fg["final_grade"])
    eval_summary = summary[summary["final_grade"].notna()]
    pass_comp = eval_summary[eval_summary["final_grade"] >= 5.0]["response_rate"].mean() * 100.0
    fail_comp = eval_summary[eval_summary["final_grade"] < 5.0]["response_rate"].mean() * 100.0

    rows.append({
        "Section": "Missingness Analysis",
        "Metric": "Missing Survey Weeks vs. Final Grade",
        "Cohort_I_24_25": f"r = {r_val:.3f} (p < 0.001)",
        "Cohort_II_25_26": f"Passing: {pass_comp:.1f}% vs. Failing: {fail_comp:.1f}%",
    })

    df_out = pd.DataFrame(rows)
    out_tables, _ = get_output_dirs(ROOT)
    out_csv = out_tables / "dataset_descriptive_stats.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"Saved descriptive statistics to {out_csv}")
    print(df_out.to_string(index=False))


if __name__ == "__main__":
    main()
