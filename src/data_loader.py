"""Data loader module for longitudinal multiview educational dataset."""
from __future__ import annotations
import re
from pathlib import Path
from typing import Optional, List

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parent.parent / "data"

COHORT_MAP = {
    "cohort_1": "cohort_1",
    "cohort_2": "cohort_2",
    "Year1": "cohort_1",
    "Year2": "cohort_2",
}

YEAR_LABEL_MAP = {
    "cohort_1": "Year1",
    "cohort_2": "Year2",
}

AFFECTIVE_ITEMS = ["satisfaction", "effort", "motivation", "difficulty", "workload"]
FRUSTRATION_COL = "frustration"
ALL_ITEMS = AFFECTIVE_ITEMS + [FRUSTRATION_COL]

FRUSTRATION_MAP = {
    "no": 0.0,
    "moderate": 1.0,
    "yes": 2.0,
}


def _resolve_cohort(cohort: str) -> tuple[Path, str]:
    """Resolve cohort identifier to folder path and standardized year label."""
    norm = str(cohort).strip()
    if norm not in COHORT_MAP:
        raise ValueError(f"Unknown cohort identifier '{cohort}'. Choose from: {list(COHORT_MAP.keys())}")
    folder_name = COHORT_MAP[norm]
    year_label = YEAR_LABEL_MAP[folder_name]
    cohort_path = DATA_ROOT / folder_name
    if not cohort_path.exists():
        raise FileNotFoundError(
            f"Missing dataset directory '{cohort_path}'. "
            "Download 'data.zip' from Zenodo (https://doi.org/10.5281/zenodo.23403842) "
            "and unzip it from the repository root ('unzip data.zip') so that "
            f"'cohort_1/' and 'cohort_2/' are extracted into '{DATA_ROOT}'."
        )
    return cohort_path, year_label


def load_surveys(cohort: str = "cohort_1") -> pd.DataFrame:
    """
    Load all weekly survey CSVs for a given cohort into a long-format DataFrame.

    Returns
    -------
    DataFrame with columns:
        year, ID, gender, week, satisfaction, effort, motivation,
        difficulty, workload, frustration
    """
    c_dir, year_label = _resolve_cohort(cohort)
    survey_dir = c_dir / "surveys"
    csv_files = sorted(survey_dir.glob("week_*.csv"))

    dfs = []
    for csv_path in csv_files:
        m = re.search(r"week_(\d+)", csv_path.name)
        week_num = int(m.group(1)) if m else -1

        df = pd.read_csv(csv_path)
        if "sex" in df.columns:
            df = df.rename(columns={"sex": "gender"})

        df["week"] = week_num
        df["year"] = year_label
        df["ID"] = df["ID"].astype(int)

        # Standardize frustration scale: No=0, Moderate=1, Yes=2
        if "frustration" in df.columns:
            df["frustration"] = (
                df["frustration"]
                .astype(str)
                .str.strip()
                .str.lower()
                .map(FRUSTRATION_MAP)
            )

        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    cols = ["year", "ID", "gender", "week"] + ALL_ITEMS
    return out[[c for c in cols if c in out.columns]]


def load_attendance(cohort: str = "cohort_1") -> pd.DataFrame:
    """
    Load attendance CSV and return a long-format DataFrame with columns:
        year, ID, gender, week, attended (0/1)
    """
    c_dir, year_label = _resolve_cohort(cohort)
    df = pd.read_csv(c_dir / "attendance.csv")
    if "sex" in df.columns:
        df = df.rename(columns={"sex": "gender"})

    session_cols = [c for c in df.columns if c.startswith("session_")]
    records = []

    for _, row in df.iterrows():
        sid = int(row["ID"])
        gender = row.get("gender", np.nan)
        for i, col in enumerate(session_cols, start=1):
            val = row[col]
            if pd.isna(val):
                continue
            s_val = str(val).strip().lower()
            attended = 1.0 if s_val in ("yes", "1", "1.0", "true") else 0.0
            records.append({
                "year": year_label,
                "ID": sid,
                "gender": gender,
                "week": i,
                "attended": attended,
            })

    return pd.DataFrame(records)


def load_grades(cohort: str = "cohort_1") -> pd.DataFrame:
    """
    Load continuous_grades.csv and final_grades.csv, merge into a student-level DataFrame.

    Returns
    -------
    DataFrame with columns:
        year, ID, gender, P1, P2, P2_individual, P2_group, test1,
        P3, P4, test2, final_grade
    """
    c_dir, year_label = _resolve_cohort(cohort)
    cg = pd.read_csv(c_dir / "continuous_grades.csv")
    fg = pd.read_csv(c_dir / "final_grades.csv")

    if "sex" in cg.columns:
        cg = cg.rename(columns={"sex": "gender"})

    merged = cg.merge(fg[["ID", "final_grade"]], on="ID", how="outer")
    merged["year"] = year_label
    merged["ID"] = merged["ID"].astype(int)

    # Compute consolidated P2 if components exist: 0.8 * group + 0.2 * individual
    if "P2_group" in merged.columns and "P2_individual" in merged.columns:
        merged["P2"] = 0.8 * merged["P2_group"] + 0.2 * merged["P2_individual"]

    # Align evaluation names: midterm_1 -> test1, midterm_2 -> test2
    merged = merged.rename(columns={"midterm_1": "test1", "midterm_2": "test2"})

    cols = ["year", "ID", "gender", "P1", "P2", "P2_individual", "P2_group",
            "test1", "P3", "P4", "test2", "final_grade"]
    return merged[[c for c in cols if c in merged.columns]]


def build_panel(cohorts: Optional[List[str]] = None) -> dict[str, pd.DataFrame]:
    """
    Build the full unified dataset across cohorts.

    Returns
    -------
    dict with keys:
        - 'surveys_long':   Long-format surveys (student x week)
        - 'attendance_long': Long-format attendance (student x session/week)
        - 'grades':         Student-level continuous and final grades
        - 'merged_long':    Surveys + attendance merged panel
        - 'student_summary': One row per student with aggregated affective features
    """
    if cohorts is None:
        cohorts = ["cohort_1", "cohort_2"]

    surveys = pd.concat([load_surveys(c) for c in cohorts], ignore_index=True)
    attendance = pd.concat([load_attendance(c) for c in cohorts], ignore_index=True)
    grades = pd.concat([load_grades(c) for c in cohorts], ignore_index=True)

    merged = surveys.merge(
        attendance[["year", "ID", "week", "attended"]],
        on=["year", "ID", "week"],
        how="outer",
    )

    agg = {}
    for item in ALL_ITEMS:
        if item in surveys.columns:
            agg[f"{item}_mean"] = (item, "mean")
            agg[f"{item}_std"] = (item, "std")
            agg[f"{item}_trend"] = (item, lambda x: _linreg_slope(x))

    summary_surveys = surveys.groupby(["year", "ID"]).agg(**agg).reset_index()

    total_weeks = surveys.groupby(["year", "ID"])["week"].nunique().reset_index()
    total_weeks.columns = ["year", "ID", "weeks_responded"]
    summary_surveys = summary_surveys.merge(total_weeks, on=["year", "ID"], how="left")
    summary_surveys["response_rate"] = summary_surveys["weeks_responded"] / 11.0

    att_summary = attendance.groupby(["year", "ID"]).agg(
        attendance_total=("attended", "sum"),
        attendance_rate=("attended", "mean"),
    ).reset_index()

    frust = surveys[surveys[FRUSTRATION_COL].notna()].copy()
    frust_counts = frust.groupby(["year", "ID"]).agg(
        frustration_count=("frustration", lambda x: (x > 0).sum()),
        frustration_high_count=("frustration", lambda x: (x == 2).sum()),
    ).reset_index()

    student_df = summary_surveys.merge(att_summary, on=["year", "ID"], how="left")
    student_df = student_df.merge(frust_counts, on=["year", "ID"], how="left")

    grade_cols = [c for c in ["year", "ID", "gender", "P1", "P2",
                              "test1", "P3", "P4", "test2", "final_grade"] if c in grades.columns]
    student_df = student_df.merge(
        grades[grade_cols],
        on=["year", "ID"],
        how="left",
    )

    return {
        "surveys_long": surveys,
        "attendance_long": attendance,
        "grades": grades,
        "merged_long": merged,
        "student_summary": student_df,
    }


def _linreg_slope(series: pd.Series) -> float:
    """Compute simple linear regression slope over a series."""
    vals = series.dropna().values
    if len(vals) < 3:
        return np.nan
    x = np.arange(len(vals), dtype=float)
    try:
        slope = np.polyfit(x, vals, 1)[0]
        return float(slope)
    except Exception:
        return np.nan


if __name__ == "__main__":
    data = build_panel()
    print("Dataset loading summary:")
    for key, df in data.items():
        print(f"\n  {key}: {df.shape[0]} rows x {df.shape[1]} cols")
        print(f"    Columns: {list(df.columns)}")
    print()
    s = data["student_summary"]
    print(f"  Students with final grade: {s['final_grade'].notna().sum()}")
    print(f"  Students by cohort:\n{s.groupby('year')['ID'].nunique()}")
    print(f"\n  Final grade distribution:")
    print(s["final_grade"].describe().to_string())
