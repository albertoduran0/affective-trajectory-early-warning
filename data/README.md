# Multiview Educational Dataset for Student Performance Prediction

This directory is the dataset root for the open-access heterogeneous multiview benchmark dataset (Zenodo Version v1, DOI: [10.5281/zenodo.23403842](https://doi.org/10.5281/zenodo.23403842)). To keep code and data separated, raw data files are hosted on Zenodo rather than tracked in GitHub: download `data.zip` from Zenodo and unzip it from the repository root (`unzip data.zip`), which extracts `cohort_1/` and `cohort_2/` directly into this `data/` folder (avoid `unzip data.zip -d data/`, which would create a nested `data/data/` directory).

## Cohort Structure
- **cohort_1**: Academic Year 2024/25 ($N = 73$ students)
- **cohort_2**: Academic Year 2025/26 ($N = 82$ students)

Total enrolled sample: $N = 155$. Across cohorts, $145$ students received initial transcript grades ($73$ in Cohort I and $72$ in Cohort II, with $15$ scoring $< 5.0$ prior to the Test 1 retake: $6$ in Cohort I and $9$ in Cohort II). Twelve enrolled students in Cohort II were excluded from longitudinal modeling:
- $10$ lacked a final course grade (never completed the theory examinations / final grading: $6$ submitted neither weekly surveys nor continuous assessments, $2$ submitted only early surveys in weeks 1–4 without assessments, and $2$ — IDs `31` and `32` — submitted early surveys in weeks 1–3 and practical assignments up to week 10 without sitting the theory exams);
- $2$ (IDs `12` and `41`) received final grades but submitted zero weekly surveys and recorded $0\%$ attendance (`No` across all 11 held practical sessions in `attendance.csv`).

This leaves $N = 143$ active evaluated students ($73$ in Cohort I and $70$ in Cohort II). Note that `cohort_2/final_grades.csv` stores the pre-retake transcript marks ($15$ cases $< 5.0$ across the $143$ active students), which serve as the continuous regression target, whereas operational binary at-risk classification in `scripts/run_classification_metrics.py` evaluates post-retake final pass/fail status ($11$ failing cases $< 5.0$ and $132$ passing cases after Cohort II IDs `22`, `38`, `72`, and `78` cleared the Test 1 retake threshold).

## Files per Cohort
1. **surveys/**: 11 weekly survey CSV files (`week_01.csv` to `week_11.csv`)
   - `ID`: Anonymized student identifier (integer).
   - `sex`: Student sex (`M` / `F`).
   - `satisfaction`, `effort`, `motivation`, `difficulty`, `workload`: Likert scale ratings (1 to 5).
   - `frustration`: Personal frustration indicator (`No`, `Moderate`, `Yes`).
2. **attendance.csv**: Laboratory session attendance logs (`Yes`, `No`, or empty for holidays).
3. **continuous_grades.csv**: Continuous assessment grades in [0, 10]:
   - `P1`, `P2_individual`, `P2_group`, `midterm_1`, `P3`, `P4`, `midterm_2`.
   - Practice 2 consolidated grade: $P2 = 0.8 \times P2\_group + 0.2 \times P2\_individual$.
4. **final_grades.csv**: Continuous regression target:
   - `final_grade`: Final course mark $y \in [0, 10]$.
