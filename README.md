# Affective Trajectory Early Warning

Repository containing multiview data ingestion, temporal affective feature engineering, cross-cohort evaluation scripts, result tables, and figures for early prediction of academic performance and at-risk student identification.


---

## Overview

Rather than relying on static semester averages or invasive physiological sensors, this codebase models students' weekly self-reported affective dynamics alongside attendance logs and continuous assessment grades across six dimensions (`satisfaction`, `effort`, `motivation`, `difficulty`, `workload`, and `frustration`):

* **Exponentially Weighted Temporal Aggregation:** Recency-weighted trajectory mean, volatility, linear trend, and recency shift across observed weeks.
* **Psychological Trajectory Indicators:** Consecutive frustration streaks, difficulty–motivation gap, and multi-item engagement decay (`effort`, `satisfaction`, `motivation`).
* **Leave-One-Cohort-Out (LOCO) Evaluation:** Bidirectional cross-cohort validation across two consecutive academic years ($N = 155$ enrolled; $N = 143$ active evaluated students) with nested 5-fold cross-validation, SHAP attributions, ablation experiments, and early-warning classification metrics.

---

## Repository Structure

```text
.
├── data/
│   └── README.md                       # Dataset schema and Zenodo instructions
├── src/
│   ├── data_loader.py                  # Multiview panel ingestion and cohort alignment
│   ├── features.py                     # Temporal and trajectory feature extraction (58 features)
│   ├── output_utils.py                 # Safe output directory resolution (--overwrite)
│   └── __init__.py
├── scripts/
│   ├── dataset_descriptive_stats.py    # Cohort descriptive statistics & missingness analysis
│   ├── run_sota_comparison.py          # Cross-cohort benchmark & statistical significance tests
│   ├── run_ablation_study.py           # Component & temporal-baseline ablation (Weeks 4-11)
│   ├── run_shap_analysis.py            # SHAP TreeExplainer feature attributions
│   ├── run_classification_metrics.py   # Operational binary at-risk metrics & confusion matrices
│   ├── benchmark_runtime.py            # Training time and inference latency benchmark
│   ├── generate_plots.py               # Benchmark and early-warning comparison plots
│   ├── generate_dataset_plots.py       # Weekly affective evolution plot
│   ├── generate_ewta_demo.py           # Exponential decay weighting demo plot
│   ├── generate_confusion_matrices.py  # Confusion matrix grid plot
│   └── run_pipeline.py                 # End-to-end execution script
├── results/
│   ├── figures/                        # Output figures (PNG)
│   └── tables/                         # Summary CSV/TXT result tables
├── LICENSE                             # MIT License
├── pyproject.toml                      # Python version constraint (>=3.12,<3.13)
├── requirements.txt                    # Pinned Python/CUDA dependencies
└── README.md
```

---

## Dataset

The longitudinal dataset is hosted separately on Zenodo: **[DOI: 10.5281/zenodo.23403842](https://doi.org/10.5281/zenodo.23403842)**.

It comprises two academic cohorts (**Cohort I**, $N = 73$; **Cohort II**, $N = 82$) with three aligned input views per cohort:
1. **Affective surveys (`surveys/`):** 11 weekly self-reports (`week_01.csv` to `week_11.csv`, totaling 1,112 submissions and 6,672 ratings).
2. **Attendance (`attendance.csv`):** Practical session attendance logs across 12–13 sessions (1,778 recorded observations across 1,933 slots).
3. **Academic grades (`continuous_grades.csv`, `final_grades.csv`):** Four programming practices ($P1$–$P4$), two midterm exams (`midterm_1`, `midterm_2`), and the continuous final grade $y \in [0, 10]$.

Download `data.zip` from Zenodo and unzip it from the repository root (`unzip data.zip`), which extracts `cohort_1/` and `cohort_2/` into `data/` (`data/cohort_1/`, `data/cohort_2/`). Further schema details are provided in [`data/README.md`](data/README.md).

---

## Installation & Usage

Requires Python 3.12 (`>=3.12,<3.13`) and the pinned dependencies in [`requirements.txt`](requirements.txt):

```bash
pip install -r requirements.txt
```

Run the full evaluation and figure generation pipeline:

```bash
python scripts/run_pipeline.py
```

---

## Results

All tables and figures generated under the environment pinned in [`requirements.txt`](requirements.txt) on Linux x86_64 with an NVIDIA GeForce RTX 3090 GPU (Python 3.12, CUDA 12.8) are provided under `results/`:
* **`results/tables/`:** Descriptive statistics, Week 11 cross-cohort benchmark metrics, paired permutation significance tests, Weeks 4–11 early-warning and ablation sweeps, runtime/latency measurements, binary at-risk classification metrics/confusion matrices (threshold $\hat{y}_i < 5.0$), and top SHAP features.
* **`results/figures/`:** Script-generated plots (dataset evolution, decay weighting comparison, performance bar and early-warning progression plots, confusion matrices, and SHAP summary beeswarm plot) alongside the conceptual and architectural diagrams (`Emotions_plot1.png` to `Emotions_plot3.png`).

---

## License

Source code is released under the [MIT License](LICENSE). The accompanying benchmark dataset hosted on Zenodo (`10.5281/zenodo.23403842`) is distributed separately under the **CC-BY-NC-ND-4.0** license.
