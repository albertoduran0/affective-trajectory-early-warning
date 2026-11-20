"""
Generate confusion matrix grid for ATEW-XGB (Week 5 vs. Week 11) under LOCO protocol.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
from src.output_utils import get_output_dirs, parse_overwrite_flag, resolve_input_table

plt.style.use("seaborn-v0_8-white")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "Computer Modern Roman"],
    "text.usetex": False,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.titlesize": 15,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_confusion_matrix(df: pd.DataFrame, week: str, method: str = "ATEW-XGB") -> np.ndarray:
    """Return 2x2 matrix with rows Actual [At-Risk, Passing] and cols Predicted [At-Risk, Passing]."""
    sub = df[(df["Week"] == week) & (df["Method"] == method)]
    if sub.empty:
        raise ValueError(f"Missing confusion matrix entry for {method} at {week}")
    row = sub.iloc[0]
    return np.array([[int(row["TP"]), int(row["FN"])], [int(row["FP"]), int(row["TN"])]])


def plot_confusion_matrices():
    """Plot side-by-side Week 5 and Week 11 confusion matrices for ATEW-XGB."""
    overwrite = parse_overwrite_flag()
    _, fig_dir = get_output_dirs(ROOT_DIR, overwrite=overwrite)
    cm_path = resolve_input_table(ROOT_DIR, "confusion_matrices.csv", prefer_rerun=not overwrite)
    cm_df = pd.read_csv(cm_path)
    cm_w5 = load_confusion_matrix(cm_df, "Week 5", "ATEW-XGB")
    cm_w11 = load_confusion_matrix(cm_df, "Week 11", "ATEW-XGB")

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.3), dpi=300)

    classes = ["At-Risk (< 5.0)", "Passing (≥ 5.0)"]
    cell_names = [["TP", "FN"], ["FP", "TN"]]

    configs = [
        {
            "ax": axes[0],
            "cm": cm_w5,
            "title": "(a) Week 5: Early-Warning Horizon",
        },
        {
            "ax": axes[1],
            "cm": cm_w11,
            "title": "(b) Week 11: Final Evaluation Horizon",
        },
    ]

    for cfg in configs:
        ax = cfg["ax"]
        cm = cfg["cm"]

        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = cm / row_sums

        sns.heatmap(
            cm_norm,
            annot=False,
            cmap="Blues",
            cbar=False,
            vmin=0,
            vmax=1.0,
            ax=ax,
            linewidths=1.5,
            linecolor="white",
            square=True,
        )

        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                pct = cm_norm[i, j] * 100
                tag = cell_names[i][j]

                text_color = "white" if cm_norm[i, j] > 0.55 else "#1a252f"

                cell_text = f"{tag} = {val}\n({pct:.1f}%)"
                ax.text(
                    j + 0.5, i + 0.5,
                    cell_text,
                    ha="center", va="center",
                    color=text_color,
                    fontsize=12,
                    fontweight="bold",
                )

        ax.set_title(cfg["title"], pad=14)
        ax.set_xlabel("Predicted Class", labelpad=8, fontweight="bold")
        ax.set_xticklabels(classes)
        ax.set_yticklabels(classes, rotation=0, va="center")

    axes[0].set_ylabel("Actual Class", labelpad=8, fontweight="bold")
    axes[1].set_ylabel("")

    plt.suptitle("ATEW-XGB Confusion Matrices for At-Risk Identification (LOCO Protocol)", y=0.98, fontweight="bold")

    plt.tight_layout()
    plt.subplots_adjust(top=0.79, bottom=0.12, wspace=0.25)

    out_png = fig_dir / "confusion_matrices_at_risk.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Figure saved successfully:\n- {out_png}")


if __name__ == "__main__":
    plot_confusion_matrices()
