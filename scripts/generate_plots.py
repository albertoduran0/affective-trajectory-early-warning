"""Publication figure generator for benchmark comparison plots."""
import sys
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.output_utils import get_output_dirs, parse_overwrite_flag, resolve_input_table

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.5)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Computer Modern Roman", "DejaVu Serif"],
    "text.usetex": False,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 13,
    "legend.title_fontsize": 14,
    "figure.titlesize": 20,
    "lines.linewidth": 2.5,
    "axes.grid": True,
    "grid.alpha": 0.6,
    "grid.linestyle": "--",
})

COLORS = {
    "Grades-LR": "#34495e",
    "Academic-Only": "#7f8c8d",
    "Static-XGB": "#f39c12",
    "BiLSTM-Extractor": "#2980b9",
    "Seq-Pattern-Mining": "#8e44ad",
    "Neural-Symbolic": "#16a085",
    "ATEW-XGB": "#c0392b",
}
LINESTYLES = {
    "Grades-LR": (0, (3, 1, 1, 1)),
    "Academic-Only": ":",
    "Static-XGB": "--",
    "BiLSTM-Extractor": "-.",
    "Seq-Pattern-Mining": "-",
    "Neural-Symbolic": "-",
    "ATEW-XGB": "-",
}
MARKERS = {
    "Grades-LR": "P",
    "Academic-Only": "x",
    "Static-XGB": "v",
    "BiLSTM-Extractor": "D",
    "Seq-Pattern-Mining": "^",
    "Neural-Symbolic": "s",
    "ATEW-XGB": "o",
}

METHOD_ORDER = [
    "Grades-LR", "Academic-Only", "Static-XGB", "BiLSTM-Extractor",
    "Seq-Pattern-Mining", "Neural-Symbolic", "ATEW-XGB"
]

def plot_early_warning_r2(df, fig_dir: Path):
    """Generates the main Early Warning R2 Degradation plot."""
    methods = [m for m in METHOD_ORDER if m in df["Method"].unique()]
    weeks = sorted(df["max_week"].unique())

    fig, ax = plt.subplots(figsize=(10, 7))

    for m in methods:
        m_df = df[df["Method"] == m].sort_values("max_week")
        
        ax.plot(m_df["max_week"], m_df["R2_mean"], 
                color=COLORS.get(m, "black"), 
                linestyle=LINESTYLES.get(m, "-"), 
                marker=MARKERS.get(m, "o"),
                linewidth=3.5 if m == "ATEW-XGB" else 2.0,
                markersize=10 if m == "ATEW-XGB" else 7,
                label=m + (" (Ours)" if m == "ATEW-XGB" else ""), 
                zorder=10 if m == "ATEW-XGB" else 5)
                
        ax.fill_between(m_df["max_week"], 
                        m_df["R2_mean"] - m_df["R2_std"]/2,
                        m_df["R2_mean"] + m_df["R2_std"]/2,
                        color=COLORS.get(m, "black"), alpha=0.1, 
                        zorder=1 if m != "ATEW-XGB" else 2)

    ax.set_xlabel("Weeks of Longitudinal Data Available")
    ax.set_ylabel("Predictive Performance ($R^2$ Score)")
    ax.set_xticks(weeks)
    
    sns.despine(left=True, bottom=True)
    
    leg = ax.legend(title="Methodology Level", loc="lower right", frameon=True, 
                    fancybox=True, shadow=True, edgecolor="black")
    leg.get_frame().set_alpha(0.9)

    plt.tight_layout()
    plt.savefig(fig_dir / "early_warning_degradation.png", format='png', dpi=600, bbox_inches="tight")
    plt.close()

def plot_final_performance_bar(df, fig_dir: Path):
    """Bar plot comparing R2 and MAE for Week 11."""
    w11 = df[df["max_week"] == 11].copy()
    w11["Method"] = pd.Categorical(w11["Method"], categories=METHOD_ORDER, ordered=True)
    w11 = w11.sort_values("Method").dropna(subset=["Method"])
    
    methods = w11["Method"].tolist()
    labels = [m + (" (Ours)" if m == "ATEW-XGB" else "") for m in methods]
    r2 = w11["R2_mean"].values
    r2_err = w11["R2_std"].values / 2  # Half-SD error bars
    
    colors = [COLORS.get(m, "gray") for m in methods]

    fig, ax = plt.subplots(figsize=(12, 7.5))
    
    bars = ax.bar(labels, r2, yerr=r2_err, color=colors, capsize=6, 
                  alpha=0.85, edgecolor="black", linewidth=1.5, error_kw=dict(lw=2, capthick=2))
    
    for i, m in enumerate(methods):
        if m == "ATEW-XGB":
            bars[i].set_alpha(1.0)
            bars[i].set_hatch('//')
            bars[i].set_linewidth(2.5)

    ax.set_ylabel("Final $R^2$ Score (Week 11)", fontsize=18, fontweight="bold")
    ax.set_ylim(bottom=0.0, top=1.0)
    sns.despine(left=True, bottom=True)

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=14, fontweight="bold")
    ax.tick_params(axis='y', labelsize=14)

    for i, (v, err) in enumerate(zip(r2, r2_err)):
        ax.text(i, v + err + 0.015, f"{v:.3f}", ha="center", va="bottom", 
                fontweight="bold" if methods[i]=="ATEW-XGB" else "normal",
                fontsize=15)

    plt.tight_layout()
    plt.savefig(fig_dir / "final_performance_bar.png", format='png', dpi=600, bbox_inches="tight")
    plt.close()

def main():
    overwrite = parse_overwrite_flag()
    _, fig_dir = get_output_dirs(ROOT, overwrite=overwrite)
    ew_path = resolve_input_table(ROOT, "sota_early_warning_summary.csv", prefer_rerun=not overwrite)
    sota_path = resolve_input_table(ROOT, "sota_summary.csv", prefer_rerun=not overwrite)

    try:
        ew_df = pd.read_csv(ew_path)
        plot_early_warning_r2(ew_df, fig_dir)
    except FileNotFoundError:
        print(f"Warning: {ew_path} not found.")

    try:
        sota_df = pd.read_csv(sota_path)
        if "R2_std" not in sota_df.columns:
            raise ValueError("Missing required 'R2_std' column in sota_summary.csv")
        plot_final_performance_bar(sota_df, fig_dir)
    except FileNotFoundError:
        print(f"Warning: {sota_path} not found.")

    print(f"Figures saved to {fig_dir}")
if __name__ == "__main__":
    main()
