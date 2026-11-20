"""Visualizer for EWTA recency-weighting decay profiles and trajectory aggregation."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data_loader import load_surveys
from src.output_utils import get_output_dirs

plt.style.use("seaborn-v0_8-whitegrid")
sns.set_context("paper", font_scale=1.5)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Computer Modern Roman", "DejaVu Serif"],
    "text.usetex": False,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "legend.title_fontsize": 12,
    "figure.titlesize": 18,
    "lines.linewidth": 2.5,
    "axes.grid": True,
    "grid.alpha": 0.6,
    "grid.linestyle": "--",
})

def main():
    # Illustrative complete trajectory: Student ID 60 (Cohort II)
    df_y2 = load_surveys("cohort_2")
    student_df = df_y2[df_y2["ID"] == 60].sort_values("week")
    
    raw_motivation = student_df["motivation"].values.astype(float)
        
    T = len(raw_motivation)
    weeks = np.arange(1, T + 1)

    def compute_ewta_mean_trajectory(vals, alpha):
        ewta_means = []
        for t_idx in range(1, len(vals) + 1):
            sub_vals = vals[:t_idx]
            w = alpha ** (np.arange(t_idx)[::-1])
            ewta_means.append(np.sum(w * sub_vals) / np.sum(w))
        return np.array(ewta_means)

    ewta_05 = compute_ewta_mean_trajectory(raw_motivation, 0.5)
    ewta_08 = compute_ewta_mean_trajectory(raw_motivation, 0.8)
    uniform_mean = np.array([np.mean(raw_motivation[:t_idx]) for t_idx in range(1, len(raw_motivation) + 1)])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), dpi=600)

    ax0 = axes[0]
    alphas_to_plot = [0.5, 0.8, 0.95]
    colors_decay = ["#c0392b", "#2980b9", "#27ae60"]
    
    for alpha, col in zip(alphas_to_plot, colors_decay):
        weights = alpha ** (T - weeks)
        ax0.plot(weeks, weights, marker="o", linestyle="-", 
                 label=f"$\\alpha = {alpha}$", color=col, linewidth=2.0, markersize=6)
                 
    ax0.set_xlabel("Week ($t$)")
    ax0.set_ylabel("Weight $w_t = \\alpha^{T-t}$")
    ax0.set_title("EWTA Weight Decay (at Week $T=11$)")
    ax0.set_xticks(weeks)
    ax0.set_ylim(-0.05, 1.05)
    ax0.legend(title="Decay Factor ($\\alpha$)")
    sns.despine(ax=ax0, left=True, bottom=True)

    ax1 = axes[1]
    ax1.plot(weeks, raw_motivation, marker="s", linestyle="--", color="#7f8c8d", 
             label="Raw Motivation ($E_{i,t,j}$)", alpha=0.7, zorder=2, markersize=6)
    ax1.plot(weeks, uniform_mean, marker="v", linestyle=":", color="#2c3e50", 
             label="Uniform Mean (baseline)", zorder=3, markersize=6)
    ax1.plot(weeks, ewta_08, marker="o", linestyle="-", color="#2980b9", 
             label="EWTA Mean ($\\alpha=0.8$)", zorder=4, markersize=6)
    ax1.plot(weeks, ewta_05, marker="^", linestyle="-", color="#c0392b", 
             label="EWTA Mean ($\\alpha=0.5$)", zorder=5, markersize=6)

    ax1.set_xlabel("Week ($t$)")
    ax1.set_ylabel("Affective Score (Motivation)")
    ax1.set_title("Affective Trajectory Aggregation Comparison")
    ax1.set_xticks(weeks)
    ax1.set_ylim(0.5, 5.5)
    ax1.legend(loc="lower left")
    sns.despine(ax=ax1, left=True, bottom=True)

    plt.tight_layout()

    _, out_dir = get_output_dirs(ROOT)
    plt.savefig(out_dir / "ewta_demo.png", bbox_inches="tight", dpi=600)
    plt.close()
    
if __name__ == "__main__":
    main()
