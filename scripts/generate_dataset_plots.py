"""Visualizer for weekly affective trajectory dynamics across cohorts."""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data_loader import build_panel, ALL_ITEMS
from src.output_utils import get_output_dirs

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
    "figure.titlesize": 20,
    "lines.linewidth": 2.5,
    "axes.grid": True,
    "grid.alpha": 0.6,
    "grid.linestyle": "--",
})

COLORS = {
    "satisfaction": "#2cbd71",
    "effort": "#f39c12",
    "motivation": "#2980b9",
    "difficulty": "#e67e22",
    "workload": "#8e44ad",
    "frustration": "#c0392b",
}
MARKERS = {
    "satisfaction": "o",
    "effort": "s",
    "motivation": "^",
    "difficulty": "v",
    "workload": "D",
    "frustration": "x",
}
LINE_STYLES = {
    "satisfaction": "-",
    "effort": "-",
    "motivation": "-",
    "difficulty": "--",
    "workload": "--",
    "frustration": "-",
}
LABELS = {
    "satisfaction": "Satisfaction",
    "effort": "Effort",
    "motivation": "Motivation",
    "difficulty": "Difficulty",
    "workload": "Workload",
    "frustration": "Frustration (Scaled)",
}

def main():
    _, fig_dir = get_output_dirs(ROOT)
    data = build_panel()
    surveys = data["surveys_long"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), sharey=True)

    cohorts = [("Year1", "Cohort I (Academic Year 2024-25)"), ("Year2", "Cohort II (Academic Year 2025-26)")]

    lines_for_legend = []
    labels_for_legend = []

    for idx, (yr, title) in enumerate(cohorts):
        ax = axes[idx]
        yr_data = surveys[surveys["year"] == yr]

        grp = yr_data.groupby("week")[ALL_ITEMS].mean().reset_index()

        for item in ALL_ITEMS:
            values = grp[item]
            if item == "frustration":
                # Scale frustration (0-2) to (1-5) for visual comparison on the same scale:
                # 0 -> 1.0, 1 -> 3.0, 2 -> 5.0
                values = 1.0 + values * 2.0

            line, = ax.plot(
                grp["week"], values,
                color=COLORS[item],
                marker=MARKERS[item],
                linestyle=LINE_STYLES[item],
                markersize=8,
                linewidth=2.5,
                label=LABELS[item]
            )
            if idx == 0:
                lines_for_legend.append(line)
                labels_for_legend.append(LABELS[item])

        ax.set_xlabel("Instructional Week", fontweight="bold", labelpad=10)
        if idx == 0:
            ax.set_ylabel("Mean Response Value (Likert 1-5)", fontweight="bold", labelpad=10)
        ax.set_title(title, fontweight="bold", fontsize=16, pad=15)
        ax.set_xticks(range(1, 12))
        ax.set_xlim(0.8, 11.2)
        ax.set_ylim(0.8, 5.2)

        ax.yaxis.set_major_locator(mticker.MultipleLocator(1.0))
        ax.xaxis.set_major_locator(mticker.MultipleLocator(1.0))

        sns.despine(ax=ax, left=True, bottom=True)

    fig.legend(
        lines_for_legend,
        labels_for_legend,
        loc="lower center",
        ncol=6,
        frameon=True,
        fancybox=True,
        shadow=True,
        edgecolor="black",
        bbox_to_anchor=(0.5, 0.02)
    )

    plt.subplots_adjust(bottom=0.2, top=0.88, left=0.08, right=0.96, wspace=0.15)

    out_path_png = fig_dir / "dataset_affective_evolution.png"
    plt.savefig(out_path_png, dpi=600, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()

