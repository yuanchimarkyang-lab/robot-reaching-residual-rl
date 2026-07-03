from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def plot_metric(summary, experiment, metric, ylabel, output_dir):
    exp_df = summary[summary["experiment"] == experiment].copy()

    plt.figure()

    for policy in exp_df["policy"].unique():
        sub = exp_df[exp_df["policy"] == policy].sort_values("level")
        plt.plot(sub["level"], sub[metric], marker="o", label=policy)

    plt.xlabel(experiment.replace("_", " "))
    plt.ylabel(ylabel)
    plt.title(f"{ylabel} under {experiment.replace('_', ' ')}")
    plt.legend()
    plt.tight_layout()

    output_path = output_dir / f"day5_{experiment}_{metric}.png"
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    summary_path = Path("results/metrics/robustness_summary.csv")

    if not summary_path.exists():
        raise FileNotFoundError(
            "Missing results/metrics/robustness_summary.csv. "
            "Run src/evaluate_robustness.py first."
        )

    summary = pd.read_csv(summary_path)

    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = [
        ("success_rate", "Success rate"),
        ("mean_final_distance", "Mean final distance"),
        ("mean_return", "Mean episode return"),
        ("mean_action_smoothness", "Mean action change"),
    ]

    for experiment in summary["experiment"].unique():
        for metric, ylabel in metrics:
            plot_metric(
                summary=summary,
                experiment=experiment,
                metric=metric,
                ylabel=ylabel,
                output_dir=output_dir,
            )

    print("Saved Day 5 robustness plots to results/plots/")


if __name__ == "__main__":
    main()

