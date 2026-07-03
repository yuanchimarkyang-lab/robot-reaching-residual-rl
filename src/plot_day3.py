from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_policy_summary(path, policy_name):
    df = pd.read_csv(path)

    return {
        "policy": policy_name,
        "success_rate": df["success"].mean(),
        "mean_final_distance": df["final_distance"].mean(),
        "median_final_distance": df["final_distance"].median(),
        "mean_return": df["total_return"].mean(),
        "mean_action_norm": df["mean_action_norm"].mean(),
        "mean_action_smoothness": df["action_smoothness"].mean()
        if "action_smoothness" in df.columns
        else None,
    }


def main():
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = []

    random_path = Path("results/metrics/random_policy_metrics.csv")
    baseline_path = Path("results/metrics/proportional_controller_metrics.csv")
    sac_path = Path("results/metrics/sac_policy_metrics.csv")

    if random_path.exists():
        summaries.append(load_policy_summary(random_path, "random"))

    if baseline_path.exists():
        baseline_df = pd.read_csv(baseline_path)

        baseline_summary = baseline_df.groupby("kp").agg(
            success_rate=("success", "mean"),
            mean_final_distance=("final_distance", "mean"),
            median_final_distance=("final_distance", "median"),
            mean_return=("total_return", "mean"),
            mean_action_norm=("mean_action_norm", "mean"),
            mean_action_smoothness=("action_smoothness", "mean"),
        ).reset_index()

        best_baseline = baseline_summary.sort_values(
            ["success_rate", "mean_final_distance"],
            ascending=[False, True],
        ).iloc[0]

        summaries.append(
            {
                "policy": f"proportional_kp_{best_baseline['kp']}",
                "success_rate": best_baseline["success_rate"],
                "mean_final_distance": best_baseline["mean_final_distance"],
                "median_final_distance": best_baseline["median_final_distance"],
                "mean_return": best_baseline["mean_return"],
                "mean_action_norm": best_baseline["mean_action_norm"],
                "mean_action_smoothness": best_baseline["mean_action_smoothness"],
            }
        )

    if sac_path.exists():
        summaries.append(load_policy_summary(sac_path, "sac"))

    comparison = pd.DataFrame(summaries)
    comparison_path = Path("results/metrics/day3_policy_comparison.csv")
    comparison.to_csv(comparison_path, index=False)

    print("\nDay 3 policy comparison:")
    print(comparison)

    # Plot 1: success rate
    plt.figure()
    plt.bar(comparison["policy"], comparison["success_rate"])
    plt.ylabel("Success rate")
    plt.title("Policy Comparison: Success Rate")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_dir / "day3_success_rate_comparison.png", dpi=200)

    # Plot 2: final distance
    plt.figure()
    plt.bar(comparison["policy"], comparison["mean_final_distance"])
    plt.ylabel("Mean final distance")
    plt.title("Policy Comparison: Final Distance")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_dir / "day3_final_distance_comparison.png", dpi=200)

    # Plot 3: return
    plt.figure()
    plt.bar(comparison["policy"], comparison["mean_return"])
    plt.ylabel("Mean episode return")
    plt.title("Policy Comparison: Return")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_dir / "day3_return_comparison.png", dpi=200)

    # Plot 4: action smoothness, if available
    if comparison["mean_action_smoothness"].notna().all():
        plt.figure()
        plt.bar(comparison["policy"], comparison["mean_action_smoothness"])
        plt.ylabel("Mean action change")
        plt.title("Policy Comparison: Action Smoothness")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(output_dir / "day3_action_smoothness_comparison.png", dpi=200)

    print(f"\nSaved comparison CSV to: {comparison_path}")
    print("Saved Day 3 plots to results/plots/")


if __name__ == "__main__":
    main()

