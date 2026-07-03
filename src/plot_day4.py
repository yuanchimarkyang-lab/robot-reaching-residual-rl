from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def summarize_policy_csv(path, policy_name):
    df = pd.read_csv(path)

    summary = {
        "policy": policy_name,
        "success_rate": df["success"].mean(),
        "mean_final_distance": df["final_distance"].mean(),
        "median_final_distance": df["final_distance"].median(),
        "mean_return": df["total_return"].mean(),
        "mean_action_norm": df["mean_action_norm"].mean(),
    }

    if "action_smoothness" in df.columns:
        summary["mean_action_smoothness"] = df["action_smoothness"].mean()
    else:
        summary["mean_action_smoothness"] = None

    return summary


def summarize_best_baseline(path):
    df = pd.read_csv(path)

    summary = df.groupby("kp").agg(
        success_rate=("success", "mean"),
        mean_final_distance=("final_distance", "mean"),
        median_final_distance=("final_distance", "median"),
        mean_return=("total_return", "mean"),
        mean_action_norm=("mean_action_norm", "mean"),
        mean_action_smoothness=("action_smoothness", "mean"),
    ).reset_index()

    best = summary.sort_values(
        ["success_rate", "mean_final_distance"],
        ascending=[False, True],
    ).iloc[0]

    return {
        "policy": f"proportional_kp_{best['kp']}",
        "success_rate": best["success_rate"],
        "mean_final_distance": best["mean_final_distance"],
        "median_final_distance": best["median_final_distance"],
        "mean_return": best["mean_return"],
        "mean_action_norm": best["mean_action_norm"],
        "mean_action_smoothness": best["mean_action_smoothness"],
    }


def main():
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {
        "random": Path("results/metrics/random_policy_metrics.csv"),
        "baseline": Path("results/metrics/proportional_controller_metrics.csv"),
        "sac": Path("results/metrics/sac_policy_metrics.csv"),
        "residual_sac": Path("results/metrics/residual_sac_policy_metrics.csv"),
    }

    summaries = []

    if paths["random"].exists():
        summaries.append(summarize_policy_csv(paths["random"], "random"))

    if paths["baseline"].exists():
        summaries.append(summarize_best_baseline(paths["baseline"]))

    if paths["sac"].exists():
        summaries.append(summarize_policy_csv(paths["sac"], "sac"))

    if paths["residual_sac"].exists():
        summaries.append(
            summarize_policy_csv(paths["residual_sac"], "residual_sac")
        )

    comparison = pd.DataFrame(summaries)

    comparison_path = Path("results/metrics/day4_policy_comparison.csv")
    comparison.to_csv(comparison_path, index=False)

    print("\nDay 4 policy comparison:")
    print(comparison)

    # Success rate comparison
    plt.figure()
    plt.bar(comparison["policy"], comparison["success_rate"])
    plt.ylabel("Success rate")
    plt.title("Policy Comparison: Success Rate")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_dir / "day4_success_rate_comparison.png", dpi=200)

    # Final distance comparison
    plt.figure()
    plt.bar(comparison["policy"], comparison["mean_final_distance"])
    plt.ylabel("Mean final distance")
    plt.title("Policy Comparison: Final Distance")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_dir / "day4_final_distance_comparison.png", dpi=200)

    # Return comparison
    plt.figure()
    plt.bar(comparison["policy"], comparison["mean_return"])
    plt.ylabel("Mean episode return")
    plt.title("Policy Comparison: Return")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(output_dir / "day4_return_comparison.png", dpi=200)

    # Action smoothness comparison
    if comparison["mean_action_smoothness"].notna().all():
        plt.figure()
        plt.bar(comparison["policy"], comparison["mean_action_smoothness"])
        plt.ylabel("Mean action change")
        plt.title("Policy Comparison: Action Smoothness")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(output_dir / "day4_action_smoothness_comparison.png", dpi=200)

    print(f"\nSaved comparison CSV to: {comparison_path}")
    print("Saved Day 4 plots to results/plots/")


if __name__ == "__main__":
    main()

