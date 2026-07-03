import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def main():
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline_df = pd.read_csv("results/metrics/proportional_controller_metrics.csv")

    # Plot 1: success rate vs kp
    summary = baseline_df.groupby("kp").agg(
        success_rate=("success", "mean"),
        mean_final_distance=("final_distance", "mean"),
        median_final_distance=("final_distance", "median"),
        mean_return=("total_return", "mean"),
        mean_action_smoothness=("action_smoothness", "mean"),
    ).reset_index()

    plt.figure()
    plt.plot(summary["kp"], summary["success_rate"], marker="o")
    plt.xlabel("Proportional gain Kp")
    plt.ylabel("Success rate")
    plt.title("Proportional Controller: Success Rate vs Kp")
    plt.tight_layout()
    plt.savefig(output_dir / "proportional_success_rate_vs_kp.png", dpi=200)

    # Plot 2: final distance vs kp
    plt.figure()
    plt.plot(summary["kp"], summary["mean_final_distance"], marker="o", label="Mean")
    plt.plot(summary["kp"], summary["median_final_distance"], marker="o", label="Median")
    plt.xlabel("Proportional gain Kp")
    plt.ylabel("Final distance to target")
    plt.title("Proportional Controller: Final Distance vs Kp")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "proportional_final_distance_vs_kp.png", dpi=200)

    # Plot 3: action smoothness vs kp
    plt.figure()
    plt.plot(summary["kp"], summary["mean_action_smoothness"], marker="o")
    plt.xlabel("Proportional gain Kp")
    plt.ylabel("Mean action change")
    plt.title("Proportional Controller: Action Smoothness vs Kp")
    plt.tight_layout()
    plt.savefig(output_dir / "proportional_action_smoothness_vs_kp.png", dpi=200)

    # Optional: compare against random policy if Day 1 file exists
    random_path = Path("results/metrics/random_policy_metrics.csv")

    if random_path.exists():
        random_df = pd.read_csv(random_path)

        random_summary = {
            "policy": "random",
            "success_rate": random_df["success"].mean(),
            "mean_final_distance": random_df["final_distance"].mean(),
            "mean_return": random_df["total_return"].mean(),
        }

        best_kp_row = summary.sort_values(
            ["success_rate", "mean_final_distance"],
            ascending=[False, True],
        ).iloc[0]

        comparison = pd.DataFrame([
            random_summary,
            {
                "policy": f"proportional_kp_{best_kp_row['kp']}",
                "success_rate": best_kp_row["success_rate"],
                "mean_final_distance": best_kp_row["mean_final_distance"],
                "mean_return": best_kp_row["mean_return"],
            },
        ])

        comparison.to_csv(
            "results/metrics/day2_random_vs_baseline_summary.csv",
            index=False,
        )

        plt.figure()
        plt.bar(comparison["policy"], comparison["success_rate"])
        plt.ylabel("Success rate")
        plt.title("Random Policy vs Best Proportional Controller")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(output_dir / "day2_random_vs_baseline_success_rate.png", dpi=200)

        plt.figure()
        plt.bar(comparison["policy"], comparison["mean_final_distance"])
        plt.ylabel("Mean final distance")
        plt.title("Random Policy vs Best Proportional Controller")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(output_dir / "day2_random_vs_baseline_final_distance.png", dpi=200)

        print("\nRandom vs baseline comparison:")
        print(comparison)

    print("\nBaseline summary:")
    print(summary)

    print("\nSaved Day 2 plots to results/plots/")


if __name__ == "__main__":
    main()

