import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    df = pd.read_csv("results/metrics/random_policy_metrics.csv")
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.hist(df["final_distance"], bins=20)
    plt.xlabel("Fianl distance to target")
    plt.ylabel("Number of episodes")
    plt.title("Random Policy: Final Distance Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "random_policy_final_distance_hist.png", dpi=200)


    plt.figure()
    plt.hist(df["total_return"], bins=20)
    plt.xlabel("Episode return")
    plt.ylabel("Number of episodes")
    plt.title("Random Policy: Return Distribution")
    plt.tight_layout()
    plt.savefig(output_dir / "random_policy_return_hist.png", dpi=200)

    print("Saved Day 1 plots to results/plots/")

if __name__=="__main__":
    main()
