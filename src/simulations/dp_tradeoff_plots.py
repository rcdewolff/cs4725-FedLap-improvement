import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_and_aggregate(csv_path):
    df = pd.read_csv(csv_path)

    df = df[df["dp_enabled"] == True].copy()

    group_cols = ["clip_norm", "noise_multiplier"]
    agg = (
        df.groupby(group_cols)
        .agg(
            test_acc_mean=("test_acc", "mean"),
            test_acc_std=("test_acc", "std"),
            attack_auc_mean=("attack_auc", "mean"),
            attack_ap_mean=("attack_ap", "mean"),
            attack_f1_mean=("attack_f1", "mean"),
        )
        .reset_index()
    )
    agg["privacy_strength"] = 1.0 - agg["attack_auc_mean"]
    return agg


def plot_tradeoff_scatter(agg, out_dir):
    plt.figure(figsize=(8, 6))
    sizes = 200 * (agg["clip_norm"] / agg["clip_norm"].max())
    sc = plt.scatter(
        agg["test_acc_mean"],
        agg["privacy_strength"],
        c=agg["noise_multiplier"],
        s=sizes,
        cmap="viridis",
        edgecolors="k",
        alpha=0.9,
    )
    plt.xlabel("Test accuracy (mean)")
    plt.ylabel("Privacy strength (1 - attack AUC)")
    plt.title("Accuracy vs Privacy Tradeoff (DP)")
    cbar = plt.colorbar(sc)
    cbar.set_label("Noise multiplier")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tradeoff_scatter.png"), dpi=200)


def plot_tradeoff_scatter_all(df, out_dir):
    df = df.copy()
    df["privacy_strength"] = 1.0 - df["attack_auc"]
    df = df.dropna(subset=["test_acc", "privacy_strength"]).reset_index(drop=True)
    if len(df) == 0:
        return

    plt.figure(figsize=(8, 6))
    max_clip = df["clip_norm"].max() if df["clip_norm"].max() > 0 else 1.0
    sizes = 80 + 160 * (df["clip_norm"] / max_clip)
    sc = plt.scatter(
        df["test_acc"],
        df["privacy_strength"],
        c=df["noise_multiplier"],
        #s=sizes,
        cmap="viridis",
        edgecolors="k",
        alpha=0.75,
    )
    plt.xlabel("Test accuracy")
    plt.ylabel("Privacy strength (1 - attack AUC)")
    plt.title("Accuracy vs Privacy (All Runs)")
    cbar = plt.colorbar(sc)
    cbar.set_label("Noise multiplier")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "tradeoff_scatter_all.png"), dpi=200)


def plot_heatmap(agg, value_col, out_path, title):
    pivot = agg.pivot(index="clip_norm", columns="noise_multiplier", values=value_col)
    pivot = pivot.sort_index().sort_index(axis=1)

    plt.figure(figsize=(8, 6))
    plt.imshow(pivot.values, aspect="auto", origin="lower", cmap="viridis")
    plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=45)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("Noise multiplier")
    plt.ylabel("Clip norm")
    plt.title(title)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to dp_grid CSV")
    parser.add_argument("--out", default=None, help="Output directory for plots")
    args = parser.parse_args()

    out_dir = args.out or os.path.dirname(args.csv)
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(args.csv)
    agg = load_and_aggregate(args.csv)
    plot_tradeoff_scatter(agg, out_dir)
    plot_tradeoff_scatter_all(df, out_dir)
    plot_heatmap(
        agg,
        "test_acc_mean",
        os.path.join(out_dir, "heatmap_test_acc.png"),
        "Test Accuracy (mean)",
    )
    plot_heatmap(
        agg,
        "attack_auc_mean",
        os.path.join(out_dir, "heatmap_attack_auc.png"),
        "Attack AUC (mean)",
    )


if __name__ == "__main__":
    main()
