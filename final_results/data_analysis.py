import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------
# 0. Point this at your CSVs (one file per dataset)
# ---------------------------------------------------------------
dataset_names = [
    "chameleon",
    "CiteSeer",
    "Cora",
    "PubMed",
    "Photo",
    "Amazon-ratings",
]

dfs = []
for name in dataset_names:
    try:
      print(f"{name}.csv")
      df = pd.read_csv(f"{name}.csv")
    except FileNotFoundError as e:
      print(e)
      continue
    df["dataset"] = name
    dfs.append(df)
data = pd.concat(dfs, ignore_index=True)

metrics = ["test_acc", "mse", "cosine_sim", "feat_corr"]
labels = {
    "test_acc": "Test accuracy",
    "mse": "MSE",
    "cosine_sim": "Cosine similarity",
    "feat_corr": "Feature correlation",
}

grouped = data.groupby(["dataset", "train_epochs", "noise_multiplier"])[metrics]
summary = grouped.agg(["mean", "std"])
summary.columns = [f"{col}_{stat}" for col, stat in summary.columns]
summary = summary.reset_index()
summary.to_csv("summary_table.csv", index=False)  # raw mean/std columns, easy to plot from
print(summary)
 
# A "paper-style" version with formatted "mean +- std" strings, one column per metric
summary_fmt = summary[["dataset", "train_epochs", "noise_multiplier"]].copy()
for metric in metrics:
    summary_fmt[metric] = (
        summary[f"{metric}_mean"].map(lambda x: f"{x:.4f}")
        + " +- "
        + summary[f"{metric}_std"].map(lambda x: f"{x:.4f}")
    )
summary_fmt.to_csv("summary_table_formatted.csv", index=False)
print(summary_fmt)
 
# If it's too large, just look at epochs == 100:
summary_100 = summary_fmt[summary_fmt["train_epochs"] == 100]
summary_100.to_csv("summary_table_100epochs.csv", index=False)

# Plots

plot_data = data[data["train_epochs"] == 100]
agg = plot_data.groupby(["dataset", "noise_multiplier"])[metrics].mean().reset_index()

for metric, ylabel in labels.items():
    plt.figure()
    for dataset_name, group in agg.groupby("dataset"):
        group = group.sort_values("noise_multiplier")
        plt.plot(group["noise_multiplier"], group[metric], marker="o", label=dataset_name)
    plt.xlabel("Noise multiplier")
    plt.ylabel(ylabel)
    if metric == "mse":
        plt.yscale('logit')
    plt.title(f"{ylabel} vs noise multiplier (epochs=100)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{metric}_vs_noise_epochs100.png", dpi=150)
    plt.close()

# ---------------------------------------------------------------
# 3. Effect of train_epochs on the privacy/utility tradeoff
#    One subplot per dataset, one line per epoch count, test_acc vs noise
# ---------------------------------------------------------------
agg_all = data.groupby(["dataset", "train_epochs", "noise_multiplier"])[metrics].mean().reset_index()
datasets = agg_all["dataset"].unique()

fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 4), sharey=True)
if len(datasets) == 1:
    axes = [axes]

for ax, dataset_name in zip(axes, datasets):
    sub = agg_all[agg_all["dataset"] == dataset_name]
    for epochs, group in sub.groupby("train_epochs"):
        group = group.sort_values("noise_multiplier")
        ax.plot(group["noise_multiplier"], group["test_acc"], marker="o", label=f"{epochs} epochs")
    ax.set_title(dataset_name)
    ax.set_xlabel("Noise multiplier")

axes[0].set_ylabel("Test accuracy")
axes[0].legend()
plt.tight_layout()
plt.savefig("epochs_effect_on_tradeoff.png", dpi=150)
plt.close()

print("Done: summary_table.csv, summary_table_100epochs.csv, "
      "4 metric plots, and epochs_effect_on_tradeoff.png")