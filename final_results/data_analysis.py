import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
summary_100_raw = summary[summary["train_epochs"] == 100].copy()


def display_name(dataset_name):
    if dataset_name == "chameleon":
        return "Chameleon"
    if dataset_name == "Photo":
        return "Amazon Photo"
    return dataset_name

for metric, ylabel in labels.items():
    plt.figure()
    for dataset_name, group in agg.groupby("dataset"):
        group = group.sort_values("noise_multiplier")
        name = display_name(dataset_name)
        # UNCOMMENT TO ALSO PLOT STANDARD DEVIATIONS
        # stds = summary[summary["dataset"] == dataset_name]
        # stds = stds[stds["train_epochs"] == 100]
        # stds = stds[f"{metric}_std"]
        x = group["noise_multiplier"].values
        y = group[metric].values
        line, = plt.plot(x, y, marker="o", label=name)
        # plt.fill_between(x, y - stds, y + stds, color=line.get_color(), alpha=0.2)
    plt.xlabel("Noise multiplier")
    plt.ylabel(ylabel)
    if metric == "mse":
        plt.yscale('logit')
    plt.title(f"{ylabel} vs noise multiplier (epochs=100)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{metric}_vs_noise_epochs100_lines.png", dpi=150)
    plt.close()

metric_colors = {
    "mse": "RdYlGn",
    "cosine_sim": "RdYlGn_r",
    "feat_corr": "RdYlGn_r",
    "test_acc": "RdYlGn"
}

for metric, ylabel in labels.items():
    heatmap_means = summary_100_raw.pivot(index="dataset", columns="noise_multiplier", values=f"{metric}_mean")
    heatmap_stds = summary_100_raw.pivot(index="dataset", columns="noise_multiplier", values=f"{metric}_std")

    heatmap_means = heatmap_means.reindex(dataset_names)
    heatmap_stds = heatmap_stds.reindex(dataset_names)
    heatmap_means = heatmap_means.reindex(sorted(heatmap_means.columns), axis=1)
    heatmap_stds = heatmap_stds.reindex(heatmap_means.columns, axis=1)

    mean_values = heatmap_means.to_numpy() * 100
    std_values = heatmap_stds.to_numpy() * 100
    display_values = np.array([
        [f"{mean_values[i, j]:.2f} \n+-\n {std_values[i, j]:.2f}" for j in range(mean_values.shape[1])]
        for i in range(mean_values.shape[0])
    ])

    plt.figure(figsize=(1.4 * mean_values.shape[1] + 3, 0.9 * mean_values.shape[0] + 2.5))
    image = plt.imshow(mean_values, aspect="auto", cmap=metric_colors[metric])
    plt.colorbar(image, label=ylabel)

    plt.xticks(np.arange(mean_values.shape[1]), [str(x) for x in heatmap_means.columns], rotation=45, ha="right")
    plt.yticks(np.arange(mean_values.shape[0]), [display_name(name) for name in heatmap_means.index])
    plt.xlabel("Noise multiplier")
    plt.ylabel("Dataset")
    plt.title(f"{ylabel} heatmap (epochs=100, values are x100)")

    meanvalue = np.nanmean(mean_values)
    text_color_threshold_low = np.nanquantile(mean_values, q=.05) if metric == "cosine_sim" else np.nanquantile(mean_values, q=.2)
    text_color_threshold_high = np.nanquantile(mean_values, q=.8)
    for i in range(mean_values.shape[0]):
        for j in range(mean_values.shape[1]):
            value = mean_values[i, j]
            if np.isnan(value):
                continue
            text_color = "black" if text_color_threshold_low <= value <= text_color_threshold_high else "white"
            plt.text(j, i, display_values[i, j], ha="center", va="center", color=text_color, fontsize=12)

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