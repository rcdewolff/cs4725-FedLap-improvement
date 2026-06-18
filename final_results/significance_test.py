"""
Tests whether each metric shows a significant TREND across the full noise
sweep (at a fixed train_epochs) -- using all noise levels and all seeds,
not just the noise=0 vs noise=0.05 endpoints.

Two complementary tests are run per (dataset, metric):

1. Linear regression slope test (scipy.stats.linregress)
   Tests for a LINEAR relationship between noise_multiplier and the metric.
   H0: slope == 0. Good general-purpose trend test, but assumes the
   relationship is roughly a straight line -- a few non-monotonic "wiggle"
   points can pull the p-value up even if there's a real overall trend.

2. Spearman rank correlation (scipy.stats.spearmanr)
   Tests for a MONOTONIC relationship (doesn't assume linearity, only that
   higher noise tends to rank higher/lower on the metric). More robust to
   the kind of local non-monotonic noise you noticed in some plots, since
   it only depends on rank order, not exact values.

Why not Mann-Kendall? It's the more "textbook" monotonic-trend test for
ordered data, but for this design (a handful of fixed noise levels, several
seeds at each) it reduces to something very close to Spearman's correlation
anyway -- so Spearman is kept here for simplicity, as requested.

Both tests use every (noise_multiplier, seed) data point for a given
dataset/metric/epochs combination (e.g. 5 noise levels x 3 seeds = 15
points), so the result reflects the trend across the whole sweep rather
than a comparison of just two settings.

Expects RAW per-run data (not the aggregated summary table), with columns:
dataset, seed, train_epochs, noise_multiplier, and the metric columns.
"""

import pandas as pd
from scipy import stats


def trend_significance(
    df,
    datasets,
    metrics=("test_acc", "mse", "cosine_sim", "feat_corr"),
    epochs=100,
    alpha=0.05,
):
    rows = []
    for dataset in datasets:
        sub = df[(df["dataset"] == dataset) & (df["train_epochs"] == epochs)]
        x = sub["noise_multiplier"].values

        for metric in metrics:
            y = sub[metric].values

            lin = stats.linregress(x, y)
            rho, p_rho = stats.spearmanr(x, y)

            rows.append({
                "dataset": dataset,
                "metric": metric,
                "n_points": len(x),
                "slope": lin.slope,
                "pearson_r": lin.rvalue,
                "linreg_p_value": lin.pvalue,
                "linreg_significant": lin.pvalue < alpha,
                "spearman_rho": rho,
                "spearman_p_value": p_rho,
                "spearman_significant": p_rho < alpha,
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    # --- usage example ---
    # Load each dataset's raw (per-seed) csv and tag with a dataset name,
    # same pattern as the earlier scripts.
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
        d = pd.read_csv(f"{name}.csv")
        d["dataset"] = name
        dfs.append(d)
    data = pd.concat(dfs, ignore_index=True)

    result = trend_significance(data, datasets=dataset_names)
    result.to_csv("significance_results.csv", index=False)
    print(result)