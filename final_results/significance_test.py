"""
For each (dataset, metric), tests whether the difference between
noise_multiplier=0.0 and noise_multiplier=0.05 (at train_epochs=100) is
statistically significant.

Uses a PAIRED t-test (scipy.stats.ttest_rel), pairing the two noise
conditions by seed -- since seed 1/2/3 represents the same run setup
in both conditions, a paired test is more appropriate (and more
powerful) than an unpaired/independent t-test.

IMPORTANT CAVEAT: with only 3 seeds, this comparison has very low
statistical power (2 degrees of freedom). A non-significant result
here does NOT mean there's no real effect -- it may just mean 3 runs
aren't enough to detect it. Take p-values as a rough guide, not proof.
(A non-parametric alternative, Wilcoxon signed-rank, was considered
but with n=3 it cannot reach p<0.05 at all -- there are only 8 possible
sign patterns, so the smallest achievable two-sided p-value is 0.25.
A parametric t-test is the only option that's even capable of
detecting significance with this few runs.)

Expects RAW per-run data (not the aggregated summary table), with
columns: dataset, seed, train_epochs, noise_multiplier, and the metric
columns (test_acc, mse, cosine_sim, feat_corr, ...).
"""

import pandas as pd
from scipy import stats


def paired_significance(
    df,
    datasets,
    metrics=("test_acc", "mse", "cosine_sim", "feat_corr"),
    epochs=100,
    noise_a=0.0,
    noise_b=0.05,
    alpha=0.05,
):
    rows = []
    for dataset in datasets:
        sub = df[(df["dataset"] == dataset) & (df["train_epochs"] == epochs)]
        a = sub[sub["noise_multiplier"] == noise_a][["seed"] + list(metrics)]
        b = sub[sub["noise_multiplier"] == noise_b][["seed"] + list(metrics)]
        merged = pd.merge(a, b, on="seed", suffixes=("_a", "_b"))

        if len(merged) == 0:
            print(f"Warning: no matching seeds for {dataset} at epochs={epochs}")
            continue

        for metric in metrics:
            vals_a = merged[f"{metric}_a"]
            vals_b = merged[f"{metric}_b"]
            t_stat, p_val = stats.ttest_rel(vals_a, vals_b)
            rows.append({
                "dataset": dataset,
                "metric": metric,
                "n_pairs": len(merged),
                f"mean_noise{noise_a}": vals_a.mean(),
                f"mean_noise{noise_b}": vals_b.mean(),
                "t_stat": t_stat,
                "p_value": p_val,
                "significant_p<0.05": p_val < alpha,
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

    result = paired_significance(data, datasets=dataset_names)
    result.to_csv("significance_results.csv", index=False)
    print(result)