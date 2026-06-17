"""
Turns the summary CSV (one row per dataset/train_epochs/noise_multiplier,
with <metric>_mean and <metric>_std columns) into LaTeX table code.

Expected input columns (this is what summary_table.csv from the earlier
analysis script looks like):
    dataset, train_epochs, noise_multiplier,
    test_acc_mean, test_acc_std,
    mse_mean, mse_std,
    cosine_sim_mean, cosine_sim_std,
    feat_corr_mean, feat_corr_std

Layout produced (matches the sketch you gave):

                  |  dataset 1   |  dataset 2   |  dataset 3
    epochs=10     |  noise=0.0   |  metrics...  |  metrics...  | metrics...
                  |  noise=0.005 |  metrics...  |  metrics...  | metrics...
                  |  ...
    epochs=30     |  ...
    epochs=100    |  ...

With 6 datasets total and max_cols_per_table=3, you get two separate
`table` environments (datasets 1-3, then 4-6) that you place side by
side / one after another in the paper.

Required LaTeX packages: booktabs, multirow, makecell
(metrics are stacked vertically inside each cell -- comma-separating
4 metrics on one line overflows the page width even in landscape with
only 3 dataset columns, confirmed by test-compiling it).
"""

import copy

import pandas as pd


def _escape_latex(s):
    repl = {
        '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
        '_': r'\_', '{': r'\{', '}': r'\}',
        '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
        '\\': r'\textbackslash{}',
    }
    return ''.join(repl.get(c, c) for c in str(s))


def _fmt(mean, std, decimals=2, scale=100):
    return f"${mean * scale:.{decimals}f} \\pm {std * scale:.{decimals}f}$"


def build_latex_tables(
    df,
    datasets,
    metrics=("test_acc", "mse", "cosine_sim", "feat_corr"),
    metric_labels=None,
    decimals=2,
    scale=100,
    noise_decimals=3,
    max_cols_per_table=1,
    caption="Privacy-utility trade-off results",
    label_prefix="tab:results",
):
    """
    df       : DataFrame with the columns described in the module docstring.
    datasets : ordered list of dataset names. Defines column order, and is
               chopped into chunks of `max_cols_per_table` -> one LaTeX
               `table` per chunk.
    metrics  : which <metric>_mean/<metric>_std column pairs to include,
               in the order they should appear within each cell.
    decimals : decimal places shown for each value (after scaling).
    scale    : every value (mean and std) is multiplied by this before
               formatting -- default 100, so e.g. 0.2238 -> 22.38.
    Returns one string containing all the generated `table` environments.
    """
    if metric_labels is None:
        metric_labels = {
            "test_acc": "Acc",
            "mse": "MSE",
            "cosine_sim": "CosSim",
            "feat_corr": "FeatCorr",
        }

    epochs_sorted = sorted(df["train_epochs"].unique())
    chunks = [datasets[i:i + max_cols_per_table]
              for i in range(0, len(datasets), max_cols_per_table)]

    cell_structure = ", ".join(metric_labels[m] for m in metrics)

    all_tables = []
    for t_idx, chunk in enumerate(chunks):
        n_cols = len(chunk)
        col_spec = "ll" + "c" * n_cols
        lines = [
            r"\begin{table}[ht]",
            r"\centering",
            r"\small",
            rf"\begin{{tabular}}{{{col_spec}}}",
            r"\toprule",
        ]
        # explanatory row describing what's inside each cell, e.g.
        # "Each cell: Acc, MSE, CosSim, FeatCorr (x100)"
        title = copy.deepcopy(chunk[0])
        if title == "chameleon":
            title = "Chameleon"
        if title == "Photo":
            title == "Amazon Photo"
        lines.append(
            rf"\multicolumn{{{2 + n_cols}}}{{c}}{{\textbf{{{_escape_latex(title)}}}}} \\"
        )
        lines.append(
            rf"\multicolumn{{{2 + n_cols}}}{{c}}{{Each cell: {cell_structure} "
            rf"(values $\times {scale}$)}} \\"
        )
        # lines.append(r"\midrule")
        # header = " & ".join(["", ""] + [rf"\textbf{{{_escape_latex(d)}}}" for d in chunk])
        # lines.append(header + r" \\")
        lines.append(r"\midrule")

        for epochs in epochs_sorted:
            sub_epoch = df[df["train_epochs"] == epochs]
            noise_vals = sorted(sub_epoch["noise_multiplier"].unique())
            n_noise = len(noise_vals)
            for i, noise in enumerate(noise_vals):
                row = [rf"\multirow{{{n_noise}}}{{*}}{{e={epochs}}}" if i == 0 else ""]
                row.append(f"$\\sigma={noise:.{noise_decimals}f}$")

                for dataset in chunk:
                    match = df[(df["dataset"] == dataset) &
                               (df["train_epochs"] == epochs) &
                               (df["noise_multiplier"] == noise)]
                    if match.empty:
                        row.append("--")
                        continue
                    r = match.iloc[0]
                    metric_lines = [
                        f"{_fmt(r[f'{m}_mean'], r[f'{m}_std'], decimals, scale)}"# f"{metric_labels[m]}: {_fmt(r[f'{m}_mean'], r[f'{m}_std'], decimals, scale)}"
                        for m in metrics
                    ]
                    row.append(r"\makecell[l]{%s}" % r" , ".join(metric_lines))

                lines.append(" & ".join(row) + r" \\")
            lines.append(r"\midrule")

        if lines[-1] == r"\midrule":
            lines.pop()
        lines.append(r"\bottomrule")
        lines.append(r"\end{tabular}")
        start = t_idx * max_cols_per_table + 1
        end = start + len(chunk) - 1
        lines.append(rf"\caption{{{caption} (datasets {start}--{end})}}")
        lines.append(rf"\label{{{label_prefix}_{t_idx + 1}}}")
        lines.append(r"\end{table}")
        all_tables.append("\n".join(lines))

    return "\n\n".join(all_tables)


if __name__ == "__main__":
    # --- usage example ---
    df = pd.read_csv("summary_table.csv")

    dataset_order = [
    "chameleon",
    "CiteSeer",
    "Cora",
    "PubMed",
    "Photo",
    "Amazon-ratings",
]

    latex = build_latex_tables(df, datasets=dataset_order)

    with open("results_table.tex", "w") as f:
        f.write(latex)

    print(latex)