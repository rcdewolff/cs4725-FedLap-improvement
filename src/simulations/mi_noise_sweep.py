"""
Membership Inference Attack vs Differential Privacy — FedLap Online Phase.

Uses the SFV gradient-norm attack: training nodes receive direct gradient
signal so ||grad_SFV||_2 is larger for them than for non-training nodes.
An adversarial server uses these norms as a membership score.

DP adds equal Gaussian noise to every node's gradient, collapsing the norm
gap and pushing AUC toward 0.5 (random guessing).

Efficiency: hop2vec is computed ONCE per seed (expensive). The initialized
server is deepcopied for each noise level, exactly like online_phase_dlg_experiment.py.
"""

import os
import sys
import random
from copy import deepcopy

pythonpath = os.getcwd()
if pythonpath not in sys.path:
    sys.path.append(pythonpath)

simulations_path = os.path.join(pythonpath, "src", "simulations")
if simulations_path not in sys.path:
    sys.path.append(simulations_path)

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import *
from src.server import Server
from src.GNN.GNN_server import GNNServer
from src.utils.define_graph import define_graph
from src.utils.graph_partitioning import partition_graph
import src.utils.utils as utils_module
import privacy_attack_experiment as pae

# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

SEEDS = [1, 2, 3]
TRAIN_EPOCHS = 30
NOISE_MULTIPLIERS = [0.0, 0.005, 0.01, 0.02, 0.05]
CLIP_NORM = 1.0

DATA_TYPE = "f+s"
SMODEL_TYPE = "Laplace"
FMODEL_TYPE = "GNN"
STRUCTURE_TYPE = "hop2vec"

# ---------------------------------------------------------------------------
# Helpers (same pattern as online_phase_dlg_experiment.py)
# ---------------------------------------------------------------------------

def set_seed(value):
    utils_module.seed = value
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def prepare_graph_and_subgraphs():
    graph = define_graph(config.dataset.dataset_name)
    graph.add_masks(
        train_ratio=config.subgraph.train_ratio,
        test_ratio=config.subgraph.test_ratio,
    )
    subgraphs = partition_graph(
        graph,
        config.subgraph.num_subgraphs,
        config.subgraph.partitioning,
    )
    return graph, subgraphs


def build_server(graph, subgraphs):
    server = GNNServer(graph)
    for subgraph in subgraphs:
        server.add_client(subgraph)
    return server


def initialize_server(graph, subgraphs):
    """Build + FL init (hop2vec). Called ONCE per seed."""
    server = build_server(graph, subgraphs)
    server.initialize_FL(
        smodel_type=SMODEL_TYPE,
        fmodel_type=FMODEL_TYPE,
        data_type=DATA_TYPE,
        structure_type=STRUCTURE_TYPE,
    )
    server.share_weights()
    return server


def train_server(server, noise_mult):
    """Train the already-initialized server with DP at every gradient upload."""
    config.dp.enabled = noise_mult > 0
    config.dp.noise_multiplier = noise_mult
    config.dp.clip_norm = CLIP_NORM
    config.dp.mode = "local"
    config.dp.separate_sfv = True

    return Server.joint_train_g(
        server,
        epochs=TRAIN_EPOCHS,
        FL=True,
        log=False,
        plot=False,
        model_type="FL f+s Laplace-GNN GA",
    )


def run_attack_round(server, noise_mult):
    """
    One extra gradient upload on the trained model.
    Captures raw SFV grads, applies the same DP noise the server would see,
    then scores each node by ||grad_SFV||_2.
    Returns the attack metric summary dict.
    """
    saved = config.dp.enabled
    config.dp.enabled = False

    server.reset_trainings()
    server.set_train_mode()
    server.train_clients(eval_=False)
    clients_grads = server.get_grads()

    config.dp.enabled = saved

    if noise_mult > 0:
        std = noise_mult * CLIP_NORM
        for cg in clients_grads:
            clip_grads_(cg, CLIP_NORM, separate_sfv=True)
            add_noise_(cg, std)

    _, summary = pae.compute_attack_metrics(clients_grads, server.clients)
    return summary

# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run():
    output_dir = os.path.join(save_path, "privacy_attack", "mi_dp")
    os.makedirs(output_dir, exist_ok=True)

    rows = []

    for seed_value in SEEDS:
        set_seed(seed_value)
        LOGGER.info(f"\n=== Seed {seed_value} ===")

        base_graph, base_subgraphs = prepare_graph_and_subgraphs()

        # hop2vec computed here — once per seed
        LOGGER.info("  Initializing FL (hop2vec — done once)...")
        init_server = initialize_server(deepcopy(base_graph), deepcopy(base_subgraphs))

        for noise_mult in NOISE_MULTIPLIERS:
            LOGGER.info(f"  sigma={noise_mult:.4f}")

            server = deepcopy(init_server)
            train_res = train_server(server, noise_mult)
            test_acc = float(train_res["Average"]["Test Acc"])

            summary = run_attack_round(server, noise_mult)
            if not summary:
                LOGGER.info("    No SFV gradients — skipping.")
                continue

            LOGGER.info(
                f"    test_acc={test_acc:.4f}  "
                f"AUC={summary.get('AUC', float('nan')):.4f}  "
                f"AP={summary.get('AP', float('nan')):.4f}  "
                f"F1={summary.get('F1', float('nan')):.4f}"
            )

            rows.append({
                "seed": seed_value,
                "noise_multiplier": noise_mult,
                "dp_enabled": noise_mult > 0,
                "test_acc": test_acc,
                "auc": summary.get("AUC", float("nan")),
                "ap": summary.get("AP", float("nan")),
                "f1": summary.get("F1", float("nan")),
                "acc": summary.get("Acc", float("nan")),
            })

    if not rows:
        LOGGER.info("No results collected.")
        return

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_dir, f"mi_dp_{now}_{config.dataset.dataset_name}.csv")
    df.to_csv(csv_path, index=False)
    LOGGER.info(f"\nSaved: {csv_path}")

    _plot(df, output_dir)
    return df


def _plot(df, output_dir):
    agg = df.groupby("noise_multiplier").mean(numeric_only=True).reset_index()

    panel_cfg = [
        ("auc",      "AUC",              "Membership Inference AUC\n(0.5 = random / perfect privacy)"),
        ("ap",       "Average Precision", "Average Precision (↓ = better privacy)"),
        ("f1",       "F1 Score",          "F1 at Optimal Threshold (↓ = better privacy)"),
        ("test_acc", "Test Accuracy",     "Model Utility (↑ = better)"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for ax, (col, ylabel, title) in zip(axes, panel_cfg):
        ax.plot(agg["noise_multiplier"], agg[col], marker="o", linewidth=2, color="darkorange")
        if col == "auc":
            ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.6, label="Random baseline")
            ax.legend(fontsize=8)
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Noise Multiplier σ")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    plt.suptitle(
        f"FedLap Online-Phase: Membership Inference vs DP  |  "
        f"Dataset: {config.dataset.dataset_name}  |  Clip norm: {CLIP_NORM}  |  Epochs: {TRAIN_EPOCHS}",
        fontsize=11,
    )
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f"mi_dp_plot_{now}_{config.dataset.dataset_name}.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    LOGGER.info(f"Plot saved: {plot_path}")


if __name__ == "__main__":
    run()
