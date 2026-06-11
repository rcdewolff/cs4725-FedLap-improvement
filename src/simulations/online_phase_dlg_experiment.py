"""
Online-phase gradient inversion experiment for FedLap.

Demonstrates that the FedLap online phase (gradient sharing) is vulnerable to
DLG-style gradient inversion attacks without DP, and that DP mitigates the attack
at the cost of model accuracy (the real privacy-utility tradeoff).

The offline phase (spectral decomposition via decentralised Arnoldi iteration) is
secured by homomorphic encryption as described in the FedLap paper and is NOT
the target here.

Experiment flow per (seed, noise_multiplier):
  1. Initialize FL once per seed (hop2vec structural features computed here only).
  2. Deepcopy the initialized server for each noise level.
  3. Train FedLap for TRAIN_EPOCHS rounds WITH DP applied at the correct place
     (every gradient upload during training) — this shows the accuracy cost.
  4. After training, capture raw gradients from one attack round, apply DP noise,
     and run the DLG attack — this shows the privacy benefit.
  5. Record test_acc (utility) and MSE/cosine_sim/feat_corr (attack quality).

Results are saved as a CSV and a summary plot showing both sides of the tradeoff.
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
import gradient_inversion_attack as gia


# ---------------------------------------------------------------------------
# Experiment hyper-parameters
# ---------------------------------------------------------------------------

SEEDS = [1, 2, 3]
EPOCH_COUNTS = [10, 30, 100]   # sweep to show effect of training length on the tradeoff
ATTACK_CLIENT_IDX = 0

DLG_ITERS = 300
DLG_LR = 0.1
DLG_TV_COEF = 1e-4

# 0.0 = no DP baseline; rest are local DP noise levels
NOISE_MULTIPLIERS = [0.0, 0.005, 0.01, 0.02, 0.05]
CLIP_NORM = 1.0

DATA_TYPE = "f+s"
SMODEL_TYPE = "Laplace"
FMODEL_TYPE = "GNN"
STRUCTURE_TYPE = "hop2vec"


# ---------------------------------------------------------------------------
# Helpers
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
    """
    Build server and run FL initialization (computes structural features).
    Returns the initialized server with all client classifiers set up and
    weights synced — ready to train directly without re-initialization.
    """
    server = build_server(graph, subgraphs)
    server.initialize_FL(
        smodel_type=SMODEL_TYPE,
        fmodel_type=FMODEL_TYPE,
        data_type=DATA_TYPE,
        structure_type=STRUCTURE_TYPE,
    )
    server.share_weights()
    return server


def train_initialized_server(server, noise_mult, epochs):
    """
    Run FL training on an already-initialized server with DP applied during
    every gradient communication round (the correct location in the online phase).

    Calls Server.joint_train_g directly to skip GNNServer's re-initialization.
    config.dp is set here so the training loop in server.py picks it up.
    """
    dp_enabled = noise_mult > 0
    config.dp.enabled = dp_enabled
    config.dp.noise_multiplier = noise_mult
    config.dp.clip_norm = CLIP_NORM
    config.dp.mode = "local"
    config.dp.separate_sfv = True

    # Server.joint_train_g runs the training loop with DP via config.dp —
    # bypasses GNNServer.joint_train_g which would re-initialize the model.
    return Server.joint_train_g(
        server,
        epochs=epochs,
        FL=True,
        log=False,
        plot=False,
        model_type="FL f+s Laplace-GNN GA",
    )


def capture_raw_client_grads(server, client_idx):
    """
    Run one additional FL step and return raw per-client gradients before
    any aggregation or DP — what the adversary intercepts from the upload.
    """
    # Temporarily disable DP so we capture the raw gradient, then apply it
    # manually below to control exactly what noise level the attacker sees.
    saved_dp = config.dp.enabled
    config.dp.enabled = False

    server.reset_trainings()
    server.set_train_mode()
    server.train_clients(eval_=False)
    clients_grads = server.get_grads()

    config.dp.enabled = saved_dp
    return clients_grads[client_idx]


def apply_dp_to_grads(client_grads, noise_multiplier, clip_norm):
    """Clip + Gaussian noise — simulates what a local-DP client sends."""
    grads = deepcopy(client_grads)
    clip_grads_(grads, clip_norm, separate_sfv=True)
    if noise_multiplier > 0:
        std = noise_multiplier * clip_norm
        add_noise_(grads, std)
    return grads


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


def run_dlg_experiment():
    output_dir = os.path.join(save_path, "privacy_attack", "dlg")
    os.makedirs(output_dir, exist_ok=True)

    rows = []

    for seed_value in SEEDS:
        set_seed(seed_value)
        LOGGER.info(f"\n=== Seed {seed_value} ===")

        base_graph, base_subgraphs = prepare_graph_and_subgraphs()

        # Run FL initialization ONCE per seed — this is where the expensive
        # structural feature computation (hop2vec) happens. We deepcopy the
        # result for each noise level so we don't repeat it.
        LOGGER.info("  Initializing FL (computing structural features)...")
        init_server = initialize_server(
            deepcopy(base_graph), deepcopy(base_subgraphs)
        )

        for train_epochs in EPOCH_COUNTS:
            LOGGER.info(f"  --- train_epochs={train_epochs} ---")
            for noise_mult in NOISE_MULTIPLIERS:
                dp_enabled = noise_mult > 0
                LOGGER.info(f"    noise_multiplier={noise_mult:.4f}  dp_enabled={dp_enabled}")

                # Fresh copy of initialized server with the same structural features
                # and initial weights — different DP noise → different training trajectory.
                server = deepcopy(init_server)

                # Train WITH DP applied at every gradient upload (the correct place).
                # This gives the real accuracy cost of DP.
                train_results = train_initialized_server(server, noise_mult, train_epochs)
                test_acc = float(train_results["Average"]["Test Acc"])
                LOGGER.info(f"      test_acc={test_acc:.4f}")

                # Capture one attack round: raw gradient from the trained model.
                # Apply DP noise to simulate what the adversarial server observes.
                raw_grads = capture_raw_client_grads(server, ATTACK_CLIENT_IDX)
                observed_grads = apply_dp_to_grads(raw_grads, noise_mult, CLIP_NORM)

                target_tensors = gia.get_fmodel_target_grads(observed_grads)
                if not target_tensors:
                    LOGGER.info("      No fmodel gradients — skipping attack.")
                    continue

                client = server.clients[ATTACK_CLIENT_IDX]
                x_real = client.classifier.fmodel.graph.x

                x_rec, _ = gia.gradient_inversion_attack(
                    client=client,
                    target_grads=target_tensors,
                    n_iters=DLG_ITERS,
                    lr=DLG_LR,
                    tv_coef=DLG_TV_COEF,
                )

                metrics = gia.compute_reconstruction_metrics(x_real, x_rec)
                LOGGER.info(
                    f"      MSE={metrics['mse']:.6f}  "
                    f"cos_sim={metrics['cosine_sim']:.4f}  "
                    f"feat_corr={metrics['feat_corr']:.4f}"
                )

                rows.append({
                    "seed": seed_value,
                    "train_epochs": train_epochs,
                    "noise_multiplier": noise_mult,
                    "dp_enabled": dp_enabled,
                    "clip_norm": CLIP_NORM if dp_enabled else 0.0,
                    "test_acc": test_acc,
                    "mse": metrics["mse"],
                    "cosine_sim": metrics["cosine_sim"],
                    "feat_corr": metrics["feat_corr"],
                    "dlg_iters": DLG_ITERS,
                })

    if not rows:
        LOGGER.info("No results collected.")
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    csv_path = os.path.join(
        output_dir, f"dlg_epochs_{now}_{config.dataset.dataset_name}.csv"
    )
    df.to_csv(csv_path, index=False)
    LOGGER.info(f"\nSaved results to {csv_path}")

    _plot_results(df, output_dir)
    return df


def _plot_results(df, output_dir):
    epoch_counts = sorted(df["train_epochs"].unique())
    colors = plt.cm.tab10.colors

    panel_cfg = [
        ("mse",        "MSE",                "Attack quality (↑ = better privacy)"),
        ("cosine_sim", "Cosine Similarity",   "Attack quality (↓ = better privacy)"),
        ("feat_corr",  "Feature Correlation", "Attack quality (↓ = better privacy)"),
        ("test_acc",   "Test Accuracy",       "Model utility (↑ = better utility)"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    for ax, (col, ylabel, title) in zip(axes, panel_cfg):
        for idx, ep in enumerate(epoch_counts):
            agg = (
                df[df["train_epochs"] == ep]
                .groupby("noise_multiplier")[col]
                .mean()
                .reset_index()
            )
            ax.plot(
                agg["noise_multiplier"],
                agg[col],
                marker="o",
                linewidth=2,
                color=colors[idx % len(colors)],
                label=f"{ep} epochs",
            )
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Noise Multiplier σ")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(fontsize=8)

    plt.suptitle(
        f"FedLap Online-Phase: Privacy-Utility Tradeoff — Effect of Training Length\n"
        f"Dataset: {config.dataset.dataset_name}  |  Clip norm: {CLIP_NORM}  |  "
        f"Epochs: {epoch_counts}",
        fontsize=11,
    )
    plt.tight_layout()

    plot_path = os.path.join(
        output_dir, f"dlg_epochs_plot_{now}_{config.dataset.dataset_name}.png"
    )
    plt.savefig(plot_path, dpi=150)
    plt.close()
    LOGGER.info(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    run_dlg_experiment()
