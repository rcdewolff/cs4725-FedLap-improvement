"""
Three-condition online-phase defense ablation for FedLap.

This experiment separates clipping from Gaussian noise and keeps the defense
used during training consistent with the gradients observed by the attacker:

    none        -> raw training updates, raw attack gradient
    clip_only   -> clipped training updates, clipped attack gradient
    clip_noise  -> clipped/noised training updates, clipped/noised attack gradient

The existing DLG sweep couples clipping and noise for every positive noise
multiplier and clips the attack gradient even for its no-DP row. This runner is
intentionally separate so the original experiment remains reproducible.
"""

import os
import sys
import time
from copy import deepcopy
from dataclasses import dataclass

pythonpath = os.getcwd()
if pythonpath not in sys.path:
    sys.path.append(pythonpath)

simulations_path = os.path.join(pythonpath, "src", "simulations")
if simulations_path not in sys.path:
    sys.path.append(simulations_path)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import *
from src.server import Server
import gradient_inversion_attack as gia
import online_phase_dlg_experiment as dlg_base


SEEDS = [1, 2, 3]
TRAIN_EPOCHS = 100
ATTACK_CLIENT_IDX = 0

DLG_ITERS = 300
DLG_LR = 0.1
DLG_TV_COEF = 1e-4

CLIP_NORM = 1.0
NOISE_MULTIPLIER = 0.05


@dataclass(frozen=True)
class DefenseCondition:
    name: str
    clipping_enabled: bool
    noise_multiplier: float


CONDITIONS = [
    DefenseCondition("none", clipping_enabled=False, noise_multiplier=0.0),
    DefenseCondition("clip_only", clipping_enabled=True, noise_multiplier=0.0),
    DefenseCondition(
        "clip_noise",
        clipping_enabled=True,
        noise_multiplier=NOISE_MULTIPLIER,
    ),
]


def configure_condition(condition):
    """Configure the shared training loop for one ablation condition."""
    config.dp.enabled = condition.clipping_enabled or condition.noise_multiplier > 0
    config.dp.clip_norm = CLIP_NORM
    config.dp.noise_multiplier = condition.noise_multiplier
    config.dp.mode = "local"
    config.dp.separate_sfv = True


def apply_condition_to_grads(client_grads, condition):
    """Create exactly the client update that the attacker is assumed to see."""
    observed_grads = deepcopy(client_grads)
    if condition.clipping_enabled:
        clip_grads_(observed_grads, CLIP_NORM, separate_sfv=True)
    if condition.noise_multiplier > 0:
        std = condition.noise_multiplier * CLIP_NORM
        add_noise_(observed_grads, std)
    return observed_grads


def summarize_training_telemetry(server, condition):
    records = server.last_joint_train_g_dp_stats
    epoch_times = server.last_joint_train_g_epoch_times

    if not records:
        return {
            "raw_grad_norm_mean": float("nan"),
            "raw_grad_norm_median": float("nan"),
            "raw_grad_norm_p75": float("nan"),
            "raw_sfv_grad_norm_median": float("nan"),
            "raw_non_sfv_grad_norm_median": float("nan"),
            "would_clip_fraction": float("nan"),
            "actual_clip_fraction": float("nan"),
            "mean_applied_clip_scale": float("nan"),
            "mean_epoch_seconds": float(np.mean(epoch_times)) if epoch_times else float("nan"),
        }

    raw_norms = np.asarray([record["raw_grad_norm"] for record in records])
    sfv_norms = np.asarray([record["raw_sfv_grad_norm"] for record in records])
    non_sfv_norms = np.asarray(
        [record["raw_non_sfv_grad_norm"] for record in records]
    )
    would_clip = np.asarray([record["would_clip"] for record in records], dtype=float)
    clip_scales = np.asarray([record["min_clip_scale"] for record in records])

    return {
        "raw_grad_norm_mean": float(np.mean(raw_norms)),
        "raw_grad_norm_median": float(np.median(raw_norms)),
        "raw_grad_norm_p75": float(np.percentile(raw_norms, 75)),
        "raw_sfv_grad_norm_median": float(np.median(sfv_norms)),
        "raw_non_sfv_grad_norm_median": float(np.median(non_sfv_norms)),
        "would_clip_fraction": float(np.mean(would_clip)),
        "actual_clip_fraction": (
            float(np.mean(would_clip)) if condition.clipping_enabled else 0.0
        ),
        "mean_applied_clip_scale": (
            float(np.mean(clip_scales)) if condition.clipping_enabled else 1.0
        ),
        "mean_epoch_seconds": float(np.mean(epoch_times)),
    }


def _plot_metric_bars(ax, summary, metric, ylabel, title, log_scale=False):
    labels = [condition.name for condition in CONDITIONS]
    means = []
    errors = []
    for label in labels:
        values = summary[summary["condition"] == label][metric]
        means.append(float(values.mean()))
        errors.append(float(values.std(ddof=1)) if len(values) > 1 else 0.0)

    positions = np.arange(len(labels))
    ax.bar(positions, means, yerr=errors, capsize=4)
    ax.set_xticks(positions, labels, rotation=20)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_scale and all(value > 0 for value in means):
        ax.set_yscale("log")


def plot_ablation(summary, output_dir):
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    _plot_metric_bars(
        axes[0], summary, "raw_grad_norm_median", "L2 norm", "Raw update norm"
    )
    _plot_metric_bars(
        axes[1], summary, "would_clip_fraction", "Fraction", "Updates above C"
    )
    _plot_metric_bars(
        axes[2], summary, "test_acc", "Test accuracy", "Model utility"
    )
    _plot_metric_bars(
        axes[3], summary, "mse", "MSE", "DLG reconstruction error", log_scale=True
    )

    fig.suptitle(
        f"FedLap defense ablation - {config.dataset.dataset_name} - "
        f"{TRAIN_EPOCHS} epochs - C={CLIP_NORM} - noise={NOISE_MULTIPLIER}"
    )
    fig.tight_layout()
    plot_path = os.path.join(
        output_dir,
        f"defense_ablation_{now}_{config.dataset.dataset_name}.png",
    )
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    LOGGER.info(f"Ablation plot saved to {plot_path}")


def run_defense_ablation_experiment():
    output_dir = os.path.join(save_path, "privacy_attack", "dlg_ablation")
    os.makedirs(output_dir, exist_ok=True)
    summary_path = os.path.join(
        output_dir,
        f"defense_ablation_{now}_{config.dataset.dataset_name}.csv",
    )
    telemetry_path = os.path.join(
        output_dir,
        f"defense_ablation_telemetry_{now}_{config.dataset.dataset_name}.csv",
    )

    original_dp = {
        "enabled": bool(config.dp.enabled),
        "clip_norm": float(config.dp.clip_norm),
        "noise_multiplier": float(config.dp.noise_multiplier),
        "mode": str(config.dp.mode),
        "separate_sfv": bool(config.dp.separate_sfv),
    }

    summary_rows = []
    telemetry_rows = []

    try:
        for seed_value in SEEDS:
            dlg_base.set_seed(seed_value)
            LOGGER.info(f"\n=== Ablation seed {seed_value} ===")

            base_graph, base_subgraphs = dlg_base.prepare_graph_and_subgraphs()
            LOGGER.info("Initializing FL once for all defense conditions...")
            init_server = dlg_base.initialize_server(
                deepcopy(base_graph), deepcopy(base_subgraphs)
            )

            for condition in CONDITIONS:
                configure_condition(condition)
                dlg_base.set_seed(seed_value)
                server = deepcopy(init_server)
                run_started = time.perf_counter()

                LOGGER.info(
                    f"Starting condition={condition.name}, seed={seed_value}, "
                    f"epochs={TRAIN_EPOCHS}."
                )
                training_started = time.perf_counter()
                train_results = Server.joint_train_g(
                    server,
                    epochs=TRAIN_EPOCHS,
                    FL=True,
                    log=False,
                    plot=False,
                    model_type=f"FL f+s Laplace-GNN GA [{condition.name}]",
                    log_epoch_time=True,
                    collect_dp_stats=True,
                )
                training_seconds = time.perf_counter() - training_started

                telemetry = summarize_training_telemetry(server, condition)
                for record in server.last_joint_train_g_dp_stats:
                    telemetry_rows.append(
                        {
                            "dataset": config.dataset.dataset_name,
                            "seed": seed_value,
                            "condition": condition.name,
                            "clipping_enabled": condition.clipping_enabled,
                            "noise_multiplier": condition.noise_multiplier,
                            **record,
                        }
                    )

                raw_grads = dlg_base.capture_raw_client_grads(
                    server, ATTACK_CLIENT_IDX
                )

                # Use fixed seeds for attack noise and DLG initialization so the
                # comparison is not confounded by condition ordering.
                dlg_base.set_seed(100_000 + seed_value)
                observed_grads = apply_condition_to_grads(raw_grads, condition)
                target_tensors = gia.get_fmodel_target_grads(observed_grads)
                if not target_tensors:
                    LOGGER.info("No fmodel gradients; skipping this condition.")
                    continue

                dlg_base.set_seed(200_000 + seed_value)
                attack_started = time.perf_counter()
                client = server.clients[ATTACK_CLIENT_IDX]
                x_real = client.classifier.fmodel.graph.x
                x_rec, _ = gia.gradient_inversion_attack(
                    client=client,
                    target_grads=target_tensors,
                    n_iters=DLG_ITERS,
                    lr=DLG_LR,
                    tv_coef=DLG_TV_COEF,
                )
                attack_seconds = time.perf_counter() - attack_started
                metrics = gia.compute_reconstruction_metrics(x_real, x_rec)

                row = {
                    "dataset": config.dataset.dataset_name,
                    "seed": seed_value,
                    "condition": condition.name,
                    "train_epochs": TRAIN_EPOCHS,
                    "clipping_enabled": condition.clipping_enabled,
                    "clip_norm": CLIP_NORM if condition.clipping_enabled else 0.0,
                    "noise_multiplier": condition.noise_multiplier,
                    "test_acc": float(train_results["Average"]["Test Acc"]),
                    **metrics,
                    **telemetry,
                    "training_seconds": training_seconds,
                    "attack_seconds": attack_seconds,
                    "run_seconds": time.perf_counter() - run_started,
                    "dlg_iters": DLG_ITERS,
                }
                summary_rows.append(row)
                # Checkpoint after every completed condition so a long run can
                # resume its analysis even if a later condition fails.
                pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
                pd.DataFrame(telemetry_rows).to_csv(telemetry_path, index=False)
                LOGGER.info(f"Ablation result: {row}")
    finally:
        config.dp.enabled = original_dp["enabled"]
        config.dp.clip_norm = original_dp["clip_norm"]
        config.dp.noise_multiplier = original_dp["noise_multiplier"]
        config.dp.mode = original_dp["mode"]
        config.dp.separate_sfv = original_dp["separate_sfv"]

    summary = pd.DataFrame(summary_rows)
    telemetry = pd.DataFrame(telemetry_rows)
    if summary.empty:
        LOGGER.info("No ablation results collected.")
        return summary

    summary.to_csv(summary_path, index=False)
    telemetry.to_csv(telemetry_path, index=False)
    plot_ablation(summary, output_dir)

    LOGGER.info(f"Ablation summary saved to {summary_path}")
    LOGGER.info(f"Ablation telemetry saved to {telemetry_path}")
    return summary


if __name__ == "__main__":
    run_defense_ablation_experiment()
