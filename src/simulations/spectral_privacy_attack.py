import json
import os
from copy import deepcopy
import random
import sys

pythonpath = os.getcwd()
if pythonpath not in sys.path:
    sys.path.append(pythonpath)

import numpy as np
import torch
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score

from src import *
from src.GNN.GNN_server import GNNServer
from src.utils.define_graph import define_graph
from src.utils.graph_partitioning import partition_graph
import src.utils.utils as utils_module


def set_seed(value):
    utils_module.seed = value
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def tensor_to_numpy(tensor):
    if tensor is None:
        return None
    if torch.is_tensor(tensor):
        return tensor.detach().cpu().numpy()
    return np.asarray(tensor)


def align_rows(matrix, node_ids, local_size):
    if matrix is None:
        return None

    matrix_np = tensor_to_numpy(matrix)
    if matrix_np is None or matrix_np.ndim == 0:
        return None

    num_rows = matrix_np.shape[0]
    node_ids_np = tensor_to_numpy(node_ids).astype(int)
    max_node_id = int(node_ids_np.max()) if node_ids_np.size > 0 else -1

    # Prefer direct node-id indexing only when the shared matrix is clearly global.
    if num_rows > max_node_id:
        return matrix_np[node_ids_np]

    # Otherwise assume the matrix is already aligned to the local node order.
    if num_rows == local_size or num_rows == node_ids_np.size:
        return matrix_np

    # Last resort: if the matrix has at least as many rows as the local graph,
    # truncate to the local size instead of indexing by global ids.
    if num_rows > local_size:
        return matrix_np[:local_size]

    return matrix_np


def spectral_row_score(U, D=None):
    U_np = tensor_to_numpy(U)
    if U_np is None:
        return None

    if D is None:
        weights = np.ones(U_np.shape[1], dtype=U_np.dtype)
    else:
        D_np = tensor_to_numpy(D)
        if D_np.ndim == 2:
            D_np = np.diag(D_np)
        if D_np.ndim == 1 and D_np.size > 0:
            weights = np.sqrt(np.abs(D_np[: U_np.shape[1]]))
            if weights.shape[0] < U_np.shape[1]:
                weights = np.pad(weights, (0, U_np.shape[1] - weights.shape[0]), constant_values=1.0)
        else:
            weights = np.ones(U_np.shape[1], dtype=U_np.dtype)

    weighted = U_np * weights[None, :]
    return np.linalg.norm(weighted, axis=1)


def compute_scores_from_spectral_share(share, clients):
    # share expected to contain spectral basis data. We use node-level scores from
    # U/D when possible, and only fall back to SFV if it is row-aligned to a node set.
    U = share.get("U", None)
    D = share.get("D", None)
    SFV = share.get("SFV", None)
    per_client = []

    if U is None and SFV is None:
        return per_client, {}

    for client in clients:
        local_size = int(client.graph.node_ids.numel())
        labels = tensor_to_numpy(client.graph.train_mask).astype(int)

        score = None

        if U is not None:
            U_c = align_rows(U, client.graph.node_ids, local_size)
            if U_c is not None and U_c.ndim == 2 and U_c.shape[0] == local_size:
                score = spectral_row_score(U_c, D)

        # Only use SFV if it is actually node-aligned. In this codebase the shared
        # SFV may be a feature matrix, so do not index it by global node ids.
        if score is None and SFV is not None:
            SFV_c = align_rows(SFV, client.graph.node_ids, local_size)
            if SFV_c is not None and SFV_c.ndim == 2 and SFV_c.shape[0] == local_size:
                score = np.linalg.norm(SFV_c, axis=1)

        if score is None:
            continue

        if score.shape[0] != labels.shape[0]:
            continue
        if np.unique(labels).size < 2:
            continue

        auc = roc_auc_score(labels, score)
        ap = average_precision_score(labels, score)
        k = int(labels.sum())
        if k == 0 or k == len(labels):
            acc = float("nan")
            f1 = float("nan")
        else:
            threshold = np.partition(score, -k)[-k]
            preds = score >= threshold
            acc = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds)

        per_client.append({"AUC": auc, "AP": ap, "Acc": acc, "F1": f1})

    if len(per_client) == 0:
        return per_client, {}

    summary = {}
    keys = per_client[0].keys()
    for key in keys:
        vals = [entry[key] for entry in per_client]
        summary[key] = float(np.nanmean(vals))

    return per_client, summary


def build_server(graph, subgraphs):
    server = GNNServer(graph)
    for subgraph in subgraphs:
        server.add_client(subgraph)
    return server


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


def run_spectral_privacy_attack_experiment(
    seeds=(1, 2, 3),
    smodel_type="SpectralLaplace",
    fmodel_type="GNN",
    structure_type="hop2vec",
):
    experiment_path = os.path.join(save_path, "spectral_privacy_attack")
    os.makedirs(experiment_path, exist_ok=True)

    results = {}
    for seed_value in seeds:
        set_seed(seed_value)
        base_graph, base_subgraphs = prepare_graph_and_subgraphs()

        seed_results = {}
        for mode_name, dp_enabled in [("baseline", False), ("dp", True)]:
            config.dp.enabled = dp_enabled

            graph_train = deepcopy(base_graph)
            subgraphs_train = deepcopy(base_subgraphs)
            server_train = build_server(graph_train, subgraphs_train)

            # initialize to get spectral share; do not need to train
            share = server_train.initialize(
                smodel_type=smodel_type,
                fmodel_type=fmodel_type,
                data_type="f+s",
                spectral_len=config.spectral.spectral_len,
                structure_type=structure_type,
                log=False,
            )

            # attack uses the offline shared artifacts (U, D, SFV)
            per_client, summary = compute_scores_from_spectral_share(share, server_train.clients)

            seed_results[mode_name] = {
                "attack": summary,
                "dp": {
                    "enabled": bool(config.dp.enabled),
                    "clip_norm": float(config.dp.clip_norm),
                    "noise_multiplier": float(config.dp.noise_multiplier),
                    "delta": float(config.dp.delta),
                    "mode": str(config.dp.mode),
                },
            }

            LOGGER.info(f"Seed {seed_value} mode {mode_name} attack summary: {summary}")

        results[str(seed_value)] = seed_results

    file_name = os.path.join(
        experiment_path,
        f"spectral_privacy_attack_{now}_{config.dataset.dataset_name}.json",
    )
    with open(file_name, "w") as f:
        json.dump(results, f, indent=2)

    LOGGER.info(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    run_spectral_privacy_attack_experiment()
