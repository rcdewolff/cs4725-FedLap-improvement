import json
import os
import random
from copy import deepcopy
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


def extract_sfv_grad(grads):
    if grads is None:
        return None
    if isinstance(grads, dict):
        if "SFV" in grads:
            sfv = grads["SFV"]
            if isinstance(sfv, list) and len(sfv) > 0:
                return sfv[0]
            if torch.is_tensor(sfv):
                return sfv
        for val in grads.values():
            res = extract_sfv_grad(val)
            if res is not None:
                return res
    elif isinstance(grads, list):
        for val in grads:
            res = extract_sfv_grad(val)
            if res is not None:
                return res
    return None


def compute_attack_metrics(clients_grads, clients):
    per_client = []
    for grads, client in zip(clients_grads, clients):
        sfv_grad = extract_sfv_grad(grads)
        if sfv_grad is None:
            continue
        scores = torch.norm(sfv_grad.detach(), dim=1).cpu().numpy()
        labels = client.graph.train_mask.cpu().numpy().astype(int)
        if scores.shape[0] != labels.shape[0]:
            node_ids = client.graph.node_ids.cpu().numpy()
            if scores.shape[0] > int(node_ids.max()):
                scores = scores[node_ids]
            else:
                continue
        if np.unique(labels).size < 2:
            continue
        auc = roc_auc_score(labels, scores)
        ap = average_precision_score(labels, scores)
        k = int(labels.sum())
        if k == 0 or k == len(labels):
            acc = float("nan")
            f1 = float("nan")
        else:
            threshold = np.partition(scores, -k)[-k]
            preds = scores >= threshold
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


def compute_comm_stats(clients_grads, coef):
    client_bytes = [grads_bytes(grads) for grads in clients_grads]
    total_client_bytes = int(sum(client_bytes))
    avg_client_bytes = float(np.mean(client_bytes)) if len(client_bytes) > 0 else 0.0
    agg_grads = sum_lod(clients_grads, coef)
    agg_bytes = int(grads_bytes(agg_grads))
    return {
        "client_bytes_total": total_client_bytes,
        "client_bytes_avg": avg_client_bytes,
        "aggregate_bytes": agg_bytes,
    }


def apply_dp_to_clients_grads(clients_grads):
    if not config.dp.enabled:
        return clients_grads
    for client_grads in clients_grads:
        clip_grads_(
            client_grads,
            config.dp.clip_norm,
            separate_sfv=config.dp.separate_sfv,
        )
        if config.dp.mode == "local":
            std = config.dp.noise_multiplier * config.dp.clip_norm
            add_noise_(client_grads, std)
    return clients_grads


def apply_dp_to_aggregate(grads, num_clients):
    if not config.dp.enabled or config.dp.mode != "central":
        return grads
    std = config.dp.noise_multiplier * config.dp.clip_norm
    if num_clients > 0:
        std = std / num_clients
    add_noise_(grads, std)
    return grads


def run_attack_round(
    server,
    epochs=1,
    data_type="f+s",
    smodel_type="Laplace",
    fmodel_type="GNN",
    structure_type="hop2vec",
):
    server.initialize_FL(
        smodel_type=smodel_type,
        fmodel_type=fmodel_type,
        data_type=data_type,
        structure_type=structure_type,
    )
    server.share_weights()

    num_nodes = sum([client.num_nodes() for client in server.clients])
    coef = [client.num_nodes() / num_nodes for client in server.clients]

    attack_history = []
    comm_history = []
    for _ in range(epochs):
        server.reset_trainings()
        server.set_train_mode()
        server.train_clients(eval_=False)

        clients_grads = server.get_grads()
        clients_grads = apply_dp_to_clients_grads(clients_grads)

        _, attack_summary = compute_attack_metrics(clients_grads, server.clients)
        attack_history.append(attack_summary)
        comm_history.append(compute_comm_stats(clients_grads, coef))

        grads = sum_lod(clients_grads, coef)
        grads = apply_dp_to_aggregate(grads, len(server.clients))
        server.share_grads(grads)
        server.update_models()

    if len(attack_history) == 0:
        attack_agg = {}
    else:
        attack_agg = {
            key: float(np.nanmean([entry.get(key, float("nan")) for entry in attack_history]))
            for key in attack_history[0].keys()
        }

    if len(comm_history) == 0:
        comm_agg = {}
    else:
        comm_agg = {
            key: float(np.mean([entry.get(key, 0.0) for entry in comm_history]))
            for key in comm_history[0].keys()
        }

    return attack_agg, comm_agg


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


def run_privacy_attack_experiment(
    seeds=(1, 2, 3),
    train_epochs=config.model.iterations,
    attack_epochs=1,
    data_type="f+s",
    smodel_type="Laplace",
    fmodel_type="GNN",
    structure_type="hop2vec",
):
    experiment_path = os.path.join(save_path, "privacy_attack")
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

            train_res = server_train.joint_train_g(
                epochs=train_epochs,
                smodel_type=smodel_type,
                fmodel_type=fmodel_type,
                FL=True,
                data_type=data_type,
                plot=False,
                log=False,
                structure_type=structure_type,
            )

            graph_attack = deepcopy(base_graph)
            subgraphs_attack = deepcopy(base_subgraphs)
            server_attack = build_server(graph_attack, subgraphs_attack)

            attack_res, comm_res = run_attack_round(
                server_attack,
                epochs=attack_epochs,
                data_type=data_type,
                smodel_type=smodel_type,
                fmodel_type=fmodel_type,
                structure_type=structure_type,
            )

            seed_results[mode_name] = {
                "test_acc": float(train_res["Average"]["Test Acc"]),
                "attack": attack_res,
                "comm": comm_res,
                "dp": {
                    "enabled": bool(config.dp.enabled),
                    "clip_norm": float(config.dp.clip_norm),
                    "noise_multiplier": float(config.dp.noise_multiplier),
                    "delta": float(config.dp.delta),
                    "mode": str(config.dp.mode),
                },
            }

        results[str(seed_value)] = seed_results

    file_name = os.path.join(
        experiment_path,
        f"privacy_attack_{now}_{config.dataset.dataset_name}.json",
    )
    with open(file_name, "w") as f:
        json.dump(results, f, indent=2)

    LOGGER.info(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    run_privacy_attack_experiment()
