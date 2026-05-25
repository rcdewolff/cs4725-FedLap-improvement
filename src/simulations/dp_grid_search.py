import os
import sys
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import torch

pythonpath = os.getcwd()
if pythonpath not in sys.path:
    sys.path.append(pythonpath)

simulations_path = os.path.join(pythonpath, "src", "simulations")
if simulations_path not in sys.path:
    sys.path.append(simulations_path)

from src import *
import src.utils.utils as utils_module
from src.GNN.GNN_server import GNNServer
from src.utils.define_graph import define_graph
from src.utils.graph_partitioning import partition_graph
import privacy_attack_experiment as pae


# Keep this small for a less expensive grid.
NOISE_MULTIPLIERS = [0.0, 0.0025, 0.005, 0.01, 0.02]
CLIP_NORMS = [0.25, 0.5, 1.0, 2.0]
SEEDS = [1, 2, 3]

TRAIN_EPOCHS = 30
ATTACK_EPOCHS = 1

DATA_TYPE = "f+s"
SMODEL_TYPE = "Laplace"
FMODEL_TYPE = "GNN"
STRUCTURE_TYPE = "hop2vec"

INCLUDE_BASELINE = True


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


def run_one(base_graph, base_subgraphs):
    graph_train = deepcopy(base_graph)
    subgraphs_train = deepcopy(base_subgraphs)
    server_train = build_server(graph_train, subgraphs_train)

    train_res = server_train.joint_train_g(
        epochs=TRAIN_EPOCHS,
        smodel_type=SMODEL_TYPE,
        fmodel_type=FMODEL_TYPE,
        FL=True,
        data_type=DATA_TYPE,
        plot=False,
        log=False,
        structure_type=STRUCTURE_TYPE,
    )

    graph_attack = deepcopy(base_graph)
    subgraphs_attack = deepcopy(base_subgraphs)
    server_attack = build_server(graph_attack, subgraphs_attack)

    attack_res, comm_res = pae.run_attack_round(
        server_attack,
        epochs=ATTACK_EPOCHS,
        data_type=DATA_TYPE,
        smodel_type=SMODEL_TYPE,
        fmodel_type=FMODEL_TYPE,
        structure_type=STRUCTURE_TYPE,
    )

    return train_res, attack_res, comm_res


def main():
    output_dir = os.path.join(save_path, "privacy_attack", "grid_search")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(
        output_dir,
        f"dp_grid_{now}_{config.dataset.dataset_name}.csv",
    )

    rows = []
    for seed_value in SEEDS:
        set_seed(seed_value)
        base_graph, base_subgraphs = prepare_graph_and_subgraphs()

        if INCLUDE_BASELINE:
            config.dp.enabled = False
            train_res, attack_res, comm_res = run_one(base_graph, base_subgraphs)
            rows.append(
                {
                    "seed": seed_value,
                    "dp_enabled": False,
                    "clip_norm": 0.0,
                    "noise_multiplier": 0.0,
                    "test_acc": float(train_res["Average"]["Test Acc"]),
                    "attack_auc": float(attack_res.get("AUC", float("nan"))),
                    "attack_ap": float(attack_res.get("AP", float("nan"))),
                    "attack_acc": float(attack_res.get("Acc", float("nan"))),
                    "attack_f1": float(attack_res.get("F1", float("nan"))),
                    "client_bytes_total": float(comm_res.get("client_bytes_total", float("nan"))),
                    "client_bytes_avg": float(comm_res.get("client_bytes_avg", float("nan"))),
                    "aggregate_bytes": float(comm_res.get("aggregate_bytes", float("nan"))),
                }
            )

        for clip_norm in CLIP_NORMS:
            for noise_multiplier in NOISE_MULTIPLIERS:
                config.dp.enabled = True
                config.dp.clip_norm = clip_norm
                config.dp.noise_multiplier = noise_multiplier
                config.dp.mode = "local"
                config.dp.separate_sfv = True

                train_res, attack_res, comm_res = run_one(base_graph, base_subgraphs)
                rows.append(
                    {
                        "seed": seed_value,
                        "dp_enabled": True,
                        "clip_norm": clip_norm,
                        "noise_multiplier": noise_multiplier,
                        "test_acc": float(train_res["Average"]["Test Acc"]),
                        "attack_auc": float(attack_res.get("AUC", float("nan"))),
                        "attack_ap": float(attack_res.get("AP", float("nan"))),
                        "attack_acc": float(attack_res.get("Acc", float("nan"))),
                        "attack_f1": float(attack_res.get("F1", float("nan"))),
                        "client_bytes_total": float(comm_res.get("client_bytes_total", float("nan"))),
                        "client_bytes_avg": float(comm_res.get("client_bytes_avg", float("nan"))),
                        "aggregate_bytes": float(comm_res.get("aggregate_bytes", float("nan"))),
                    }
                )

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    LOGGER.info(f"Saved grid results to {csv_path}")


if __name__ == "__main__":
    main()
