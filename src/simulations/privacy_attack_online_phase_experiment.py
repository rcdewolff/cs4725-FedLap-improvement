from __future__ import annotations

import json
import os
import random
import sys
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

pythonpath = os.getcwd()
if pythonpath not in sys.path:
    sys.path.append(pythonpath)

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

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


def _detach_clone(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().clone()
    return value


def _to_dense_tensor(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        if value.is_sparse:
            return value.to_dense()
        return value
    return None


def _resolve_structural_tensor(target_client):
    classifier = getattr(target_client, "classifier", None)
    if classifier is None:
        return None

    smodel = getattr(classifier, "smodel", None)
    if smodel is None:
        return None

    model = getattr(smodel, "model", None)
    if model is not None:
        candidate = None
        for parameter in model.parameters():
            if torch.is_tensor(parameter) and parameter.ndim >= 2:
                return parameter
            if candidate is None and torch.is_tensor(parameter):
                candidate = parameter
        if candidate is not None:
            return candidate

    if hasattr(smodel, "W"):
        tensor = getattr(smodel, "W")
        if torch.is_tensor(tensor):
            return tensor

    if hasattr(smodel, "graph") and torch.is_tensor(smodel.graph.x):
        return smodel.graph.x

    if hasattr(smodel, "get_W"):
        tensor = smodel.get_W()
        if torch.is_tensor(tensor):
            return tensor

    return None


def _resolve_rank(attack_state):
    if attack_state.r > 0:
        return int(attack_state.r)
    rank = int(getattr(config.spectral, "lanczos_iter", 0) or 0)
    if rank <= 0:
        raise ValueError("Missing spectral rank. Set attack_state.r or config.spectral.lanczos_iter.")
    attack_state.r = rank
    return rank


def _build_cross_edge_matrix(graph, left_nodes, right_nodes, device=None):
    if device is None:
        device = graph.edge_index.device

    matrix = torch.zeros((len(left_nodes), len(right_nodes)), dtype=torch.float32, device=device)
    if len(left_nodes) == 0 or len(right_nodes) == 0:
        return matrix

    left_lookup = {int(node_id): index for index, node_id in enumerate(left_nodes.tolist())}
    right_lookup = {int(node_id): index for index, node_id in enumerate(right_nodes.tolist())}

    edge_index = graph.get_edges() if hasattr(graph, "get_edges") else graph.edge_index
    if edge_index is None or edge_index.numel() == 0:
        return matrix

    for src, dst in edge_index.t().tolist():
        src = int(src)
        dst = int(dst)
        if src in left_lookup and dst in right_lookup:
            matrix[left_lookup[src], right_lookup[dst]] = 1.0
        if dst in left_lookup and src in right_lookup:
            matrix[left_lookup[dst], right_lookup[src]] = 1.0

    return matrix


def _build_real_offline_state(server, target_subgraph, target_client_id, smodel_type, data_type, structure_type):
    offline_state = {}
    rank = int(getattr(config.spectral, "lanczos_iter", 0) or 0)
    share = server.initialize_FL(
        smodel_type=smodel_type,
        fmodel_type=config.model.fmodel_type,
        data_type=data_type,
        spectral_len=rank,
        structure_type=structure_type,
    )

    global_u = share.get("U", None)
    if torch.is_tensor(global_u):
        global_u = global_u.detach().clone()
        if rank > 0 and global_u.ndim >= 2 and global_u.shape[1] > rank:
            global_u = global_u[:, :rank]
        target_nodes = target_subgraph.node_ids.long().cpu()
        all_nodes = server.graph.node_ids.long().cpu()
        mask = torch.ones(all_nodes.shape[0], dtype=torch.bool)
        mask[target_nodes] = False
        other_nodes = all_nodes[mask]

        q_v2 = None
        if 0 <= target_client_id < len(server.clients):
            target_client = server.clients[target_client_id]
            classifier = getattr(target_client, "classifier", None)
            smodel = getattr(classifier, "smodel", None)
            q_v2 = _detach_clone(getattr(smodel, "Q", None))
            if q_v2 is None and torch.is_tensor(getattr(smodel, "graph", None).x if getattr(smodel, "graph", None) is not None else None):
                q_v2 = _detach_clone(smodel.graph.x)

        if len(other_nodes) > 0:
            q_v1_true = global_u[other_nodes]
            a_v2_v1 = _build_cross_edge_matrix(server.graph, target_nodes, other_nodes, device=q_v1_true.device)
            tau_v2 = a_v2_v1 @ q_v1_true
            offline_state["q_v2"] = q_v2
            offline_state["a_v2_v1"] = a_v2_v1
            offline_state["tau_v2_history"] = [tau_v2]
            offline_state["h_r"] = share.get("H_r", None)
            offline_state["v_r"] = share.get("V_r", None)
            offline_state["sigma_r"] = share.get("D", None)

    return offline_state


def _collect_offline_state(
    server,
    attack_state,
    q_v2=None,
    a_v2_v1=None,
    h_r=None,
    v_r=None,
    sigma_r=None,
    tau_v2_history=None,
):
    if attack_state.H_r is None:
        attack_state.H_r = _detach_clone(h_r if h_r is not None else getattr(server, "H_r", None))
    if attack_state.V_r is None:
        attack_state.V_r = _detach_clone(v_r if v_r is not None else getattr(server, "V_r", None))
    if attack_state.Sigma_r is None:
        attack_state.Sigma_r = _detach_clone(sigma_r if sigma_r is not None else getattr(server, "Sigma_r", None))
    if attack_state.Q_V2 is None:
        attack_state.Q_V2 = _detach_clone(q_v2 if q_v2 is not None else getattr(server, "Q_V2", None))
    if attack_state.A_V2_V1 is None:
        attack_state.A_V2_V1 = _detach_clone(a_v2_v1 if a_v2_v1 is not None else getattr(server, "A_V2_V1", None))

    source_tau = tau_v2_history if tau_v2_history is not None else getattr(server, "tau_V2_history", None)
    if not attack_state.tau_V2_history and source_tau is not None:
        attack_state.tau_V2_history = [
            _detach_clone(tau_item) for tau_item in source_tau if tau_item is not None
        ]

    if attack_state.r <= 0:
        if torch.is_tensor(attack_state.H_r) and attack_state.H_r.ndim >= 1:
            attack_state.r = int(attack_state.H_r.shape[0])
        elif torch.is_tensor(attack_state.Q_V2) and attack_state.Q_V2.ndim >= 2:
            attack_state.r = int(attack_state.Q_V2.shape[1])
        elif torch.is_tensor(attack_state.Sigma_r) and attack_state.Sigma_r.ndim >= 1:
            attack_state.r = int(attack_state.Sigma_r.shape[0])
        else:
            attack_state.r = int(config.spectral.lanczos_iter)

    return attack_state


def _merge_non_none(base_state, override_state):
    merged = dict(base_state or {})
    for key, value in (override_state or {}).items():
        if value is not None:
            merged[key] = value
    return merged


def _align_gradient_for_lstsq(lhs, grad):
    rhs = grad.detach().float().cpu()
    if rhs.ndim == 1:
        rhs = rhs.unsqueeze(1)
    if rhs.shape[0] != lhs.shape[0] and rhs.ndim == 2 and rhs.shape[1] == lhs.shape[0]:
        rhs = rhs.T
    if rhs.shape[0] != lhs.shape[0]:
        raise ValueError(f"Gradient shape {tuple(rhs.shape)} incompatible with lhs shape {tuple(lhs.shape)}")
    return rhs


def _normalize_rows(matrix, target_row_norm):
    if matrix is None or matrix.numel() == 0:
        return matrix
    row_norms = matrix.norm(dim=1, keepdim=True).clamp_min(1e-12)
    scale = float(target_row_norm) / row_norms
    return matrix * scale


def _adjacency_from_edge_index(edge_index, num_nodes, device=None):
    if edge_index is None or edge_index.numel() == 0:
        if device is None:
            device = dev
        return torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    if device is None:
        device = edge_index.device
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=device)
    src = edge_index[0].long()
    dst = edge_index[1].long()
    adjacency[src, dst] = 1.0
    adjacency[dst, src] = 1.0
    adjacency.fill_diagonal_(0.0)
    return adjacency


def _normalized_laplacian_from_graph(graph):
    adjacency = _adjacency_from_edge_index(
        graph.edge_index if hasattr(graph, "edge_index") else graph.get_edges(),
        graph.num_nodes,
        device=graph.x.device if torch.is_tensor(graph.x) else dev,
    )
    degrees = adjacency.sum(dim=1)
    inv_sqrt = torch.zeros_like(degrees)
    mask = degrees > 0
    inv_sqrt[mask] = degrees[mask].pow(-0.5)
    d_inv_sqrt = torch.diag(inv_sqrt)
    identity = torch.eye(graph.num_nodes, device=adjacency.device)
    return identity - d_inv_sqrt @ adjacency @ d_inv_sqrt


def _majority_threshold_metrics(pred_scores, true_labels):
    pred_scores = pred_scores.flatten().float().cpu()
    true_labels = true_labels.flatten().float().cpu()

    if pred_scores.numel() == 0 or true_labels.numel() == 0:
        return {
            "precision": float("nan"),
            "recall": float("nan"),
            "precision_plus_recall": float("nan"),
            "best_threshold": float("nan"),
        }

    valid = torch.isfinite(pred_scores)
    pred_scores = pred_scores[valid]
    true_labels = true_labels[valid]

    if true_labels.unique().numel() < 2:
        return {
            "precision": float("nan"),
            "recall": float("nan"),
            "precision_plus_recall": float("nan"),
            "best_threshold": float("nan"),
        }

    lower = float(pred_scores.min().item())
    upper = float(pred_scores.max().item())
    thresholds = torch.linspace(lower, upper, 200)

    best = {
        "precision": float("nan"),
        "recall": float("nan"),
        "precision_plus_recall": float("nan"),
        "best_threshold": float("nan"),
    }

    positive_mask = true_labels > 0.5
    positive_total = positive_mask.sum().item()
    if positive_total == 0:
        return best

    for threshold in thresholds:
        predicted = pred_scores >= threshold
        tp = (predicted & positive_mask).sum().item()
        fp = (predicted & ~positive_mask).sum().item()
        fn = ((~predicted) & positive_mask).sum().item()

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        score = precision + recall

        if score > best["precision_plus_recall"]:
            best = {
                "precision": float(precision),
                "recall": float(recall),
                "precision_plus_recall": float(score),
                "best_threshold": float(threshold.item()),
            }

    return best


@dataclass
class AttackerState:
    grad_W_history: list[torch.Tensor] = field(default_factory=list)
    W_history: list[torch.Tensor] = field(default_factory=list)
    tau_V2_history: list[torch.Tensor] = field(default_factory=list)
    H_r: torch.Tensor | None = None
    V_r: torch.Tensor | None = None
    Sigma_r: torch.Tensor | None = None
    Q_V2: torch.Tensor | None = None
    A_V2_V1: torch.Tensor | None = None
    n1: int = 0
    r: int = config.spectral.lanczos_iter
    n_total: int = 0
    target_client_id: int = 0
    train_node_indices: torch.Tensor | None = None
    target_graph: Any | None = None


def _build_public_u_matrix(attacker_state, q_v1_hat):
    if attacker_state.Q_V2 is None or attacker_state.A_V2_V1 is None:
        return None

    if attacker_state.target_graph is None:
        return None

    if q_v1_hat is not None and attacker_state.r > 0 and q_v1_hat.ndim >= 2 and q_v1_hat.shape[1] > attacker_state.r:
        q_v1_hat = q_v1_hat[:, :attacker_state.r]

    adjacency = _adjacency_from_edge_index(
        attacker_state.target_graph.edge_index,
        attacker_state.target_graph.num_nodes,
        device=q_v1_hat.device,
    )
    degrees = adjacency.sum(dim=1)
    d_hat = torch.diag(degrees)

    # Compute aggregated influence of V1 embeddings on V2 nodes through cross edges.
    # v1_influence = A_V2_V1 @ q_v1_hat gives (344, rank) spectral embeddings for V2 nodes
    v1_influence = attacker_state.A_V2_V1.to(d_hat.device) @ q_v1_hat.to(d_hat.device)
    
    if v1_influence is None:
        return None

    # Apply Arnoldi regularization if available
    if attacker_state.H_r is None:
        return v1_influence

    return v1_influence - v1_influence @ attacker_state.H_r.to(v1_influence.device)


def _estimate_q_v1(attacker_state):
    LOGGER.info(f"attacker_state.A_V2_V1:\n{attacker_state.A_V2_V1}")

    LOGGER.info(f"attacker_state.tau_V2_history:\n{attacker_state.tau_V2_history}")

    if attacker_state.A_V2_V1 is None or len(attacker_state.tau_V2_history) == 0:
        return None

    a_matrix = _to_dense_tensor(attacker_state.A_V2_V1).float().cpu()
    if a_matrix is None or a_matrix.numel() == 0:
        return None

    tau_tensor = _to_dense_tensor(_detach_clone(attacker_state.tau_V2_history[0]))

    LOGGER.info(f"tau_tensor:\n{tau_tensor}")

    if tau_tensor is None:
        return None
    if tau_tensor.ndim == 1:
        tau_tensor = tau_tensor.unsqueeze(1)
    if tau_tensor.shape[0] != a_matrix.shape[0]:
        limit = min(tau_tensor.shape[0], a_matrix.shape[0])
        tau_tensor = tau_tensor[:limit]
        a_matrix = a_matrix[:limit]

    q_v1_hat = torch.linalg.pinv(a_matrix) @ tau_tensor.float().cpu()
    if attacker_state.r > 0 and q_v1_hat.ndim >= 2 and q_v1_hat.shape[1] > attacker_state.r:
        q_v1_hat = q_v1_hat[:, :attacker_state.r]
    return q_v1_hat


def _infer_adjacency(u_hat, q_v1_hat, attacker_state):
    if u_hat is None:
        return None

    full_u = _build_public_u_matrix(attacker_state, q_v1_hat)
    if full_u is None:
        return None

    full_u = full_u.clone()

    if u_hat is not None:
        if attacker_state.train_node_indices is not None:
            train_indices = attacker_state.train_node_indices.long().cpu()
            if train_indices.numel() == u_hat.shape[0]:
                full_u[train_indices] = 0.5 * full_u[train_indices] + 0.5 * u_hat.to(full_u.device)
            else:
                limit = min(train_indices.numel(), u_hat.shape[0], full_u.shape[0])
                full_u[train_indices[:limit]] = 0.5 * full_u[train_indices[:limit]] + 0.5 * u_hat[:limit].to(full_u.device)
        else:
            limit = min(u_hat.shape[0], full_u.shape[0])
            full_u[:limit] = 0.5 * full_u[:limit] + 0.5 * u_hat[:limit].to(full_u.device)

    if q_v1_hat is not None and q_v1_hat.ndim >= 2 and q_v1_hat.shape[0] == full_u.shape[0]:
        adjacency_hat = full_u @ torch.linalg.pinv(q_v1_hat.to(full_u.device))
    else:
        adjacency_hat = full_u @ full_u.T

    adjacency_hat = (adjacency_hat + adjacency_hat.T) / 2.0
    adjacency_hat = torch.clamp(adjacency_hat, min=0.0)
    adjacency_hat.fill_diagonal_(0.0)
    return adjacency_hat


def _evaluate_attack(adjacency_hat, true_graph):
    true_adj = _adjacency_from_edge_index(
        true_graph.edge_index if hasattr(true_graph, "edge_index") else true_graph.get_edges(),
        true_graph.num_nodes,
        device=adjacency_hat.device,
    )

    mask = torch.triu(torch.ones_like(true_adj, dtype=torch.bool), diagonal=1)
    pred_scores = adjacency_hat[mask]
    true_labels = true_adj[mask]

    result = _majority_threshold_metrics(pred_scores, true_labels)
    try:
        result["auc"] = float(roc_auc_score(true_labels.cpu().numpy(), pred_scores.detach().cpu().numpy()))
    except Exception:
        result["auc"] = float("nan")
    try:
        result["ap"] = float(average_precision_score(true_labels.cpu().numpy(), pred_scores.detach().cpu().numpy()))
    except Exception:
        result["ap"] = float("nan")
    result["edge_count"] = int(true_labels.sum().item())
    result["node_count"] = int(true_graph.num_nodes)
    return result


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


def collect_online_phase_state(
    server,
    epochs,
    target_client_id,
    attack_state,
    data_type="f+s",
    smodel_type="SpectralLaplace",
    fmodel_type="GNN",
    structure_type="hop2vec",
    debug_fast=False,
    skip_training=False,
):
    LOGGER.info("Initializing server FL...")
    server.initialize_FL(
        smodel_type=smodel_type,
        fmodel_type=fmodel_type,
        data_type=data_type,
        structure_type=structure_type,
    )
    server.share_weights()

    num_nodes = sum(client.num_nodes() for client in server.clients)
    coef = [client.num_nodes() / num_nodes for client in server.clients]

    if skip_training:
        target_client = server.clients[target_client_id]
        target_tensor = _resolve_structural_tensor(target_client)
        if target_tensor is not None:
            attack_state.W_history.append(target_tensor.detach().clone())
        return attack_state

    for i in range(epochs):
        LOGGER.info(f"Collecting online phase state in epoch {i + 1}/{epochs}...")

        server.reset_trainings()
        server.set_train_mode()
        server.train_clients(eval_=False)

        target_client = server.clients[target_client_id]
        target_tensor = _resolve_structural_tensor(target_client)
        target_grad_tensor = None if target_tensor is None else target_tensor.grad
        target_weight_tensor = None if target_tensor is None else target_tensor.detach()

        if target_grad_tensor is not None:
            attack_state.grad_W_history.append(target_grad_tensor.detach().clone())
        if target_weight_tensor is not None:
            attack_state.W_history.append(target_weight_tensor.detach().clone())

        clients_grads = apply_dp_to_clients_grads(server.get_grads())

        grads = sum_lod(clients_grads, coef)
        grads = apply_dp_to_aggregate(grads, len(server.clients))
        server.share_grads(grads)
        server.update_models()

        if debug_fast:
            break

    return attack_state


def invert_gradients_for_u(
    attacker_state,
    max_outer_iters=200,
    lr_u=0.05,
    lambda_deloc=0.1,
    lambda_arnoldi=0.1,
    convergence_threshold=1e-6,
):
    if len(attacker_state.grad_W_history) == 0:
        return None, {"reconstruction_loss": float("nan")}

    grad_history = [grad.detach().float().cpu() for grad in attacker_state.grad_W_history]
    grad_shape = grad_history[0].shape
    if len(grad_shape) == 1:
        grad_history = [grad.reshape(-1, 1) for grad in grad_history]
        grad_shape = grad_history[0].shape

    device = grad_history[0].device
    rank = int(grad_shape[0])
    if attacker_state.r != rank:
        attacker_state.r = rank
    num_train_nodes = attacker_state.train_node_indices.numel() if attacker_state.train_node_indices is not None else grad_shape[0]
    expected_norm = float(np.sqrt(rank / max(attacker_state.n_total, 1)))

    u_hat = torch.randn((num_train_nodes, rank), device=device)
    u_hat = _normalize_rows(u_hat, expected_norm)
    u_hat = torch.nn.Parameter(u_hat)
    optimizer = torch.optim.Adam([u_hat], lr=lr_u)

    best_loss = float("inf")
    best_u = None

    for _ in range(max_outer_iters):
        optimizer.zero_grad()

        deltas = []
        for grad in grad_history:
            lhs = u_hat.T
            rhs = _align_gradient_for_lstsq(lhs, grad)
            delta = torch.linalg.lstsq(lhs, rhs).solution
            deltas.append(delta.detach())

        reconstruction_loss = torch.zeros((), device=device)
        for grad, delta in zip(grad_history, deltas):
            reconstruction = u_hat.T @ delta
            reconstruction_loss = reconstruction_loss + torch.mean((reconstruction - grad) ** 2)

        delocalization_loss = torch.mean((u_hat.norm(dim=1) - expected_norm) ** 2)

        if debug_shape_only := False:
            pass

        arnoldi_loss = torch.zeros((), device=device)
        if attacker_state.target_graph is not None and attacker_state.Sigma_r is not None:
            q_v1_hat = _estimate_q_v1(attacker_state)
            public_u = _build_public_u_matrix(attacker_state, q_v1_hat)
            if public_u is not None:
                full_u = public_u.clone()
                if attacker_state.train_node_indices is not None:
                    train_indices = attacker_state.train_node_indices.long().cpu()
                    limit = min(train_indices.numel(), u_hat.shape[0], full_u.shape[0])
                    full_u[train_indices[:limit]] = 0.5 * full_u[train_indices[:limit]] + 0.5 * u_hat[:limit]
                else:
                    limit = min(u_hat.shape[0], full_u.shape[0])
                    full_u[:limit] = 0.5 * full_u[:limit] + 0.5 * u_hat[:limit]

                laplacian = _normalized_laplacian_from_graph(attacker_state.target_graph).to(device)
                sigma = attacker_state.Sigma_r.to(device).flatten()
                sigma = sigma[: full_u.shape[1]]
                arnoldi_loss = torch.mean((laplacian @ full_u - full_u * sigma.unsqueeze(0)) ** 2)

        total_loss = reconstruction_loss + lambda_deloc * delocalization_loss + lambda_arnoldi * arnoldi_loss
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            u_hat.data = _normalize_rows(u_hat.data, expected_norm)

        current_loss = float(total_loss.detach().cpu().item())
        if current_loss < best_loss:
            best_loss = current_loss
            best_u = u_hat.detach().clone()

        if current_loss < convergence_threshold:
            break

    metrics = {
        "reconstruction_loss": float(best_loss),
        "outer_iters": int(max_outer_iters),
        "train_node_count": int(num_train_nodes),
        "rank": int(rank),
    }
    return best_u, metrics


def run_online_phase_attack(
    server,
    attack_state,
    target_graph,
    attack_epochs=20,
    global_graph=None,
    offline_state=None,
    debug_fast=False,
    skip_training=False,
):
    attack_state.target_graph = target_graph
    attack_state.n1 = int(target_graph.num_nodes)
    if global_graph is not None:
        attack_state.n_total = int(global_graph.num_nodes)
    elif hasattr(server, "graph") and server.graph is not None:
        attack_state.n_total = int(server.graph.num_nodes)
    else:
        attack_state.n_total = int(target_graph.num_nodes)
    attack_state.train_node_indices = target_graph.train_mask.nonzero(as_tuple=False).flatten().cpu()

    attack_state = _collect_offline_state(server, attack_state, **(offline_state or {}))

    attack_state = collect_online_phase_state(
        server=server,
        epochs=attack_epochs,
        target_client_id=attack_state.target_client_id,
        attack_state=attack_state,
        debug_fast=debug_fast,
        skip_training=skip_training,
    )

    if skip_training:
        target_tensor = _resolve_structural_tensor(server.clients[attack_state.target_client_id])
        return {
            "debug": True,
            "skip_training": True,
            "rank": int(attack_state.r),
            "n1": int(attack_state.n1),
            "n_total": int(attack_state.n_total),
            "train_node_count": int(attack_state.train_node_indices.numel()) if attack_state.train_node_indices is not None else 0,
            "W_shape": list(target_tensor.shape) if target_tensor is not None else None,
            "tau_rounds": len(attack_state.tau_V2_history),
        }, {
            "debug_fast": True,
            "skip_training": True,
            "collected_rounds": len(attack_state.grad_W_history),
        }

    if debug_fast:
        return {
            "debug": True,
            "rank": int(attack_state.r),
            "n1": int(attack_state.n1),
            "n_total": int(attack_state.n_total),
            "train_node_count": int(attack_state.train_node_indices.numel()) if attack_state.train_node_indices is not None else 0,
            "grad_W_shape": list(attack_state.grad_W_history[-1].shape) if len(attack_state.grad_W_history) > 0 else None,
            "W_shape": list(attack_state.W_history[-1].shape) if len(attack_state.W_history) > 0 else None,
            "tau_rounds": len(attack_state.tau_V2_history),
        }, {
            "debug_fast": True,
            "collected_rounds": len(attack_state.grad_W_history),
        }

    u_hat, inversion_metrics = invert_gradients_for_u(attack_state)

    LOGGER.info(f"u_hat:\n{u_hat}")

    q_v1_hat = _estimate_q_v1(attack_state)

    LOGGER.info(f"q_v1_hat:\n{q_v1_hat}")
    adjacency_hat = _infer_adjacency(u_hat, q_v1_hat, attack_state)

    LOGGER.info(f"adjacency_hat:\n{adjacency_hat}")

    if adjacency_hat is None:
        return {}, inversion_metrics

    evaluation = _evaluate_attack(adjacency_hat, target_graph)
    evaluation.update(inversion_metrics)
    evaluation["collected_rounds"] = len(attack_state.grad_W_history)
    evaluation["tau_rounds"] = len(attack_state.tau_V2_history)
    return evaluation, inversion_metrics


def run_privacy_attack_online_phase_experiment(
    seeds=(1, 2, 3),
    train_epochs=config.model.iterations,
    attack_epochs=20,
    data_type="f+s",
    smodel_type="SpectralLaplace",
    fmodel_type="GNN",
    structure_type="hop2vec",
    target_client_id=0,
    q_v2=None,
    a_v2_v1=None,
    h_r=None,
    v_r=None,
    sigma_r=None,
    tau_v2_history=None,
    offline_state=None,
    debug_fast=False,
    skip_training=False,
):
    experiment_path = os.path.join(save_path, "privacy_attack_online_phase")
    os.makedirs(experiment_path, exist_ok=True)

    original_dp = {
        "enabled": bool(config.dp.enabled),
        "clip_norm": float(config.dp.clip_norm),
        "noise_multiplier": float(config.dp.noise_multiplier),
        "mode": str(config.dp.mode),
        "separate_sfv": bool(config.dp.separate_sfv),
    }

    results = {}
    for seed_value in seeds:
        set_seed(seed_value)
        base_graph, base_subgraphs = prepare_graph_and_subgraphs()
        global_node_count = int(base_graph.num_nodes)

        seed_results = {}
        for mode_name, dp_enabled in [("baseline", False), ("dp", True)]:
            LOGGER.info(f"Running online-phase attack with seed {seed_value} and mode {mode_name}.")

            config.dp.enabled = dp_enabled

            graph_train = deepcopy(base_graph)
            subgraphs_train = deepcopy(base_subgraphs)
            server_train = build_server(graph_train, subgraphs_train)

            effective_train_epochs = 1 if debug_fast else train_epochs
            effective_attack_epochs = 1 if debug_fast else attack_epochs

            LOGGER.info(f"Starting train on server train...")

            train_res = server_train.joint_train_g(
                epochs=effective_train_epochs,
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
            target_subgraph = subgraphs_attack[target_client_id]

            offline_state_from_arnoldi = _build_real_offline_state(
                server=server_train,
                target_subgraph=target_subgraph,
                target_client_id=target_client_id,
                smodel_type=smodel_type,
                data_type=data_type,
                structure_type=structure_type,
            )

            attack_state = AttackerState(
                target_client_id=target_client_id,
            )
            attack_state.n1 = int(target_subgraph.num_nodes)
            attack_state.n_total = global_node_count
            attack_state.train_node_indices = target_subgraph.train_mask.nonzero(as_tuple=False).flatten().cpu()

            resolved_offline_state = _merge_non_none(
                offline_state_from_arnoldi,
                {
                    **(offline_state or {}),
                    "q_v2": q_v2,
                    "a_v2_v1": a_v2_v1,
                    "h_r": h_r,
                    "v_r": v_r,
                    "sigma_r": sigma_r,
                    "tau_v2_history": tau_v2_history,
                },
            )

            attack_state = _collect_offline_state(server_train, attack_state, **resolved_offline_state)

            LOGGER.info("Starting online phase attack...")

            attack_res, inversion_res = run_online_phase_attack(
                server=server_attack,
                attack_state=attack_state,
                target_graph=target_subgraph,
                attack_epochs=effective_attack_epochs,
                global_graph=graph_attack,
                offline_state=resolved_offline_state,
                debug_fast=debug_fast,
                skip_training=skip_training,
            )

            seed_results[mode_name] = {
                "test_acc": float(train_res["Average"]["Test Acc"]),
                "attack": attack_res,
                "inversion": inversion_res,
                "dp": {
                    "enabled": bool(config.dp.enabled),
                    "clip_norm": float(config.dp.clip_norm),
                    "noise_multiplier": float(config.dp.noise_multiplier),
                    "delta": float(config.dp.delta),
                    "mode": str(config.dp.mode),
                },
            }

            LOGGER.info(f"Results:\n {seed_results[mode_name]} \n")

        results[str(seed_value)] = seed_results

    config.dp.enabled = original_dp["enabled"]
    config.dp.clip_norm = original_dp["clip_norm"]
    config.dp.noise_multiplier = original_dp["noise_multiplier"]
    config.dp.mode = original_dp["mode"]
    config.dp.separate_sfv = original_dp["separate_sfv"]

    file_name = os.path.join(
        experiment_path,
        f"privacy_attack_online_phase_{now}_{config.dataset.dataset_name}.json",
    )
    with open(file_name, "w") as f:
        json.dump(results, f, indent=2)

    LOGGER.info(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    run_privacy_attack_online_phase_experiment(debug_fast=False, skip_training=False)
