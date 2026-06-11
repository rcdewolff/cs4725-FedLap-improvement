"""
DLG-style gradient inversion attack for FedLap online phase.

Reconstructs node features from shared gradients by minimizing
    ||∇θ(x_dummy) - ∇θ_observed||²
using second-order differentiation (create_graph=True).

Reference: Zhu et al., "Deep Leakage from Gradients", NeurIPS 2019.
"""

import torch
import torch.nn.functional as F
import numpy as np


def get_fmodel_target_grads(client_grads):
    """
    Extract the flat list of fmodel gradient tensors from a client's gradient dict.

    client_grads structure: {"fmodel": {"model": [grad, ...]}, "smodel": {...}}
    Returns a list of tensors (some may be None).
    """
    fmodel_grads = client_grads.get("fmodel", {})
    return fmodel_grads.get("model", [])


def gradient_inversion_attack(
    client,
    target_grads,
    n_iters=300,
    lr=0.1,
    tv_coef=1e-4,
):
    """
    Reconstruct node features from observed fmodel gradients (DLG attack).

    The attacker knows the model architecture and current weights (shared at
    the start of each FL round) and observes the gradient upload from one client.
    It optimizes dummy node features x_dummy by minimizing the gradient matching
    loss using second-order backprop.

    Args:
        client: GNNClient whose fmodel is the attack target.
        target_grads: flat list of gradient tensors from fmodel.model.parameters(),
                      as observed by the server (possibly DP-noised).
        n_iters: number of optimization iterations.
        lr: learning rate for x_dummy optimizer.
        tv_coef: total variation regularization weight (smoothness prior on features).

    Returns:
        x_reconstructed (Tensor): reconstructed node features [num_nodes, num_features].
        history (list of (iter, grad_loss)): convergence trace sampled every 50 iters.
    """
    fmodel = client.classifier.fmodel
    model = fmodel.model  # ModelBinder (GNN + MLP)
    graph = fmodel.graph

    edge_index = graph.edge_index
    train_mask = graph.train_mask
    y = graph.y

    num_nodes, num_features = graph.x.shape
    dev = graph.x.device

    # Initialize dummy features with small random noise
    x_dummy = torch.randn(num_nodes, num_features, requires_grad=True, device=dev)

    optimizer = torch.optim.Adam([x_dummy], lr=lr)

    # Parameters of the fmodel — order must match target_grads
    all_params = list(model.parameters())

    # Filter out None target gradients and their paired parameters
    paired = [
        (p, tg.detach())
        for p, tg in zip(all_params, target_grads)
        if tg is not None
    ]
    if not paired:
        return x_dummy.detach(), []

    params_valid, targets_valid = zip(*paired)

    model.eval()  # disable dropout for stable gradient computation

    history = []
    for it in range(n_iters):
        optimizer.zero_grad()

        # Forward pass with dummy features
        H_dummy = model(x_dummy, edge_index)
        loss_dummy = F.cross_entropy(H_dummy[train_mask], y[train_mask])

        # Compute gradients of the dummy loss w.r.t. model params (2nd-order)
        dummy_grads = torch.autograd.grad(
            loss_dummy,
            params_valid,
            create_graph=True,
            allow_unused=True,
        )

        # Gradient matching: sum of squared differences
        grad_loss = sum(
            ((dg - tg) ** 2).sum()
            for dg, tg in zip(dummy_grads, targets_valid)
            if dg is not None
        )

        # Total variation regularization: encourages smooth feature reconstructions
        if tv_coef > 0 and num_features > 1:
            tv = ((x_dummy[:, 1:] - x_dummy[:, :-1]) ** 2).sum()
            grad_loss = grad_loss + tv_coef * tv

        grad_loss.backward()
        optimizer.step()

        if it % 50 == 0:
            history.append((it, float(grad_loss.item())))

    model.train()
    return x_dummy.detach(), history


def compute_reconstruction_metrics(x_real, x_reconstructed):
    """
    Compute how well x_reconstructed matches x_real.

    Returns a dict with:
        mse           — mean squared error (lower = better reconstruction)
        cosine_sim    — global cosine similarity (higher = better reconstruction)
        feat_corr     — mean absolute per-feature Pearson correlation
    """
    x_r = x_real.detach().cpu().float()
    x_rec = x_reconstructed.detach().cpu().float()

    mse = float(F.mse_loss(x_rec, x_r).item())

    cos_sim = float(
        F.cosine_similarity(
            x_rec.flatten().unsqueeze(0),
            x_r.flatten().unsqueeze(0),
        ).item()
    )

    correlations = []
    for feat_idx in range(x_r.shape[1]):
        col_r = x_r[:, feat_idx].numpy()
        col_rec = x_rec[:, feat_idx].numpy()
        if col_r.std() > 1e-8 and col_rec.std() > 1e-8:
            corr = float(np.corrcoef(col_r, col_rec)[0, 1])
            correlations.append(abs(corr))
    feat_corr = float(np.mean(correlations)) if correlations else 0.0

    return {"mse": mse, "cosine_sim": cos_sim, "feat_corr": feat_corr}
