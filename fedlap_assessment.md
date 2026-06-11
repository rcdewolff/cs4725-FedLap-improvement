# FedLap repository assessment

## Scope
This note summarizes how the repository is structured and where the FedLap method is implemented. It focuses on the federated learning flow, the structure-feature pipeline, and the main places an online-phase privacy layer would hook in.

## Repository layout
- config: YAML configs for datasets and model settings used by the runs.
- datasets: cached datasets and PyG processed data.
- results: experiment outputs.
- src: all implementations (training loops, models, utilities, simulations).

Key entry points and utilities:
- src/main.py: primary run that loads a graph, partitions it, instantiates servers, and runs local/FL baselines.
- src/simulations/simulation.py: batch experiment driver for repeated runs and hyper-parameter sweeps.
- src/utils/utils.py: global config load via CONFIGPATH, logging, metrics, aggregation helpers, and plotting.
- src/utils/config_parser.py: YAML schema for dataset/model/structure configuration.

## Configuration and runtime wiring
- Configuration is loaded from CONFIGPATH (environment variable) and parsed into a Config object.
- Result paths are assembled from RESULTPATH plus dataset and model settings, and logging uses that path.
- The global config and logging objects are initialized in src/utils/utils.py and imported throughout via src/__init__.py.

## Federated learning architecture (base pattern)
- Client and server base classes are in src/client.py and src/server.py.
- Server.joint_train_g implements gradient aggregation:
  - optionally shares server weights with clients at the start of a round,
  - clients train locally,
  - the server aggregates client gradients with sum_lod weighted by client node counts,
  - shared gradients are pushed back to clients and the server.
- Server.joint_train_w implements weight averaging:
  - shares weights, trains clients,
  - aggregates only structural feature gradients (just_SFV=True),
  - averages model weights after local updates.
- Aggregation uses sum_lod in src/utils/utils.py and uses client node counts as coefficients.

## FedLap implementation (where it lives)
FedLap is not a separate module with a single entry point; it is integrated into the GNN server/client and model stack.

Core pieces:
- src/GNN/GNN_server.py:
  - builds the shared structural features (SFV) at the server and passes them to clients in initialize/initialize_FL.
  - computes Laplacian/eigendecomposition artifacts for spectral variants.
- src/GNN/GNN_client.py:
  - constructs the classifier based on the config (feature, structure, or f+s).
  - for smodel_type == "Laplace", uses FedLaplaceClassifier in f+s mode and SLaplace in structure-only mode.
- src/GNN/GNN_classifier.py:
  - FedLaplaceClassifier combines a feature model (FGNN/DGCN) and a Laplacian structure model (SLaplace).
- src/GNN/laplace.py:
  - SLaplace and SpectralLaplace define the Laplacian regularizer used in the loss.
- src/classifier.py:
  - train_step includes intrinsic_regularizer (from the structure model) scaled by config.spectral.regularizer_coef.

## Structure features and the "hop2vec" path
- Structural features are computed in src/utils/graph.py via Graph.add_structural_features.
- For structure_type == "hop2vec", the SFV is initialized as random trainable features (requires_grad=True).
- These SFVs are treated as model parameters in SClassifier (src/GNN/sGNN.py) and therefore participate in gradient exchange.

This means the FedLap online phase exchanges gradients on both model weights and (optionally) SFV parameters, depending on the training mode.

## Data partitioning for clients
- Partitioning logic is in src/utils/graph_partitioning.py (louvain, random, kmeans, metis) and used in src/main.py.
- Each partition yields a subgraph with intra-edges and optional inter-edges for some methods.

## Privacy analysis scripts
The folder src/FedLap/privacy_analysis contains scripts for theoretical/privacy analysis and plotting, but they are not wired into the training loops. These appear to be analysis utilities rather than part of the online FL phase.

## Online-phase privacy hook points
If you want to add privacy enhancements during the online FL phase, the natural integration points are:
- Gradient exchange in Server.get_grads / Server.share_grads (src/server.py).
- Aggregation in Server.joint_train_g and Server.joint_train_w (src/server.py) and sum_lod (src/utils/utils.py).
- Server-to-client sharing of SFV, Laplacian eigendecomposition, and abars in GNNServer.initialize (src/GNN/GNN_server.py).
- If protecting SFVs specifically, consider SClassifier.get_grads / set_grads (src/GNN/sGNN.py) and Graph.initialize_random_features (src/utils/graph.py).

These locations are the most direct places to add clipping, noise, secure aggregation, or feature obfuscation without changing the overall architecture.

## Online-phase privacy measures status and plan

### Differential privacy
Status: Not implemented in the online FL loop. The only privacy-related code is in the analysis scripts under [src/FedLap/privacy_analysis/privacy_main.py](src/FedLap/privacy_analysis/privacy_main.py) and [src/FedLap/privacy_analysis/prvacy_noise.py](src/FedLap/privacy_analysis/prvacy_noise.py), which are not called from training.

Plan:
- Add DP configuration fields to the config schema in [src/utils/config_parser.py](src/utils/config_parser.py) and the dataset YAMLs (enable flag, clip norm, noise multiplier, delta).
- Clip and noise client updates just before aggregation in [src/server.py](src/server.py) (inside joint_train_g and joint_train_w) or at the point of client gradient creation in [src/client.py](src/client.py).
- Ensure both model gradients and SFV gradients are covered when structure_type is hop2vec by touching the gradient paths in [src/GNN/sGNN.py](src/GNN/sGNN.py) and [src/GNN/GNN_classifier.py](src/GNN/GNN_classifier.py).
- Optionally add a simple epsilon/delta accounting log in [src/utils/utils.py](src/utils/utils.py) alongside existing logging.

### Homomorphic encryption
Status: Not implemented. There are no encryption or HE library hooks in the training loop or utilities.

Plan:
- Introduce a crypto abstraction for tensors and plug it into the gradient exchange path in [src/server.py](src/server.py) and [src/client.py](src/client.py) so clients send encrypted gradients and the server aggregates in the encrypted domain.
- Keep the integration limited to the online phase: encrypt in client.get_grads, aggregate in server, decrypt only when updates are applied.
- Because the current setup is single-process and object-based (not networked), treat HE as a simulated integration unless you add an actual client-server transport layer.

### Secure aggregation
Status: Not implemented. Aggregation is a direct weighted sum via sum_lod in [src/utils/utils.py](src/utils/utils.py) inside [src/server.py](src/server.py).

Plan:
- Implement mask-based secure aggregation on client gradients in [src/client.py](src/client.py) (mask before sending) and unmasking/combining logic in [src/server.py](src/server.py).
- Apply the same masking to SFV gradients for hop2vec by covering the SFV path in [src/GNN/sGNN.py](src/GNN/sGNN.py).
- Add a minimal threshold or dropout-handling logic if you want robustness to missing clients in a round.

## Privacy experiment implementation plan (CiteSeer only)

Goal: Compare FedLap baseline vs FedLap with added privacy measures on the CiteSeer dataset only, using identical partitions, seeds, and hyper-parameters.

Plan:
- Add a dedicated CiteSeer config that fixes dataset_name, partitioning, and model choices in config/config_CiteSeer.yml, and reference it via CONFIGPATH.
- Implement a single experiment runner that toggles privacy modes while keeping all other settings fixed in [src/main.py](src/main.py) or a new script under src/simulations/.
- Log per-run metadata (seed, partitioning, num_subgraphs, privacy mode, DP params) alongside accuracy metrics using the existing logger in [src/utils/utils.py](src/utils/utils.py).
- Use the same client partitions across modes by precomputing subgraphs once in [src/utils/graph_partitioning.py](src/utils/graph_partitioning.py) and reusing them for each privacy variant within the run.
- Evaluate only two conditions: FedLap baseline and FedLap + privacy measure (one at a time). If multiple measures are tested, run separate pairs for each measure to keep comparisons clean.
- Report for each pair: test accuracy, convergence curves, and privacy metric outputs (attack success or DP epsilon/delta if DP is used).

Suggested run matrix for CiteSeer:
- Partitioning: one fixed method (e.g., louvain) and one fixed num_subgraphs.
- Seeds: at least 3 seeds for mean and std.
- Privacy modes: baseline vs one privacy enhancement per experiment.
