# Quality-Aware P2P Federated Learning

A decentralized, peer-to-peer Federated Learning framework built with PyTorch, designed to address challenges with small-data peers and Non-IID (Independent and Identically Distributed) data distributions. This project introduces a Quality-Aware Aggregation approach, moving beyond simple quantity-based averaging to ensure high-quality model convergence even with highly skewed data among peers.

## Project Motivation

Traditional federated learning often weights peer contributions strictly by the quantity of data they hold. In environments with heavily skewed or Non-IID data distributions, this can lead to performance bottlenecks, model poisoning, or global model degradation, particularly punishing smaller peers who may have high-quality, specialized data.

This project implements and compares alternative aggregation strategies to demonstrate that quality-aware weighting and model-reward mechanisms significantly improve performance for both large and small-data peers.

## Architecture and Key Mechanisms

The system relies on a fully decentralized topology without a central parameter server. The key components include:

- Universal Pull-Based Reputation Gate: Evaluates incoming peer models dynamically. Every peer maintains a local holdout validation set. When receiving a model from a neighbor, the peer evaluates it locally; only models that prove beneficial are accepted for aggregation.
- Reward Push Mechanism: Actively rewards high-performing peers within the network. This mechanism distributes reputation-based rewards back to the network, promoting quality data contributions.
- Log-Dampened Weighting and Mini-Aggregation: Rather than weighting purely by data volume, weights are dynamically adjusted to handle Non-IID distributions and mitigate model degradation from low-quality data.

## Aggregation Modes

The framework supports runtime comparisons between three core aggregation strategies:

1. Quantity-Only (Baseline): A standard Federated Averaging approach where incoming models are weighted strictly by the size of the peer's dataset.
2. Reputation-Gated: Introduces a threshold-based gate where peers only accept and aggregate models from neighbors whose models improve local holdout accuracy above a calculated threshold.
3. Quality-Aware: A hybrid approach that combines the reputation gate with dynamic, quality-based weight scaling. It ensures that peers providing high-quality model updates receive higher representation in the final aggregation, regardless of their raw data quantity.

## Empirical Results

Simulations running with data skew (where peer data sizes range from 5000 down to 714 samples) show clear improvements in the Quality-Aware mode over the Baseline Quantity mode.

- Quantity Mode: Global accuracy often plateaus at a lower threshold (e.g., ~94.35%), as larger but potentially less diverse or noisy datasets dominate the aggregation.
- Quality and Reputation Modes: Global accuracy sees significant gains (e.g., ~98.59% to 98.91%). The quality-aware mechanism ensures that peers with smaller datasets but highly relevant features are still trusted and utilized by the network, dramatically improving overall convergence.

## Running the Simulation

Ensure you have your virtual environment activated and the necessary dependencies installed. 

### Launch the Simulation

To spin up the peer network, train the models, and perform federated aggregation across all nodes, simply run:

```bash
python run_simulation.py
```

This script will automatically:
- Launch multiple background peer processes.
- Coordinate decentralized training rounds.
- Save resulting output metrics and final compiled models into the outputs directory.
- Generate a final comparison table upon completion using the compare script.

### Evaluate Results

If you want to manually run the comparison tool against the latest simulation run, execute:

```bash
python compare.py
```

## Repository Structure

- run_simulation.py: Entry point for launching the complete P2P FL network.
- peer.py and peer_server.py: Core logic for individual peer lifecycle, local training, and peer-to-peer communication over the network.
- aggregator.py: Mathematical logic for weighting and aggregating model state dictionaries.
- model.py and data.py: Neural network architectures and PyTorch dataset/dataloader definitions. This includes logic for manual data skewing and Non-IID partitioning.
- config.py: Centralized configuration file defining the number of peers, communication rounds, dataset paths, and hyperparameters.
- compare.py and evaluate.py: Evaluation scripts designed to parse the simulation outputs and generate comparative performance metrics.

---
Developed for research into robust decentralized learning environments.
