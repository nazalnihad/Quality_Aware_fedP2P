# Quality-Aware P2P Federated Learning

A decentralized, peer-to-peer Federated Learning framework built with PyTorch, designed to address challenges with small-data peers and Non-IID data distributions. This project introduces a **Quality-Aware Aggregation** approach, moving beyond simple quantity-based averaging to ensure high-quality model convergence even with highly skewed data.

## 🧠 Architecture Overview

Our architecture relies on a fully decentralized topology without a central parameter server. The key components include:

- **Universal Pull-Based Reputation Gate**: Evaluates incoming peer models dynamically. Only models that prove beneficial based on local holdout validation are accepted for aggregation.
- **Reward Push Mechanism**: Actively rewards high-performing peers within the network, promoting quality data contributions.
- **Quality-Aware Aggregation**: Rather than weighting purely by data volume, weights are adjusted using **Log-Dampened Weighting** and **Mini-Aggregation** to handle Non-IID distributions and mitigate model poisoning or degradation from low-quality data.
- **Multi-Mode Aggregation**: Supports runtime comparisons between three core strategies:
  1. Quantity-Only (Baseline)
  2. Reputation-Gated 
  3. Quality-Aware

## 🚀 Running the Simulation

Ensure you have your virtual environment activated and dependencies installed. 

### Launch the Simulation
To spin up the peer network, train the models, and perform federated aggregation across all nodes, simply run:

```bash
python run_simulation.py
```

This script will automatically:
- Launch multiple background peer processes (`peer.py`).
- Coordinate decentralized training rounds.
- Save resulting output metrics and models into the `outputs/` directory.
- Generate a final comparison table upon completion using `compare.py`.

### Evaluate Results
If you want to manually run the comparison tool against the latest simulation run:

```bash
python compare.py
```

## 📁 Repository Structure

- `run_simulation.py`: Entry point for launching the P2P FL network.
- `peer.py` & `peer_server.py`: Core logic for individual peer lifecycle, training, and P2P communication.
- `aggregator.py`: Logic for weighting and aggregating model state dicts.
- `model.py` & `data.py`: Neural network architectures and PyTorch dataset/dataloader definitions (includes manual data skewing).
- `config.py`: Centralized configuration (number of peers, rounds, dataset paths, etc.).
- `compare.py` & `evaluate.py`: Scripts to parse `outputs/` and generate comparative performance metrics.

---
*Developed for research into robust decentralized learning environments.*
