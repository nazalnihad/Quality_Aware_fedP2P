import torch

NUM_PEERS = 7
SEED = 42

# ── Aggregation mode ────────────────────────────────────────────────────────
# "quantity"   — plain FedAvg, weight = data_size only (no reputation)
# "reputation" — weight = data_size × reputation, raw model replacement scoring
# "quality"    — weight = log(data_size) × reputation, mini-aggregation scoring,
#                reward push to high-rep small peers
AGG_MODE = "quality"

# ── Data distribution mode ──────────────────────────────────────────────────
# "iid"          — stratified balanced sampling (each peer gets all classes equally)
# "non_iid"      — Dirichlet distribution (controlled by ALPHA)
# "manual_skew"  — hardcoded class assignments via PEER_CLASS_MAP
SPLIT_MODE = "manual_skew"

ALPHA = 0.1             # Dirichlet parameter (only used when SPLIT_MODE="non_iid")
                        # higher = more IID, lower = more non-IID

# Manual class assignments per peer (only used when SPLIT_MODE="manual_skew")
# Key = peer_id, Value = list of class labels the peer trains on
PEER_CLASS_MAP = {
    0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],  # rich peer — all 10 classes
    1: [0, 1, 2, 3, 4, 5, 6, 7],          # 8 classes
    2: [0, 1, 2, 3, 4, 5],                # 6 classes
    3: [0, 1, 2, 3],                      # 4 classes
    4: [8, 9],                            # rare specialist
    5: [6, 7],                            # rare specialist
    6: [9],                               # ultra specialist
}

# Data sizes per peer — linearly decreasing: Peer 0 gets the most, last peer the least.
def _make_data_sizes(num_peers, total_samples):
    weights = list(range(num_peers, 0, -1))   # [N, N-1, ..., 1]
    total_weight = sum(weights)
    sizes = [round(w / total_weight * total_samples) for w in weights]
    # Fix rounding drift: adjust largest peer so sum == total_samples
    sizes[0] += total_samples - sum(sizes)
    return sizes

# Similarity band: a peer is eligible if its data_size is within TIER_BAND of mine
TIER_BAND = 0.40
# Reputation gate for downward aggregation:
#   REP_THRESHOLD  – min reputation a smaller peer must have before a richer
#                    peer accepts its model (0.0 = accept all, 1.0 = never)
#   REP_EMA_ALPHA  – weight of new quality score vs. old reputation (EMA)
REP_THRESHOLD  = 0.5   # tune: lower = more inclusive, higher = stricter
REP_EMA_ALPHA  = 0.2    # 0.2 → reputation changes slowly over ~5 rounds
BASE_PORT = 5000

NUM_ROUNDS = 20
LOCAL_EPOCHS = 3
LEARNING_RATE = 0.005
BATCH_SIZE = 32

DATA = "MNIST"
TRAIN_SAMPLES = 20000

# Auto-generated: Peer 0 gets the most data, last peer the least
PEER_DATA_SIZES = _make_data_sizes(NUM_PEERS, TRAIN_SAMPLES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")