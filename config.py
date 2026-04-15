import torch 

NUM_PEERS = 7
# Data sizes per peer — dynamically distributed based on NUM_PEERS and TRAIN_SAMPLES.
# Uses linearly decreasing weights: Peer 0 gets the most, last peer gets the least.
# e.g. NUM_PEERS=5, TRAIN_SAMPLES=20000 → [6667, 5333, 4000, 2667, 1333]
def _make_data_sizes(num_peers, total_samples):
    weights = list(range(num_peers, 0, -1))   # [N, N-1, ..., 1]
    total_weight = sum(weights)
    sizes = [round(w / total_weight * total_samples) for w in weights]
    # Fix rounding drift: adjust largest peer so sum == total_samples
    sizes[0] += total_samples - sum(sizes)
    return sizes

# Similarity band: a peer is eligible if its data_size >= my_data_size * (1 - TIER_BAND)
# Set to 1.0 to disable the lower bound (only upper bound: data_size <= my_data_size)
TIER_BAND = 0.40
BASE_PORT = 5000

NUM_ROUNDS = 100
LOCAL_EPOCHS = 3
LEARNING_RATE = 0.005
BATCH_SIZE = 32

DATA = "CIFAR10"  # "MNIST" or "CIFAR10"
TRAIN_SAMPLES = 20000
IID = False
ALPHA = 0.5             # Dirichlet parameter (only used when IID=False)
                        # higher = more IID, lower = more non-IID
                        # try: 0.1 (extreme), 0.5 (moderate), 1.0 (mild)

# Auto-generated: Peer 0 gets the most data, last peer the least
PEER_DATA_SIZES = _make_data_sizes(NUM_PEERS, TRAIN_SAMPLES)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")