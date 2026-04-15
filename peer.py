import argparse
import time
import random
import os
import torch
import requests

from config import *
from model import create_model
from data import get_peer_dataloader_sized, get_test_dataloader
from trainer import train_model, evaluate_model
from topology import get_neighbours, get_peer_address
from peer_server import start_server, model_buffer, buffer_lock
import peer_server
from utils import serialize_model
from aggregator import aggregate_models

parser = argparse.ArgumentParser()
parser.add_argument('--peer_id', type=int, required=True)
args = parser.parse_args()
peer_id = args.peer_id

def log(msg):
    print(f"[Peer {peer_id}] {msg}", flush=True)

log("Starting...")
random.seed(42 + peer_id)

# My data contribution size
my_data_size = PEER_DATA_SIZES[peer_id]

# Start Flask server
my_port = BASE_PORT + peer_id
log(f"Starting server on port {my_port}")
start_server(my_port)
time.sleep(5)

# Advertise data size via peer_server module variable
peer_server.MY_DATA_SIZE = my_data_size
log(f"Server ready | Data size: {my_data_size} samples")

# Create model and dataloaders
log(f"Loading {DATA} dataset ({my_data_size} samples for this peer)")
model = create_model(DATA)
train_loader = get_peer_dataloader_sized(DATA, peer_id, my_data_size, BATCH_SIZE,
                                         iid=IID, alpha=ALPHA)
test_loader = get_test_dataloader(DATA, BATCH_SIZE)
log(f"Data loaded. Training batches: {len(train_loader)}, Device: {DEVICE}")


# ── Neighbour helpers ────────────────────────────────────────────────────────

def query_data_size(neighbour_id):
    """Ask a neighbour for its data size via /peer_info. Returns None on failure."""
    url = get_peer_address(neighbour_id, BASE_PORT) + "/peer_info"
    try:
        resp = requests.get(url, timeout=3)
        return resp.json().get("data_size", 0)
    except Exception:
        return None


def get_eligible_neighbours(all_neighbours):
    """
    Symmetric similarity check: two peers aggregate together if
        |my_size - n_size| / max(my_size, n_size)  <=  TIER_BAND

    This is symmetric — if A considers B eligible, B also considers A eligible,
    so both send AND wait for each other. No deadlock.
    """
    eligible = []
    for n_id in all_neighbours:
        n_size = query_data_size(n_id)
        if n_size is None:
            log(f"  Could not reach Peer {n_id}, skipping")
            continue
        ratio = abs(my_data_size - n_size) / max(my_data_size, n_size)
        if ratio <= TIER_BAND:
            eligible.append((n_id, n_size))
        else:
            log(f"  Peer {n_id} excluded (data_size={n_size}, ratio={ratio:.2f} > {TIER_BAND})")
    return eligible


def send_to_neighbours(state_dict, round_num, targets):
    for n_id in targets:
        url = get_peer_address(n_id, BASE_PORT) + "/receive_model"
        payload = {
            "peer_id": peer_id,
            "round": round_num,
            "state_dict": serialize_model(state_dict)
        }
        try:
            requests.post(url, json=payload, timeout=5)
        except requests.exceptions.RequestException as e:
            log(f"  Failed to send to Peer {n_id}: {e}")


def wait_for_models(round_num, neighbours):
    while True:
        with buffer_lock:
            got_all = all((round_num, n_id) in model_buffer for n_id in neighbours)
        if got_all:
            break
        time.sleep(1)


def get_models(round_num, neighbours):
    with buffer_lock:
        return [model_buffer[(round_num, n_id)] for n_id in neighbours]


def cleanup_buffer(round_num, neighbours):
    with buffer_lock:
        for n_id in neighbours:
            model_buffer.pop((round_num, n_id), None)


# ── Main topology ────────────────────────────────────────────────────────────

all_neighbours = get_neighbours(peer_id, NUM_PEERS, "fully_connected")
log(f"All neighbours: {all_neighbours}")
log(f"=== Starting {NUM_ROUNDS} rounds of federated learning ===")

total_partners = 0
solo_rounds = 0
acc = 0.0

for round_num in range(NUM_ROUNDS):
    round_start = time.time()
    log(f"--- Round {round_num+1}/{NUM_ROUNDS} ---")
    current_lr = LEARNING_RATE * (0.99 ** round_num)

    # Step 1: Local training
    log(f"  Training locally ({LOCAL_EPOCHS} epochs)...")
    local_weights = train_model(model, train_loader, current_lr, LOCAL_EPOCHS, DEVICE)

    # Step 2: Determine eligible neighbours based on data size
    eligible = get_eligible_neighbours(all_neighbours)
    eligible_ids = [n_id for n_id, _ in eligible]
    eligible_sizes = [size for _, size in eligible]
    log(f"  Eligible neighbours: {eligible_ids} (of {all_neighbours})")

    if eligible_ids:
        total_partners += len(eligible_ids)

        # Step 3: Send model only to eligible neighbours
        log(f"  Sending model to {len(eligible_ids)} eligible neighbours...")
        send_to_neighbours(local_weights, round_num, eligible_ids)

        # Step 4: Wait for their models
        log(f"  Waiting for neighbour models...")
        wait_for_models(round_num, eligible_ids)
        log(f"  All neighbour models received!")

        # Step 5: Weighted FedAvg
        neighbour_models = get_models(round_num, eligible_ids)
        all_models = [local_weights] + neighbour_models
        all_weights = [my_data_size] + eligible_sizes
        log(f"  Aggregating {len(all_models)} models (Weighted FedAvg, weights={all_weights})...")
        avg_weights = aggregate_models(all_models, weights=all_weights)
        model.load_state_dict(avg_weights)

        cleanup_buffer(round_num, eligible_ids)
    else:
        solo_rounds += 1
        # No eligible neighbours — train solo this round
        log(f"  No eligible neighbours — solo round (keeping local weights)")
        model.load_state_dict(local_weights)

    # Step 6: Evaluate
    acc = evaluate_model(model, test_loader, DEVICE)
    round_time = time.time() - round_start
    log(f"  ✓ Round {round_num+1} done | Eligible peers: {len(eligible_ids)} | "
        f"Accuracy: {acc:.4f} | Time: {round_time:.1f}s")

# Save final model
import json
os.makedirs("outputs", exist_ok=True)
model_path = f"outputs/peer_{peer_id}_final_model.pt"
torch.save(model.state_dict(), model_path)

# Save stats for comparison table
stats = {
    "peer_id": peer_id,
    "data_size": my_data_size,
    "final_accuracy": round(acc, 4),
    "avg_partners_per_round": round(total_partners / NUM_ROUNDS, 2),
    "solo_rounds": solo_rounds,
}
stats_path = f"outputs/peer_{peer_id}_stats.json"
with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2)

log(f"=== Training complete! Final model saved to {model_path} ===")
log(f"=== [SUMMARY] Peer {peer_id} | Data: {my_data_size} samples | "
    f"Avg partners/round: {total_partners/NUM_ROUNDS:.1f} | "
    f"Solo rounds: {solo_rounds}/{NUM_ROUNDS} | Final accuracy: {acc:.4f} ===")