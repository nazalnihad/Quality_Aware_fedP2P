import argparse
import time
import random
import os
import torch
import requests

from config import *
# REP_THRESHOLD and REP_EMA_ALPHA are imported via *
from model import create_model
from data import get_peer_dataloader_sized, get_test_dataloader
from trainer import train_model, evaluate_model
from topology import get_neighbours, get_peer_address
from peer_server import start_server, model_buffer, buffer_lock
import peer_server
from utils import serialize_model, deserialize_model
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

# Reputation scores for neighbours: float in [0, 1]
# New peers start at 0 so they must earn downward admission over rounds.
reputation = {}  # {peer_id: float}

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
    Tiered asymmetric eligibility:
      1. Symmetric band  – peers within TIER_BAND of my data size aggregate
                           bidirectionally (unchanged behaviour).
      2. Downward gate   – peers SMALLER than me (outside the band) are
                           admitted only if their reputation >= REP_THRESHOLD.
                           They send their model up; we do NOT send ours down
                           in the same wait-group, avoiding deadlock.
      3. Upward fallback – if NO band partners exist at all, find the
                           closest richer peer(s) and pull from them
                           one-way (we receive; they don't wait for us).
                           Prevents permanently isolated solo-only peers.
    """
    eligible = []
    reachable = []  # (n_id, n_size, ratio) for all reachable neighbours

    for n_id in all_neighbours:
        n_size = query_data_size(n_id)
        if n_size is None:
            log(f"  Could not reach Peer {n_id}, skipping")
            continue
        ratio = abs(my_data_size - n_size) / max(my_data_size, n_size)
        reachable.append((n_id, n_size, ratio))
        if ratio <= TIER_BAND:
            # Symmetric band partner — same as before
            eligible.append((n_id, n_size, "band"))
        elif n_size < my_data_size:
            # Smaller peer outside band — admit only if reputation is proven
            rep = reputation.get(n_id, 0.0)
            if rep >= REP_THRESHOLD:
                eligible.append((n_id, n_size, "downward"))
                log(f"  Peer {n_id} admitted via reputation ({rep:.2f} >= {REP_THRESHOLD})")
            else:
                log(f"  Peer {n_id} excluded — below threshold "
                    f"(rep={rep:.2f}, need {REP_THRESHOLD}, data_size={n_size})")
        else:
            log(f"  Peer {n_id} excluded — larger peer outside band "
                f"(ratio={ratio:.2f} > {TIER_BAND})")

    # ── Upward fallback: no band ─────────────────────────────────────────────
    # If we ended up with zero band partners, pick the closest richer peer(s)
    # as one-way upward sources so we learn *something* each round.
    has_band = any(kind == "band" for _, _, kind in eligible)
    if not has_band:
        richer = [(n_id, n_size, ratio)
                  for n_id, n_size, ratio in reachable
                  if n_size > my_data_size]
        if richer:
            # Sort by closeness (smallest ratio first) and take the nearest one
            richer.sort(key=lambda x: x[2])
            closest_ratio = richer[0][2]
            # Admit all peers tied at the closest ratio (typically just 1)
            for n_id, n_size, ratio in richer:
                if abs(ratio - closest_ratio) < 1e-6:
                    eligible.append((n_id, n_size, "upward"))
                    log(f"  Peer {n_id} admitted via UPWARD FALLBACK "
                        f"(no band; ratio={ratio:.2f}, size={n_size})")
        else:
            log(f"  No band AND no richer peer reachable — truly solo this round")

    return eligible


def compute_rep_weight(n_id, n_size):
    """Aggregation weight = data_size × reputation (floored at 0.1 to avoid zero)."""
    rep = max(0.1, reputation.get(n_id, 1.0))
    return n_size * rep


def update_reputation(n_id, incoming_state, before_state, val_loader):
    """
    Evaluate model before and after applying neighbour's weights on our
    local validation set. Quality = accuracy gain (clipped to [0, 1]).
    Updates reputation[n_id] via EMA.
    """
    model.load_state_dict(before_state)
    acc_before = evaluate_model(model, val_loader, DEVICE)

    model.load_state_dict(incoming_state)
    acc_after = evaluate_model(model, val_loader, DEVICE)

    # Normalise gain to [0, 1]: gain of 0 → quality 0.5 (neutral)
    quality = 0.5 + (acc_after - acc_before)
    quality = max(0.0, min(1.0, quality))

    old_rep = reputation.get(n_id, 0.0)
    reputation[n_id] = (1 - REP_EMA_ALPHA) * old_rep + REP_EMA_ALPHA * quality
    log(f"  Reputation Peer {n_id}: {old_rep:.3f} → {reputation[n_id]:.3f} "
        f"(quality={quality:.3f}, acc {acc_before:.3f}→{acc_after:.3f})")

def pull_model_from(n_id, round_num):
    """Pull a smaller peer's latest model via GET /get_model, waiting until it's ready."""
    url = get_peer_address(n_id, BASE_PORT) + "/get_model"
    while True:
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("round") >= round_num:  # peer has trained this round
                    return deserialize_model(data["state_dict"])
        except Exception:
            pass
        time.sleep(1)



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
downward_admits = 0      # total downward peer slots across all rounds
upward_fallback_rounds = 0  # rounds where we had no band but used upward fallback
acc = 0.0

for round_num in range(NUM_ROUNDS):
    round_start = time.time()
    log(f"--- Round {round_num+1}/{NUM_ROUNDS} ---")
    current_lr = LEARNING_RATE * (0.99 ** round_num)

    # Step 1: Local training
    log(f"  Training locally ({LOCAL_EPOCHS} epochs)...")
    local_weights = train_model(model, train_loader, current_lr, LOCAL_EPOCHS, DEVICE)

    # Publish our latest model so richer peers can pull it
    peer_server.MY_LATEST_MODEL = serialize_model(local_weights)
    peer_server.MY_LATEST_ROUND = round_num

    # Step 2: Determine eligible neighbours
    eligible = get_eligible_neighbours(all_neighbours)
    # Band partners require bidirectional exchange;
    # downward/upward peers are pull-only (no send from us).
    band_ids     = [n_id for n_id, _, kind in eligible if kind == "band"]
    downward_ids = [n_id for n_id, _, kind in eligible if kind == "downward"]
    upward_ids   = [n_id for n_id, _, kind in eligible if kind == "upward"]
    pull_ids     = downward_ids + upward_ids   # all one-way pulls
    eligible_ids = band_ids + pull_ids
    log(f"  Band: {band_ids} | Downward (rep-gated): {downward_ids} | "
        f"Upward fallback: {upward_ids}")

    if eligible_ids:
        total_partners += len(eligible_ids)
        if upward_ids and not band_ids:
            upward_fallback_rounds += 1

        # Step 3: Send to band partners only (symmetric).
        # Downward/upward peers are pull-only — we don't push to them.
        if band_ids:
            log(f"  Sending model to band partners {band_ids}...")
            send_to_neighbours(local_weights, round_num, band_ids)

        # Step 4a: Wait for band partners to push their models
        if band_ids:
            log(f"  Waiting for band models from {band_ids}...")
            wait_for_models(round_num, band_ids)

        # Step 4b: Pull models from downward + upward peers directly
        pull_models = []
        for n_id in pull_ids:
            kind_label = "downward" if n_id in downward_ids else "upward-fallback"
            log(f"  Pulling model from {kind_label} peer {n_id}...")
            pm = pull_model_from(n_id, round_num)
            pull_models.append(pm)
        log(f"  All neighbour models received!")

        # Step 5: Reputation-weighted FedAvg
        band_models      = get_models(round_num, band_ids)
        neighbour_models = band_models + pull_models
        rep_weights = [compute_rep_weight(n_id, n_size)
                       for n_id, n_size, _ in eligible]
        all_models      = [local_weights] + neighbour_models
        all_weights_rep = [my_data_size]  + rep_weights
        weight_info = ", ".join(
            f"P{n_id}:{round(w,1)}(rep={reputation.get(n_id,1.0):.2f})"
            for (n_id, _, _), w in zip(eligible, rep_weights)
        )
        log(f"  Aggregating {len(all_models)} models | my_weight={my_data_size} | {weight_info}")
        avg_weights = aggregate_models(all_models, weights=all_weights_rep)

        # Step 5b: Update reputation for each neighbour based on their contribution
        log(f"  Updating reputations...")
        for i, (n_id, _, _) in enumerate(eligible):
            update_reputation(n_id, neighbour_models[i], local_weights, test_loader)

        model.load_state_dict(avg_weights)
        cleanup_buffer(round_num, band_ids)  # only band pushed into buffer
        downward_admits += len(downward_ids)
    else:
        solo_rounds += 1
        # No eligible neighbours — train solo this round
        log(f"  No eligible neighbours — solo round (keeping local weights)")
        model.load_state_dict(local_weights)

    # Step 6: Evaluate
    acc = evaluate_model(model, test_loader, DEVICE)
    round_time = time.time() - round_start

    # Reputation snapshot for all known peers
    rep_summary = "  ".join(
        f"P{pid}:{score:.3f}" for pid, score in sorted(reputation.items())
    ) or "none yet"

    log(f"  ✓ Round {round_num+1}/{NUM_ROUNDS} done | "
        f"Acc: {acc:.4f} | Time: {round_time:.1f}s | "
        f"Band: {band_ids} | Downward: {downward_ids} | Upward: {upward_ids}")
    log(f"  Reputations — {rep_summary}")

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
    "downward_admits": downward_admits,
    "upward_fallback_rounds": upward_fallback_rounds,
    "final_reputations": {str(k): round(v, 4) for k, v in sorted(reputation.items())},
}
stats_path = f"outputs/peer_{peer_id}_stats.json"
with open(stats_path, "w") as f:
    json.dump(stats, f, indent=2)

log(f"=== Training complete! Final model saved to {model_path} ===")
log(f"=== [SUMMARY] Peer {peer_id} | Data: {my_data_size} samples | "
    f"Avg partners/round: {total_partners/NUM_ROUNDS:.1f} | "
    f"Solo rounds: {solo_rounds}/{NUM_ROUNDS} | Final accuracy: {acc:.4f} ===")