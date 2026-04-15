"""
compare.py — Compare all peer models after a simulation run.

Usage:
    python compare.py            # use stats JSONs only (fast, default)
    python compare.py --re-eval  # re-evaluate saved .pt models on test set
"""

import os
import json
import argparse
import torch

from config import DATA, NUM_PEERS, DEVICE, BATCH_SIZE, PEER_DATA_SIZES, TIER_BAND
from model import create_model
from data import get_test_dataloader
from trainer import evaluate_model


def load_stats(peer_id):
    path = f"outputs/peer_{peer_id}_stats.json"
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def re_evaluate(peer_id, test_loader):
    path = f"outputs/peer_{peer_id}_final_model.pt"
    if not os.path.exists(path):
        print(f"    [!] {path} not found, skipping.")
        return None
    try:
        model = create_model(DATA)
        # weights_only=True avoids PyTorch interactive warning/hang on newer versions
        state_dict = torch.load(path, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        print(f"    Evaluating on test set...", end=" ", flush=True)
        acc = evaluate_model(model, test_loader, DEVICE)
        print(f"{acc*100:.2f}%")
        return acc
    except Exception as e:
        print(f"    [!] Failed: {e}")
        return None


def expected_partners(peer_id):
    """Compute which peers this peer would aggregate with, given current config."""
    my_size = PEER_DATA_SIZES[peer_id]
    partners = []
    for i, s in enumerate(PEER_DATA_SIZES):
        if i == peer_id:
            continue
        ratio = abs(my_size - s) / max(my_size, s)
        if ratio <= TIER_BAND:
            partners.append(i)
    return partners


def print_comparison(rows):
    print()
    print("=" * 75)
    print("  PEER MODEL COMPARISON SUMMARY")
    print("=" * 75)
    header = (f"  {'Peer':>4}  {'Data Size':>10}  {'Partners':>18}"
              f"  {'Solo Rds':>8}  {'Accuracy':>10}")
    print(header)
    print("  " + "-" * 71)

    # Sort by data size descending (highest contributor first)
    rows_sorted = sorted(rows, key=lambda r: r["data_size"], reverse=True)

    for r in rows_sorted:
        partners_str = str(r["expected_partners"]) if r["expected_partners"] else "none (solo)"
        solo_str = str(r["solo_rounds"]) if r.get("solo_rounds") is not None else "?"
        acc_str = f"{r['accuracy']*100:.2f}%" if r["accuracy"] is not None else "N/A"
        print(f"  {r['peer_id']:>4}  {r['data_size']:>10}  {partners_str:>18}"
              f"  {solo_str:>8}  {acc_str:>10}")

    print("  " + "-" * 71)
    print()

    # Highlight the contribution-reward relationship
    valid = [r for r in rows_sorted if r["accuracy"] is not None]
    if len(valid) >= 2:
        best = valid[0]
        worst = valid[-1]
        gap = (best["accuracy"] - worst["accuracy"]) * 100
        print(f"  Accuracy gap (highest vs lowest contributor): {gap:+.2f}%")
        print(f"    Peer {best['peer_id']} ({best['data_size']} samples)  ->  {best['accuracy']*100:.2f}%")
        print(f"    Peer {worst['peer_id']} ({worst['data_size']} samples)  ->  {worst['accuracy']*100:.2f}%")
    elif len(valid) == 0:
        print("  No accuracy data found. Run the simulation first, or use --re-eval.")
    print()
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--re-eval", action="store_true",
                        help="Re-evaluate .pt model files on test set (slow on CPU)")
    args = parser.parse_args()

    test_loader = None
    if args.re_eval:
        print(f"Loading {DATA} test set for re-evaluation...")
        test_loader = get_test_dataloader(DATA, BATCH_SIZE)

    rows = []
    for pid in range(NUM_PEERS):
        stats = load_stats(pid)

        acc = None
        solo_rounds = None

        if stats is not None:
            # Prefer stats saved during training (accurate and fast)
            acc = stats.get("final_accuracy")
            solo_rounds = stats.get("solo_rounds")

        if args.re_eval and test_loader is not None:
            # Always re-evaluate if explicitly requested
            print(f"  Peer {pid} (data_size={PEER_DATA_SIZES[pid]}):")
            acc = re_evaluate(pid, test_loader)

        rows.append({
            "peer_id": pid,
            "data_size": PEER_DATA_SIZES[pid],
            "accuracy": acc,
            "solo_rounds": solo_rounds,
            "expected_partners": expected_partners(pid),
        })

    print_comparison(rows)
