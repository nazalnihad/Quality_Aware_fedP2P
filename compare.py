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

from config import DATA, NUM_PEERS, DEVICE, BATCH_SIZE, PEER_DATA_SIZES, TIER_BAND, REP_THRESHOLD
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


def expected_band_partners(peer_id):
    """Symmetric band partners only (static config estimate)."""
    my_size = PEER_DATA_SIZES[peer_id]
    return [i for i, s in enumerate(PEER_DATA_SIZES)
            if i != peer_id and abs(my_size - s) / max(my_size, s) <= TIER_BAND]


def print_comparison(rows):
    print()
    print("=" * 85)
    print("  PEER MODEL COMPARISON SUMMARY")
    print("=" * 85)

    # ── Main table ────────────────────────────────────────────────────────────
    header = (f"  {'Peer':>4}  {'Data':>7}  {'Band Partners':>15}"
              f"  {'Solo':>5}  {'↓Admits':>8}  {'↑Fallback':>10}  {'Global Acc':>11}  {'Local Acc':>10}")
    print(header)
    print("  " + "-" * 105)

    rows_sorted = sorted(rows, key=lambda r: r["data_size"], reverse=True)
    for r in rows_sorted:
        band_str     = str(r["band_partners"])           if r["band_partners"]                        else "none"
        solo_str     = str(r["solo_rounds"])             if r["solo_rounds"]   is not None            else "?"
        admits_str   = str(r["downward_admits"])         if r.get("downward_admits") is not None      else "?"
        fallback_str = str(r["upward_fallback_rounds"])  if r.get("upward_fallback_rounds") is not None else "?"
        acc_str      = f"{r['accuracy']*100:.2f}%"       if r["accuracy"] is not None                 else "N/A"
        local_acc_str = f"{r['local_accuracy']*100:.2f}%" if r.get("local_accuracy") is not None      else "N/A"
        
        print(f"  {r['peer_id']:>4}  {r['data_size']:>7}  {band_str:>15}"
              f"  {solo_str:>5}  {admits_str:>8}  {fallback_str:>10}  {acc_str:>11}  {local_acc_str:>10}")

    print("  " + "-" * 105)

    # ── Accuracy gap ──────────────────────────────────────────────────────────
    valid = [r for r in rows_sorted if r["accuracy"] is not None]
    if len(valid) >= 2:
        best, worst = valid[0], valid[-1]
        gap = (best["accuracy"] - worst["accuracy"]) * 100
        print(f"\n  Accuracy gap (highest vs lowest data): {gap:+.2f}%")
        print(f"    Peer {best['peer_id']}  ({best['data_size']:>6} samples)  →  {best['accuracy']*100:.2f}%")
        print(f"    Peer {worst['peer_id']}  ({worst['data_size']:>6} samples)  →  {worst['accuracy']*100:.2f}%")

    # ── Reputation matrix ─────────────────────────────────────────────────────
    rep_rows = [r for r in rows_sorted if r.get("final_reputations")]
    if rep_rows:
        print()
        print(f"  FINAL REPUTATION SCORES  (threshold = {REP_THRESHOLD})")
        print("  " + "-" * 81)

        all_pids = sorted({int(k) for r in rep_rows for k in r["final_reputations"]})
        col_w = 9
        header_rep = f"  {'Rater \\ Ratee':>14}" + "".join(f"  {('P'+str(p)):>{col_w}}" for p in all_pids)
        print(header_rep)
        print("  " + "-" * 81)

        for r in rep_rows:
            reps = r["final_reputations"]
            row_str = f"  {'Peer '+str(r['peer_id']):>14}"
            for p in all_pids:
                score = reps.get(str(p))
                if score is None:
                    cell = "-"
                elif score >= REP_THRESHOLD:
                    cell = f"{score:.3f}✓"
                else:
                    cell = f"{score:.3f}"
                row_str += f"  {cell:>{col_w}}"
            print(row_str)

        print("  " + "-" * 81)
        print("  ✓ = above threshold (peer was admitted as downward/upward partner)")

        # ── Quality vs quantity insight ───────────────────────────────────────
        print()
        print("  QUALITY vs QUANTITY")
        print("  " + "-" * 81)
        print(f"  {'Peer':>4}  {'Data':>7}  {'Accuracy':>10}  {'Avg rep given':>14}  {'Peers trusted':>14}")
        print("  " + "-" * 81)
        for r in rows_sorted:
            reps = r.get("final_reputations", {})
            if not reps:
                continue
            avg_rep  = sum(reps.values()) / len(reps)
            trusted  = sum(1 for v in reps.values() if v >= REP_THRESHOLD)
            acc_str  = f"{r['accuracy']*100:.2f}%" if r["accuracy"] is not None else "N/A"
            print(f"  {r['peer_id']:>4}  {r['data_size']:>7}  {acc_str:>10}"
                  f"  {avg_rep:>14.3f}  {trusted:>6}/{len(reps)}")

    print()
    print("=" * 85)


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

        acc                   = stats.get("final_accuracy")         if stats else None
        local_acc             = stats.get("final_local_accuracy")   if stats else None
        solo_rounds           = stats.get("solo_rounds")            if stats else None
        downward_admits       = stats.get("downward_admits")        if stats else None
        upward_fallback_rounds = stats.get("upward_fallback_rounds") if stats else None
        final_reputations     = stats.get("final_reputations")      if stats else {}

        if args.re_eval and test_loader is not None:
            print(f"  Peer {pid} (data_size={PEER_DATA_SIZES[pid]}):")
            acc = re_evaluate(pid, test_loader)
            local_acc = None # Re-evaluating local acc is complex, so we just use None

        rows.append({
            "peer_id":               pid,
            "data_size":             PEER_DATA_SIZES[pid],
            "accuracy":              acc,
            "local_accuracy":        local_acc,
            "solo_rounds":           solo_rounds,
            "downward_admits":       downward_admits,
            "upward_fallback_rounds": upward_fallback_rounds,
            "final_reputations":     final_reputations,
            "band_partners":         expected_band_partners(pid),
        })

    print_comparison(rows)
