"""
Evaluate saved peer models on test data.
Generates a grid of images with predictions for each peer model.

Usage:
    python evaluate.py                    # evaluate all peer models
    python evaluate.py --peer_id 0        # evaluate only peer 0's model
    python evaluate.py --num_images 50    # show 50 images (default: 25)
"""

import os
import torch
import argparse
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms

from config import DATA, NUM_PEERS, DEVICE, BATCH_SIZE, PEER_DATA_SIZES, SPLIT_MODE, ALPHA, PEER_CLASS_MAP
from model import create_model
from data import get_test_dataloader, get_peer_dataloader_sized
from trainer import evaluate_model

# MNIST class names
CLASS_NAMES = {
    "MNIST": [str(i) for i in range(10)],
    "CIFAR10": ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
}

# Denormalization to display images properly
DENORM = {
    "MNIST": transforms.Normalize((-0.1307 / 0.3081,), (1.0 / 0.3081,)),
    "CIFAR10": transforms.Normalize((-1.0,) * 3, (2.0,) * 3),
}


def load_model(peer_id):
    """Load a saved peer model from outputs/."""
    model_path = f"outputs/peer_{peer_id}_final_model.pt"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return None
    model = create_model(DATA)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model


def get_predictions(model, images):
    """Run model on a batch of images, return predicted classes and confidence."""
    with torch.no_grad():
        outputs = model(images.to(DEVICE))
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)
    return predicted.cpu(), confidence.cpu()


def visualize_predictions(peer_id, model, test_loader, num_images=25):
    """Create a grid of test images with predictions and save it."""
    class_names = CLASS_NAMES[DATA]
    denorm = DENORM[DATA]

    # Get a batch of test images
    images, labels = next(iter(test_loader))
    images = images[:num_images]
    labels = labels[:num_images]

    # Get predictions
    predicted, confidence = get_predictions(model, images)

    # Determine grid size
    cols = 5
    rows = (num_images + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 3))
    fig.suptitle(f"Peer {peer_id} — Predictions on Test Set", fontsize=16, fontweight='bold')

    for idx in range(rows * cols):
        ax = axes[idx // cols][idx % cols] if rows > 1 else axes[idx % cols]

        if idx < num_images:
            # Denormalize image for display
            img = denorm(images[idx])
            img = img.numpy()

            # Handle grayscale vs color
            if img.shape[0] == 1:
                ax.imshow(img.squeeze(), cmap='gray')
            else:
                ax.imshow(np.transpose(img, (1, 2, 0)))

            true_label = class_names[labels[idx]]
            pred_label = class_names[predicted[idx]]
            conf = confidence[idx].item() * 100

            correct = predicted[idx] == labels[idx]
            color = "green" if correct else "red"

            ax.set_title(f"True: {true_label}\nPred: {pred_label} ({conf:.0f}%)",
                         fontsize=9, color=color)
        ax.axis('off')

    plt.tight_layout()

    # Save
    os.makedirs("outputs", exist_ok=True)
    save_path = f"outputs/peer_{peer_id}_predictions.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved prediction grid: {save_path}")


def evaluate_peer(peer_id, test_loader):
    """Evaluate a single peer model — accuracy, confusion matrix, F1, confidence."""
    print(f"\n{'='*60}")
    print(f"  Peer {peer_id}")
    print(f"{'='*60}")

    model = load_model(peer_id)
    if model is None:
        return

    # Overall accuracy
    acc = evaluate_model(model, test_loader, DEVICE)
    print(f"  Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    class_names = CLASS_NAMES[DATA]
    num_classes = 10

    # ── Collect all predictions, labels, and confidences ──────────────────────
    all_preds = []
    all_labels = []
    all_confidences = []
    # Confusion matrix: confusion[true][pred] = count
    confusion = [[0] * num_classes for _ in range(num_classes)]

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, dim=1)

            # Top-3 predictions
            _, top3 = torch.topk(probabilities, k=min(3, num_classes), dim=1)

            for i in range(len(labels)):
                true = labels[i].item()
                pred = predicted[i].item()
                conf = confidence[i].item()
                confusion[true][pred] += 1
                all_preds.append(pred)
                all_labels.append(true)
                all_confidences.append(conf)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confidences = np.array(all_confidences)

    # ── Per-class accuracy with bar ───────────────────────────────────────────
    print(f"\n  Per-class accuracy:")
    for c in range(num_classes):
        total = sum(confusion[c])
        correct = confusion[c][c]
        if total > 0:
            class_acc = correct / total
            bar = "█" * int(class_acc * 20) + "░" * (20 - int(class_acc * 20))
            print(f"    {class_names[c]:>5}: {bar} {class_acc*100:.1f}%")

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    print(f"\n  Confusion Matrix (rows=true, cols=predicted):")
    col_w = max(5, max(len(n) for n in class_names) + 1)
    header = "  " + " " * col_w + "".join(f"{class_names[c]:>{col_w}}" for c in range(num_classes))
    print(header)
    print("  " + "-" * (col_w * (num_classes + 1)))
    for true_c in range(num_classes):
        row_str = f"  {class_names[true_c]:>{col_w}}"
        total = sum(confusion[true_c])
        for pred_c in range(num_classes):
            count = confusion[true_c][pred_c]
            if true_c == pred_c:
                row_str += f"  \033[92m{count:>{col_w-2}}\033[0m"  # green for diagonal
            elif count > 0:
                row_str += f"  \033[91m{count:>{col_w-2}}\033[0m"  # red for errors
            else:
                row_str += f"{count:>{col_w}}"
        print(row_str)

    # ── Precision, Recall, F1 per class ───────────────────────────────────────
    print(f"\n  Precision / Recall / F1 per class:")
    print(f"    {'Class':>6}  {'Precision':>9}  {'Recall':>9}  {'F1':>9}  {'Support':>8}")
    print(f"    {'-'*48}")
    macro_p, macro_r, macro_f1 = 0, 0, 0
    for c in range(num_classes):
        tp = confusion[c][c]
        # Precision: tp / (sum of column c)
        col_sum = sum(confusion[r][c] for r in range(num_classes))
        precision = tp / col_sum if col_sum > 0 else 0.0
        # Recall: tp / (sum of row c)
        row_sum = sum(confusion[c])
        recall = tp / row_sum if row_sum > 0 else 0.0
        # F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        macro_p += precision
        macro_r += recall
        macro_f1 += f1
        print(f"    {class_names[c]:>6}  {precision*100:>8.1f}%  {recall*100:>8.1f}%  {f1*100:>8.1f}%  {row_sum:>8}")

    macro_p /= num_classes
    macro_r /= num_classes
    macro_f1 /= num_classes
    print(f"    {'-'*48}")
    print(f"    {'macro':>6}  {macro_p*100:>8.1f}%  {macro_r*100:>8.1f}%  {macro_f1*100:>8.1f}%")

    # ── Top-3 Accuracy ────────────────────────────────────────────────────────
    # Recompute top-3 in a second pass (lightweight)
    top3_correct = 0
    total_samples = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, top3 = torch.topk(outputs, k=min(3, num_classes), dim=1)
            for i in range(len(labels)):
                if labels[i] in top3[i]:
                    top3_correct += 1
                total_samples += 1
    top3_acc = top3_correct / total_samples if total_samples > 0 else 0
    print(f"\n  Top-1 Accuracy: {acc*100:.2f}%  |  Top-3 Accuracy: {top3_acc*100:.2f}%")

    # ── Confidence Analysis ───────────────────────────────────────────────────
    correct_mask = all_preds == all_labels
    wrong_mask = ~correct_mask
    avg_conf_correct = all_confidences[correct_mask].mean() * 100 if correct_mask.any() else 0
    avg_conf_wrong = all_confidences[wrong_mask].mean() * 100 if wrong_mask.any() else 0
    print(f"\n  Confidence Analysis:")
    print(f"    Avg confidence on CORRECT predictions: {avg_conf_correct:.1f}%")
    print(f"    Avg confidence on WRONG   predictions: {avg_conf_wrong:.1f}%")
    print(f"    Confidence gap: {avg_conf_correct - avg_conf_wrong:.1f}%  "
          f"({'good — model knows when it\'s unsure' if avg_conf_correct - avg_conf_wrong > 15 else 'low — model is overconfident on errors'})")

    # ── Most Confused Pairs ───────────────────────────────────────────────────
    pairs = []
    for true_c in range(num_classes):
        for pred_c in range(num_classes):
            if true_c != pred_c and confusion[true_c][pred_c] > 0:
                pairs.append((confusion[true_c][pred_c], true_c, pred_c))
    pairs.sort(reverse=True)
    if pairs:
        print(f"\n  Most Confused Pairs (top 5):")
        print(f"    {'True':>8}  →  {'Predicted':<10}  {'Count':>6}")
        print(f"    {'-'*35}")
        for count, true_c, pred_c in pairs[:5]:
            print(f"    {class_names[true_c]:>8}  →  {class_names[pred_c]:<10}  {count:>6}")

    # ── Local Accuracy ────────────────────────────────────────────────────────
    _, local_val_loader = get_peer_dataloader_sized(
        DATA, peer_id, PEER_DATA_SIZES[peer_id], BATCH_SIZE,
        split_mode=SPLIT_MODE, alpha=ALPHA, class_map=PEER_CLASS_MAP
    )
    local_acc = evaluate_model(model, local_val_loader, DEVICE)
    print(f"\n  Local Validation Accuracy: {local_acc*100:.2f}%")

    # ── Classes Learned ───────────────────────────────────────────────────────
    # Count classes where the peer achieves >50% accuracy on the global test set
    classes_trained_on = PEER_CLASS_MAP.get(peer_id, list(range(10))) if SPLIT_MODE == "manual_skew" else list(range(10))
    classes_above_50 = []
    for c in range(num_classes):
        total = sum(confusion[c])
        correct = confusion[c][c]
        if total > 0 and correct / total > 0.5:
            classes_above_50.append(c)
    trained_str = ", ".join(class_names[c] for c in classes_trained_on)
    learned_str = ", ".join(class_names[c] for c in classes_above_50)
    print(f"\n  Classes Learned (>50% acc): {len(classes_above_50)}/10  [{learned_str}]")
    print(f"  Classes Trained On:         {len(classes_trained_on)}/10  [{trained_str}]")
    extra_learned = [c for c in classes_above_50 if c not in classes_trained_on]
    if extra_learned:
        extra_str = ", ".join(class_names[c] for c in extra_learned)
        print(f"  Classes Learned via Federation: {extra_str}")

    # Visualize predictions
    visualize_predictions(peer_id, model, test_loader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--peer_id', type=int, default=None, help="Evaluate specific peer (default: all)")
    parser.add_argument('--num_images', type=int, default=25, help="Number of images to visualize")
    args = parser.parse_args()

    test_loader = get_test_dataloader(DATA, BATCH_SIZE)

    if args.peer_id is not None:
        evaluate_peer(args.peer_id, test_loader)
    else:
        print("\n" + "=" * 50)
        print("  Evaluating ALL peer models")
        print("=" * 50)
        for pid in range(NUM_PEERS):
            evaluate_peer(pid, test_loader)

    print(f"\n✓ Done! Check outputs/ folder for prediction images.")
