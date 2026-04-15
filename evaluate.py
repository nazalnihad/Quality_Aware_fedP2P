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

from config import DATA, NUM_PEERS, DEVICE, BATCH_SIZE
from model import create_model
from data import get_test_dataloader
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
    """Evaluate a single peer model — accuracy + visualization."""
    print(f"\n{'='*40}")
    print(f"  Peer {peer_id}")
    print(f"{'='*40}")

    model = load_model(peer_id)
    if model is None:
        return

    # Overall accuracy
    acc = evaluate_model(model, test_loader, DEVICE)
    print(f"  Test Accuracy: {acc:.4f} ({acc*100:.2f}%)")

    # Per-class accuracy
    class_names = CLASS_NAMES[DATA]
    class_correct = [0] * 10
    class_total = [0] * 10

    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1

    print(f"\n  Per-class accuracy:")
    for i in range(10):
        if class_total[i] > 0:
            class_acc = class_correct[i] / class_total[i]
            bar = "█" * int(class_acc * 20) + "░" * (20 - int(class_acc * 20))
            print(f"    {class_names[i]:>5}: {bar} {class_acc*100:.1f}%")

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
