import random
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

TRANSFORMS = {
    "MNIST": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]),
    "CIFAR10": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]),
}

DATASET_CLASS = {
    "MNIST": datasets.MNIST,
    "CIFAR10": datasets.CIFAR10,
}


# ---- IID: Stratified balanced sampling ----

def stratified_sample(full_dataset, num_samples):
    targets = full_dataset.targets  
    num_classes = 10
    samples_per_class = num_samples // num_classes 

    all_indices = []
    for class_id in range(num_classes):
        class_indices = [i for i, label in enumerate(targets) if label == class_id]
        chosen = random.sample(class_indices, samples_per_class)
        all_indices.extend(chosen)

    random.shuffle(all_indices)
    return all_indices


def iid_split(full_dataset, num_samples, num_peers):
    """IID: equal balanced samples per peer."""
    all_indices = stratified_sample(full_dataset, num_samples)
    samples_per_peer = num_samples // num_peers
    peer_indices = {}
    for peer_id in range(num_peers):
        start = peer_id * samples_per_peer
        end = start + samples_per_peer
        peer_indices[peer_id] = all_indices[start:end]
    return peer_indices


# ---- Non-IID: Dirichlet distribution ----

def dirichlet_split(full_dataset, num_samples, num_peers, alpha):
    """
    Non-IID split using Dirichlet distribution.
    
    For each class, draw a distribution over peers from Dir(alpha).
    Small alpha = extreme non-IID (each peer gets few classes).
    Large alpha = mild non-IID (close to IID).
    """
    targets = np.array(full_dataset.targets)
    num_classes = 10
    samples_per_class = num_samples // num_classes

    # First, subsample: pick samples_per_class from each class
    class_indices = {}
    for c in range(num_classes):
        all_c = np.where(targets == c)[0].tolist()
        class_indices[c] = random.sample(all_c, samples_per_class)

    # For each class, split its indices among peers using Dirichlet
    peer_indices = {peer_id: [] for peer_id in range(num_peers)}

    for c in range(num_classes):
        indices = np.array(class_indices[c])
        np.random.shuffle(indices)

        # Draw proportions from Dirichlet: how to split this class among peers
        proportions = np.random.dirichlet([alpha] * num_peers)

        # Convert proportions to actual counts
        counts = (proportions * len(indices)).astype(int)
        # Fix rounding: give leftover to random peer
        diff = len(indices) - counts.sum()
        counts[np.random.randint(num_peers)] += diff

        # Assign indices to peers
        start = 0
        for peer_id in range(num_peers):
            end = start + counts[peer_id]
            peer_indices[peer_id].extend(indices[start:end].tolist())
            start = end

    # Shuffle each peer's data
    for peer_id in range(num_peers):
        random.shuffle(peer_indices[peer_id])

    return peer_indices


def print_distribution(peer_indices, full_dataset, num_peers):
    """Print how many of each class each peer got."""
    targets = np.array(full_dataset.targets)
    print("\n  Data distribution across peers:")
    print(f"  {'Peer':<6}", end="")
    for c in range(10):
        print(f"  {c:>4}", end="")
    print(f"  {'Total':>6}")
    print("  " + "-" * 60)
    for peer_id in range(num_peers):
        print(f"  {peer_id:<6}", end="")
        for c in range(10):
            count = sum(1 for i in peer_indices[peer_id] if targets[i] == c)
            print(f"  {count:>4}", end="")
        print(f"  {len(peer_indices[peer_id]):>6}")
    print()


# ---- Main entry point ----

def get_peer_dataloader(dataset_name, peer_id, num_peers, num_samples, batch_size,
                        iid=True, alpha=0.5):
    full_dataset = DATASET_CLASS[dataset_name](
        root='data', train=True, download=True,
        transform=TRANSFORMS[dataset_name]
    )

    if iid:
        peer_indices = iid_split(full_dataset, num_samples, num_peers)
    else:
        peer_indices = dirichlet_split(full_dataset, num_samples, num_peers, alpha)

    # Print distribution (only peer 0 prints to avoid spam)
    if peer_id == 0:
        mode = "IID" if iid else f"Non-IID (Dirichlet α={alpha})"
        print(f"  Data mode: {mode}")
        print_distribution(peer_indices, full_dataset, num_peers)

    peer_dataset = Subset(full_dataset, peer_indices[peer_id])
    return DataLoader(peer_dataset, batch_size=batch_size, shuffle=True)


def get_peer_dataloader_sized(dataset_name, peer_id, data_size, batch_size,
                               split_mode="iid", alpha=0.5, class_map=None):
    """
    Build train and local-validation loaders for a single peer.

    split_mode: "iid"          — stratified balanced sampling
                "non_iid"      — Dirichlet distribution (controlled by alpha)
                "manual_skew"  — only classes listed in class_map[peer_id]

    Returns (train_loader, local_val_loader) with an 80/20 split.
    """
    full_dataset = DATASET_CLASS[dataset_name](
        root='data', train=True, download=True,
        transform=TRANSFORMS[dataset_name]
    )

    # Use a deterministic per-peer seed so subsets don't overlap
    # We use a global SEED from config so all peers use a synchronized starting point
    from config import SEED
    rng = random.Random(SEED + peer_id)
    np_rng = np.random.default_rng(SEED + peer_id)

    targets = np.array(full_dataset.targets)
    num_classes = 10

    if split_mode == "manual_skew":
        # ── Manual skew: peer only gets the classes defined in class_map ──
        if class_map is None or peer_id not in class_map:
            raise ValueError(f"manual_skew requires class_map entry for peer {peer_id}")
        allowed_classes = class_map[peer_id]
        samples_per_class = data_size // len(allowed_classes)
        indices = []
        for c in allowed_classes:
            class_idx = np.where(targets == c)[0].tolist()
            n = min(samples_per_class, len(class_idx))
            chosen = rng.sample(class_idx, n)
            indices.extend(chosen)
        # Fix rounding so total == data_size
        deficit = data_size - len(indices)
        if deficit > 0:
            c = allowed_classes[0]
            class_idx = np.where(targets == c)[0].tolist()
            already = set(indices)
            remaining = [i for i in class_idx if i not in already]
            indices.extend(rng.sample(remaining, min(deficit, len(remaining))))
        rng.shuffle(indices)

    elif split_mode == "iid":
        # ── IID: stratified balanced sampling ──
        samples_per_class = data_size // num_classes
        indices = []
        for c in range(num_classes):
            class_idx = np.where(targets == c)[0].tolist()
            chosen = rng.sample(class_idx, min(samples_per_class, len(class_idx)))
            indices.extend(chosen)
        rng.shuffle(indices)

    else:  # non_iid
        # ── Non-IID: Dirichlet over classes ──
        proportions = np_rng.dirichlet([alpha] * num_classes)
        per_class_counts = (proportions * data_size).astype(int)
        per_class_counts[per_class_counts.argmax()] += data_size - per_class_counts.sum()
        indices = []
        for c in range(num_classes):
            class_idx = np.where(targets == c)[0].tolist()
            n = min(int(per_class_counts[c]), len(class_idx))
            indices.extend(rng.sample(class_idx, n))
        rng.shuffle(indices)

    # ── 80/20 train / local-validation split ──
    split_point = int(len(indices) * 0.8)
    train_indices = indices[:split_point]
    val_indices = indices[split_point:]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def get_test_dataloader(dataset_name, batch_size):
    test_dataset = DATASET_CLASS[dataset_name](
        root='data', train=False, download=True,
        transform=TRANSFORMS[dataset_name]
    )
    return DataLoader(test_dataset, batch_size=batch_size, shuffle=False)