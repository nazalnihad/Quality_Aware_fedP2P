import torch

def aggregate_models(peer_models, weights=None):
    """
    Aggregate model state dicts.
    weights: list of floats (e.g. data sizes). If None, plain average is used.
    """
    num_peers = len(peer_models)

    if weights is None:
        weights = [1.0] * num_peers

    total = sum(weights)
    norm_weights = [w / total for w in weights]

    aggregated_state_dict = {}
    for key in peer_models[0].keys():
        stacked = torch.stack([peer_models[i][key].float() for i in range(num_peers)], dim=0)
        weight_tensor = torch.tensor(norm_weights, dtype=torch.float32, device=stacked.device)
        # Reshape weight_tensor to broadcast over all parameter dimensions
        for _ in range(stacked.dim() - 1):
            weight_tensor = weight_tensor.unsqueeze(-1)
        weighted = (stacked * weight_tensor).sum(dim=0)
        aggregated_state_dict[key] = weighted.to(peer_models[0][key].dtype)
    return aggregated_state_dict