import torch
import torch.nn.functional as F


def masked_weighted_ce(logits, targets, mask, weight=(14.4, 38.2, 1.1)):
    """
    logits:       (B, 3, T)
    targets:      (B, 3, T)
    mask:         (B, T)
    """
    if type(weight) is tuple:
        weight = torch.as_tensor(weight, device=logits.device)
    bce = F.cross_entropy(logits, targets, weight, reduction="none")

    # Apply mask
    masked_bce = bce * mask.unsqueeze(1).float()
    return masked_bce.sum() / mask.float().sum()
