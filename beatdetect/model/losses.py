import torch
import torch.nn.functional as F


def masked_weighted_bce_logits(logits, targets, mask, pos_weight=(10.0, 40.0)):
    """
    logits:   (B, 2, T)
    targets:  (B, 2, T)
    mask:     (B, T)
    pos_weight: float or (2,) for beat and downbeat channels
    """

    if isinstance(pos_weight, (float, float)):
        pos_weight = torch.tensor([pos_weight, pos_weight], device=logits.device)
    else:
        pos_weight = torch.as_tensor(pos_weight, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight.view(1, -1, 1),  # broadcast (1, 2, 1)
        reduction="none",
    )

    # Build weight mask: (B, 2, T)
    # Reduce beat weight where downbeat_target > 0
    weight = torch.ones_like(targets)
    weight[:, 0, :] = 1.0 - targets[:, 1, :]

    # Apply mask
    masked_bce = bce * weight * mask.unsqueeze(1).float()

    return masked_bce.sum() / mask.float().sum()
