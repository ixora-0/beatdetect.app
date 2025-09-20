import torch
import torch.nn.functional as F


def masked_weighted_bce_logits(logits, targets, mask, pos_weight=(10.0, 40.0)):
    """
    logits:    (B, 2, T)
    targets:   (B, 2, T), {0,1}
    mask:      (B, T), bool
    pos_weight: float or (2,) tensor — one weight per channel
    """
    # BCEWithLogitsLoss per-element → (B, 2, T)
    if isinstance(pos_weight, (float, int)):
        pos_weight = torch.tensor([pos_weight, pos_weight], device=logits.device)
    else:
        pos_weight = torch.as_tensor(pos_weight, device=logits.device)
    bce = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        pos_weight=pos_weight.view(1, -1, 1),  # broadcast (1, 2, 1)
        reduction="none",
    )

    # Mask out padding
    masked_bce = bce * mask.unsqueeze(1).float()

    # Mean over valid frames
    return masked_bce.sum() / mask.float().sum()
