import torch
import torch.nn.functional as F


def collate_fn(batch):
    """
    Custom collate function to handle batches with variable-length sequences.
    Each item in 'batch' is a tuple: (mel, flux, target)
    target has shape (2, T): [beat_row, downbeat_row].
    """
    mels, fluxes, targets = zip(*batch, strict=False)  # Unzip the batch

    # Determine the maximum sequence length in the batch
    max_len = max(mel.shape[1] for mel in mels)

    # Pad mels → (B, N_MELS, T_max)
    padded_mels = [F.pad(mel, (0, max_len - mel.shape[1])) for mel in mels]
    padded_mels = torch.stack(padded_mels)

    # Pad fluxes → (B, T_max)
    padded_fluxes = [F.pad(flux, (0, max_len - flux.shape[0])) for flux in fluxes]
    padded_fluxes = torch.stack(padded_fluxes)

    # Pad targets → (B, 2, T_max)
    padded_targets = [
        F.pad(target, (0, max_len - target.shape[1])) for target in targets
    ]
    padded_targets = torch.stack(padded_targets)

    # Build mask = True where not padded → (B, T_max)
    masks = torch.stack(
        [
            torch.cat(
                [
                    torch.ones(t.shape[1], dtype=torch.bool),  # length T_i
                    torch.zeros(max_len - t.shape[1], dtype=torch.bool),
                ]
            )
            for t in targets
        ],
        dim=0,
    )

    return padded_mels, padded_fluxes, padded_targets, masks
