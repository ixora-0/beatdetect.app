import torch
from tqdm import tqdm, trange

from ..config_loader import Config, load_config
from ..data import BeatDataset, collate_fn
from ..model import BeatDetectTCN, masked_weighted_bce_logits
from ..utils import set_seed


def main(config: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- datasets ---
    set_seed(config.random_seed)
    train_dataset = BeatDataset(config, split="train")
    val_dataset = BeatDataset(config, split="val")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # --- model ---
    model = BeatDetectTCN(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.hypers.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=1 / 5, patience=5
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    def run_epoch(loader, training: bool):
        if training:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        for mels, fluxes, targets, masks in tqdm(
            loader, desc="Train" if training else "Val", unit="batch"
        ):
            mels, fluxes, targets, masks = (
                mels.to(device),
                fluxes.to(device),
                targets.to(device),
                masks.to(device),
            )

            if training:
                optimizer.zero_grad()

            logits = model(mels, fluxes, return_logits=True)  # (B, 2, T)
            # ensure mask shape matches
            if masks.dim() == 2:  # (B, T)
                masks_expanded = masks.unsqueeze(1).expand_as(logits)
            else:
                masks_expanded = masks

            loss = masked_weighted_bce_logits(logits, targets, masks_expanded)

            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item() * mels.size(0)

        return total_loss / len(loader.dataset)

    # --- training loop ---
    num_epochs = 200
    early_stop_patience = 20

    for epoch in trange(1, num_epochs + 1, desc="Epochs"):
        train_loss = run_epoch(train_loader, training=True)
        val_loss = run_epoch(val_loader, training=False)
        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.paths.models / "model.pt")
            tqdm.write(
                f"✅ Epoch {epoch}: Improved val_loss to {val_loss:.4f}, model saved."
            )
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            tqdm.write(
                f"⚠️ Epoch {epoch}: No improvement for {epochs_no_improve} epochs."
            )

        if epochs_no_improve >= early_stop_patience:
            tqdm.write(
                f"🛑 Early stopping at epoch {epoch}. Best val_loss {best_val_loss:.4f}"
            )
            break

    # --- save losses for inspection ---
    with open("losses.txt", "w") as f:
        print(train_losses, file=f)
        print(val_losses, file=f)


if __name__ == "__main__":
    config = load_config()
    main(config)
