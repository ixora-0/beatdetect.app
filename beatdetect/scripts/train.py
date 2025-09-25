from datetime import datetime

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from ..config_loader import Config, load_config
from ..data import BeatDataset, collate_fn
from ..model import BeatDetectTCN, create_mock_inputs, masked_weighted_bce_logits
from ..utils import set_seed


def main(config: Config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- TensorBoard setup ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = config.paths.models / f"tensorboard_logs/run_{timestamp}"
    writer = SummaryWriter(log_dir=str(log_dir))

    # Log hyperparameters
    hparams = {
        "learning_rate": config.hypers.learning_rate,
        "batch_size": config.training.batch_size,
        "random_seed": config.random_seed,
    }
    writer.add_hparams(hparams, {})

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

    # Log model graph
    mock_mels, mock_fluxes = create_mock_inputs(config=config, device=device)
    writer.add_graph(model, (mock_mels, mock_fluxes))

    best_val_loss = float("inf")
    epochs_no_improve = 0
    train_losses, val_losses = [], []

    def run_epoch(loader, training: bool, epoch: int):
        if training:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        for batch_idx, (_id, mels, fluxes, targets, masks) in enumerate(
            tqdm(loader, desc="Train" if training else "Val", unit="batch")
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
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                # Log batch-level training loss
                step = (epoch - 1) * len(loader) + batch_idx
                writer.add_scalar("Loss/Train_Batch", loss.item(), step)

            total_loss += loss.item() * mels.size(0)

        return total_loss / len(loader.dataset)

    # --- training loop ---
    num_epochs = 200
    early_stop_patience = 20

    for epoch in trange(1, num_epochs + 1, desc="Epochs"):
        train_loss = run_epoch(train_loader, training=True, epoch=epoch)
        val_loss = run_epoch(val_loader, training=False, epoch=epoch)

        # Log epoch-level metrics
        writer.add_scalars("Loss", {"Train": train_loss, "Validation": val_loss}, epoch)
        writer.add_scalar("Loss/Train_Val_Diff", train_loss - val_loss, epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], epoch)

        scheduler.step(val_loss)
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config.paths.models / f"model{timestamp}.pt")
            tqdm.write(
                f"✅ Epoch {epoch}: Improved val_loss to {val_loss:.4f}, model saved."
            )
            epochs_no_improve = 0

            # Log best validation loss
            writer.add_scalar("Loss/Best_Validation", best_val_loss, epoch)
        else:
            epochs_no_improve += 1
            tqdm.write(
                f"⚠️ Epoch {epoch}: No improvement for {epochs_no_improve} epochs."
            )

        writer.add_scalar("Training/Epochs_No_Improve", epochs_no_improve, epoch)

        if epochs_no_improve >= early_stop_patience:
            tqdm.write(
                f"🛑 Early stopping at epoch {epoch}. Best val_loss {best_val_loss:.4f}"
            )
            break

    writer.add_hparams(
        {
            "learning_rate": config.hypers.learning_rate,
            "dropout": config.hypers.dropout,
            "kernel_size": config.hypers.kernel_size,
            "channels": str(config.hypers.channels),
            "dilations": str(config.hypers.dilations),
            "batch_size": config.training.batch_size,
            "random_seed": config.random_seed,
        },
        {
            "final_train_loss": train_losses[-1],
            "best_val_loss": best_val_loss,
            "epochs_trained": epoch,
            "early_stopped": epochs_no_improve >= early_stop_patience,
        },
    )

    writer.close()
    print(f"TensorBoard logs saved to: {log_dir}")


if __name__ == "__main__":
    config = load_config()
    main(config)
