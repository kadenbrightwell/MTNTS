"""Training loop with mixed-precision, warmup, early stopping, and checkpointing."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import DEVICE, MODELS_DIR, TrainConfig


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return True
        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


class WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int, base_lr: float):
        self.optimizer = optimizer
        self.warmup = warmup_epochs
        self.total = total_epochs
        self.base_lr = base_lr
        self._step = 0

    def step(self):
        self._step += 1
        if self._step <= self.warmup:
            lr = self.base_lr * self._step / max(self.warmup, 1)
        else:
            progress = (self._step - self.warmup) / max(self.total - self.warmup, 1)
            lr = 1e-6 + 0.5 * (self.base_lr - 1e-6) * (1 + np.cos(np.pi * progress))
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


def train_model(
    model: nn.Module,
    train_ds,
    val_ds,
    cfg: TrainConfig | None = None,
    save_path: Path | None = None,
    quiet: bool = False,
) -> dict:
    """Full training loop. Returns dict of metrics history."""
    cfg = cfg or TrainConfig()
    save_path = save_path or (MODELS_DIR / "best_model.pt")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True,
        pin_memory=(DEVICE.type == "cuda"), num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size, shuffle=False,
        pin_memory=(DEVICE.type == "cuda"), num_workers=0,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = WarmupCosineScheduler(
        optimizer, cfg.warmup_epochs, cfg.epochs, cfg.learning_rate
    )
    criterion = nn.HuberLoss(delta=1.0)

    use_amp = cfg.use_amp and DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler(DEVICE.type, enabled=use_amp)
    early = EarlyStopping(patience=cfg.patience)

    history = {"train_loss": [], "val_loss": [], "lr": [], "epoch_time": []}

    if len(train_ds) == 0:
        raise ValueError(
            f"Training dataset has 0 windows (need > seq_len={train_ds.seq_len} samples). "
            "Fetch more data or reduce --seq-len."
        )

    if not quiet:
        print(f"\n[TRAIN] Starting training for up to {cfg.epochs} epochs")
        print(f"  Device: {DEVICE}  |  AMP: {use_amp}  |  Batch: {cfg.batch_size}")
        print(f"  Train windows: {len(train_ds)}  |  Val windows: {len(val_ds)}")
        print(f"  Warmup: {cfg.warmup_epochs} epochs  |  Patience: {cfg.patience}\n")

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()

        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(DEVICE, non_blocking=True)
            y_batch = y_batch.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(DEVICE.type, enabled=use_amp):
                preds = model(X_batch)
                loss = criterion(preds, y_batch)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(loss.item())

        scheduler.step()

        model.eval()
        val_losses = []
        if len(val_ds) > 0:
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch = X_batch.to(DEVICE, non_blocking=True)
                    y_batch = y_batch.to(DEVICE, non_blocking=True)
                    with torch.amp.autocast(DEVICE.type, enabled=use_amp):
                        preds = model(X_batch)
                        loss = criterion(preds, y_batch)
                    val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses) if val_losses else train_loss
        lr = scheduler.get_last_lr()[0]
        elapsed = time.time() - t0

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["lr"].append(lr)
        history["epoch_time"].append(elapsed)

        improved = early.step(val_loss)
        marker = " *" if improved else ""

        if not quiet and (epoch % 10 == 0 or epoch == 1 or improved):
            print(
                f"  Epoch {epoch:3d}/{cfg.epochs}  |  "
                f"train: {train_loss:.6f}  val: {val_loss:.6f}  |  "
                f"lr: {lr:.2e}  |  {elapsed:.1f}s{marker}"
            )

        if improved:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": float(val_loss),
            }, save_path)

        if early.should_stop:
            if not quiet:
                print(f"\n  Early stopping at epoch {epoch} (patience={cfg.patience})")
            break

    if not quiet:
        print(f"\n[TRAIN] Done. Best val loss: {early.best_loss:.6f}")
        print(f"  Model saved to {save_path}")
    return history
