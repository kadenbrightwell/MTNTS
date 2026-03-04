"""Inference wrapper: loads trained model(s) and runs prediction on GPU.

Supports both single-model and multi-seed ensemble prediction.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn

from config import DEVICE, MODELS_DIR, ModelConfig
from src.model.architecture import build_model


class Predictor:
    """Loads one or more checkpoints and averages their predictions."""

    def __init__(
        self,
        n_features: int,
        model_path: Path | None = None,
        cfg: ModelConfig | None = None,
    ):
        self.cfg = cfg or ModelConfig()
        self.models: List[nn.Module] = []

        paths = self._resolve_paths(model_path)
        for p in paths:
            m = build_model(n_features, self.cfg)
            ckpt = torch.load(p, map_location=DEVICE, weights_only=False)
            m.load_state_dict(ckpt["model_state_dict"])
            m.eval()
            self.models.append(m)
            epoch = ckpt.get("epoch", "?")
            vloss = ckpt.get("val_loss", 0)
            print(f"[PREDICTOR] Loaded {p.name} (epoch {epoch}, val_loss {vloss:.6f})")

        print(f"[PREDICTOR] Ensemble size: {len(self.models)}")

    def _resolve_paths(self, model_path: Path | None) -> List[Path]:
        """Find all ensemble seed files, or fall back to single model."""
        base = model_path or (MODELS_DIR / "best_model.pt")
        if base.exists():
            stem, suffix = base.stem, base.suffix
            parent = base.parent
            ensemble = sorted(parent.glob(f"{stem}_seed*{suffix}"))
            if ensemble:
                return ensemble
            return [base]

        stem, suffix = base.stem, base.suffix
        parent = base.parent
        ensemble = sorted(parent.glob(f"{stem}_seed*{suffix}"))
        if ensemble:
            return ensemble

        raise FileNotFoundError(
            f"No model found at {base} or {parent}/{stem}_seed*{suffix}. "
            "Train a model first: python scripts/train.py"
        )

    @torch.no_grad()
    def predict(self, window: np.ndarray) -> float:
        """Predict return for a single feature window (ensemble-averaged)."""
        return self.predict_detailed(window)["mean"]

    @torch.no_grad()
    def predict_detailed(self, window: np.ndarray) -> dict:
        """Return per-model predictions with ensemble statistics.

        Returns dict with keys: mean, std, individual, n_models,
        agreement (fraction of models agreeing on direction), confidence.
        """
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(DEVICE)
        individual = []
        for m in self.models:
            with torch.amp.autocast(DEVICE.type, enabled=(DEVICE.type == "cuda")):
                individual.append(m(x).item())

        arr = np.array(individual)
        mean_pred = float(arr.mean())
        std_pred = float(arr.std())
        direction = np.sign(mean_pred)
        agree_count = np.sum(np.sign(arr) == direction) if direction != 0 else len(arr)
        agreement = float(agree_count / len(arr))
        dispersion = 1.0 / (1.0 + std_pred / max(abs(mean_pred), 1e-6))
        confidence = agreement * 0.6 + dispersion * 0.4

        return {
            "mean": mean_pred,
            "std": std_pred,
            "individual": individual,
            "n_models": len(individual),
            "agreement": agreement,
            "confidence": confidence,
        }

    @torch.no_grad()
    def predict_batch(self, windows: np.ndarray) -> np.ndarray:
        """Predict returns for a batch of windows (ensemble-averaged)."""
        x = torch.tensor(windows, dtype=torch.float32).to(DEVICE)
        all_preds = []
        for m in self.models:
            with torch.amp.autocast(DEVICE.type, enabled=(DEVICE.type == "cuda")):
                all_preds.append(m(x).cpu().numpy())
        return np.mean(all_preds, axis=0)
