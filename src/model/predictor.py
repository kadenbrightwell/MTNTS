"""Inference wrapper: loads trained model(s) and runs prediction on GPU.

Supports single-ticker and multi-seed ensemble prediction, plus a
MultiPredictor that holds one ensemble per target ticker.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict

import numpy as np
import torch
import torch.nn as nn

from config import DEVICE, MODELS_DIR, ModelConfig, TARGET_TICKERS, ticker_model_dir
from src.model.architecture import build_model, detect_checkpoint_config


class Predictor:
    """Loads one or more checkpoints for a single ticker and averages predictions."""

    def __init__(
        self,
        n_features: int,
        ticker: str | None = None,
        model_path: Path | None = None,
        cfg: ModelConfig | None = None,
    ):
        self.cfg = cfg or ModelConfig()
        self.ticker = ticker
        self.models: List[nn.Module] = []

        paths = self._resolve_paths(model_path, ticker)
        for i, p in enumerate(paths):
            ckpt = torch.load(p, map_location=DEVICE, weights_only=False)
            if i == 0:
                info = detect_checkpoint_config(ckpt)
                det_type = info["model_type"]
                det_seq = info.get("seq_len", self.cfg.seq_len)
                if det_type != self.cfg.model_type or det_seq != self.cfg.seq_len:
                    self.cfg = ModelConfig(
                        model_type=det_type, seq_len=det_seq,
                        hidden_size=self.cfg.hidden_size, num_layers=self.cfg.num_layers,
                        num_heads=self.cfg.num_heads, dropout=self.cfg.dropout,
                        fc_hidden=self.cfg.fc_hidden, dim_feedforward=self.cfg.dim_feedforward,
                    )
            m = build_model(n_features, self.cfg)
            m.load_state_dict(ckpt["model_state_dict"])
            m.eval()
            self.models.append(m)
            epoch = ckpt.get("epoch", "?")
            vloss = ckpt.get("val_loss", 0)
            label = f"[{ticker}] " if ticker else ""
            print(f"[PREDICTOR] {label}Loaded {p.name} (epoch {epoch}, val_loss {vloss:.6f})")

        label = f"[{ticker}] " if ticker else ""
        print(f"[PREDICTOR] {label}Ensemble size: {len(self.models)}")

    def _resolve_paths(self, model_path: Path | None, ticker: str | None) -> List[Path]:
        """Find all ensemble seed files, or fall back to single model."""
        if model_path:
            base = model_path
        elif ticker:
            base = ticker_model_dir(ticker) / "best_model.pt"
        else:
            base = MODELS_DIR / "best_model.pt"

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

        tkr_label = f" for {ticker}" if ticker else ""
        raise FileNotFoundError(
            f"No model found at {base} or {parent}/{stem}_seed*{suffix}{tkr_label}. "
            f"Train a model first: python scripts/train.py"
        )

    @torch.no_grad()
    def predict(self, window: np.ndarray) -> float:
        """Predict return for a single feature window (ensemble-averaged)."""
        return self.predict_detailed(window)["mean"]

    @torch.no_grad()
    def predict_detailed(self, window: np.ndarray) -> dict:
        """Return per-model predictions with ensemble statistics."""
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


class MultiPredictor:
    """Holds one Predictor per target ticker for multi-instrument trading."""

    def __init__(
        self,
        feature_counts: Dict[str, int],
        tickers: List[str] | None = None,
        cfg: ModelConfig | None = None,
    ):
        self.cfg = cfg or ModelConfig()
        self.tickers = tickers or TARGET_TICKERS
        self.predictors: Dict[str, Predictor] = {}

        for tkr in self.tickers:
            n_feat = feature_counts[tkr]
            self.predictors[tkr] = Predictor(
                n_features=n_feat,
                ticker=tkr,
                cfg=self.cfg,
            )

    def predict_detailed(self, ticker: str, window: np.ndarray) -> dict:
        """Predict for a specific ticker."""
        return self.predictors[ticker].predict_detailed(window)

    @property
    def total_models(self) -> int:
        return sum(len(p.models) for p in self.predictors.values())

    def models_per_ticker(self) -> Dict[str, int]:
        return {tkr: len(p.models) for tkr, p in self.predictors.items()}
