"""CLI: Train ETHU model with multi-seed ensemble and walk-forward cross-validation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import DEVICE, MODELS_DIR, ModelConfig, TrainConfig
from src.data.storage import read_all, init_db
from src.data.preprocessor import (
    build_features, fit_scaler, save_scaler, load_scaler,
    _select_top_features, _prune_low_variance, _prune_correlated,
    TimeSeriesDataset, prepare_datasets,
)
from src.model.architecture import build_model
from src.model.trainer import train_model

_MC = ModelConfig()
_TC = TrainConfig()


def _evaluate(model, ds, batch_size):
    """Return (loss, directional_accuracy, n_samples)."""
    if len(ds) == 0:
        return float("nan"), float("nan"), 0
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    crit = torch.nn.HuberLoss(delta=1.0)
    losses, correct, total = [], 0, 0
    model.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            p = model(xb)
            losses.append(crit(p, yb).item())
            correct += ((p > 0) == (yb > 0)).sum().item()
            total += len(yb)
    return float(np.mean(losses)), correct / total if total else 0.0, total


def _ensemble_eval(models, ds, batch_size):
    """Evaluate ensemble average on a dataset."""
    if len(ds) == 0:
        return float("nan"), float("nan"), 0
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    crit = torch.nn.HuberLoss(delta=1.0)
    all_preds, all_tgt = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        bp = []
        with torch.no_grad():
            for m in models:
                m.eval()
                bp.append(m(xb))
        all_preds.append(torch.stack(bp).mean(dim=0).cpu())
        all_tgt.append(yb)
    preds = torch.cat(all_preds)
    tgt = torch.cat(all_tgt)
    loss = crit(preds, tgt).item()
    correct = ((preds > 0) == (tgt > 0)).sum().item()
    return loss, correct / len(tgt), len(tgt)


def _walk_forward_cv(merged, mcfg, tcfg, n_folds=4, seeds_per_fold=3):
    """Expanding-window walk-forward cross-validation.

    Trains on growing windows, tests on subsequent unseen data.
    Returns per-fold metrics for statistical assessment.
    """
    feat_df, target_col = build_features(merged)
    feature_cols = [c for c in feat_df.columns if c != target_col]
    X_all = feat_df[feature_cols].values.astype(np.float32)
    y_all = feat_df[target_col].values.astype(np.float32)
    n = len(X_all)

    seq = mcfg.seq_len
    test_size = max(n // 8, seq + 5)
    val_size = max(n // 10, seq + 5)
    train_pool = n - val_size - test_size
    step = max(train_pool // max(n_folds, 1), seq + 5)
    results = []

    print(f"\n{'='*65}")
    print(f"  WALK-FORWARD CROSS-VALIDATION  ({n_folds} folds, {seeds_per_fold} seeds each)")
    print(f"{'='*65}")

    for fold in range(n_folds):
        train_end = max(int(n * 0.15), seq + 10) + step * fold
        val_end = train_end + val_size
        test_end = val_end + test_size

        if train_end < seq + 5 or test_end > n:
            continue
        if (val_end - train_end) < seq + 2 or (test_end - val_end) < seq + 2:
            continue

        X_tr = X_all[:train_end]
        y_tr = y_all[:train_end]
        X_va = X_all[train_end:val_end]
        y_va = y_all[train_end:val_end]
        X_te = X_all[val_end:test_end]
        y_te = y_all[val_end:test_end]

        max_feats = 40
        if X_tr.shape[1] > max_feats:
            keep = _select_top_features(X_tr, y_tr, feature_cols, max_feats)
            X_tr, X_va, X_te = X_tr[:, keep], X_va[:, keep], X_te[:, keep]
            n_feat = len(keep)
        else:
            n_feat = X_tr.shape[1]

        sc = fit_scaler(X_tr)
        X_tr_s = sc.transform(X_tr)
        X_va_s = sc.transform(X_va)
        X_te_s = sc.transform(X_te)

        seq = mcfg.seq_len
        tr_ds = TimeSeriesDataset(X_tr_s, y_tr, seq, noise_std=0.02, training=True)
        va_ds = TimeSeriesDataset(X_va_s, y_va, seq)
        te_ds = TimeSeriesDataset(X_te_s, y_te, seq)

        fold_models = []
        print(f"\n  Fold {fold + 1}/{n_folds}  |  train={len(tr_ds)} val={len(va_ds)} test={len(te_ds)} windows")

        for s in range(seeds_per_fold):
            seed = 100 + fold * 10 + s
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

            m = build_model(n_feat, mcfg)
            sp = MODELS_DIR / f"cv_fold{fold}_s{s}.pt"
            tcfg_quiet = TrainConfig(
                epochs=tcfg.epochs, batch_size=tcfg.batch_size,
                learning_rate=tcfg.learning_rate, patience=tcfg.patience,
                warmup_epochs=tcfg.warmup_epochs,
            )
            train_model(m, tr_ds, va_ds, tcfg_quiet, sp, quiet=True)
            ckpt = torch.load(sp, map_location=DEVICE, weights_only=False)
            m.load_state_dict(ckpt["model_state_dict"])
            m.eval()
            fold_models.append(m)
            sp.unlink(missing_ok=True)

        loss, acc, tot = _ensemble_eval(fold_models, te_ds, tcfg.batch_size)
        results.append({"fold": fold + 1, "loss": loss, "acc": acc, "n": tot})
        print(f"    Ensemble: loss={loss:.6f}  dir_acc={acc:.1%} ({int(acc*tot)}/{tot})")

    if results:
        accs = [r["acc"] for r in results if not np.isnan(r["acc"])]
        losses = [r["loss"] for r in results if not np.isnan(r["loss"])]
        print(f"\n  CV Summary ({len(results)} folds):")
        print(f"    Dir Accuracy:  mean={np.mean(accs):.1%}  std={np.std(accs):.1%}  "
              f"range=[{min(accs):.1%}, {max(accs):.1%}]")
        print(f"    Test Loss:     mean={np.mean(losses):.6f}  std={np.std(losses):.6f}")
    return results


@click.command()
@click.option("--model", "model_type", default=_MC.model_type, type=click.Choice(["lstm", "transformer"]))
@click.option("--epochs", default=_TC.epochs, help="Max training epochs.")
@click.option("--seq-len", default=_MC.seq_len, help="Lookback window length.")
@click.option("--batch-size", default=_TC.batch_size)
@click.option("--lr", default=_TC.learning_rate, help="Initial learning rate.")
@click.option("--patience", default=_TC.patience, help="Early stopping patience.")
@click.option("--seeds", default=10, help="Number of ensemble seeds to train.")
@click.option("--cv-folds", default=4, help="Walk-forward CV folds (0 to skip).")
@click.option("--save-path", default=None, help="Model save path (base name for ensemble).")
def main(model_type, epochs, seq_len, batch_size, lr, patience, seeds, cv_folds, save_path):
    """Train the ETHU neural prediction model with thorough evaluation."""
    init_db()
    print("[TRAIN] Loading data from database...")
    merged = read_all()
    if merged.empty:
        print("ERROR: No data in database. Run `python scripts/fetch_data.py` first.")
        sys.exit(1)

    mcfg = ModelConfig(model_type=model_type, seq_len=seq_len)
    tcfg = TrainConfig(
        epochs=epochs, batch_size=batch_size,
        learning_rate=lr, patience=patience,
    )

    # Walk-forward cross-validation first
    if cv_folds > 0:
        _walk_forward_cv(merged, mcfg, tcfg, n_folds=cv_folds, seeds_per_fold=3)

    # Primary ensemble training
    print(f"\n{'='*65}")
    print(f"  PRIMARY ENSEMBLE TRAINING  ({seeds} seeds)")
    print(f"{'='*65}")

    print("\n[TRAIN] Preparing datasets...")
    train_ds, val_ds, test_ds, scaler, feature_cols = prepare_datasets(
        merged, mcfg, tcfg.train_ratio, tcfg.val_ratio
    )
    n_features = len(feature_cols)
    print(f"  Features: {n_features}  |  Seq len: {seq_len}")

    base_path = Path(save_path) if save_path else MODELS_DIR / "best_model.pt"
    models = []
    seed_results = []

    for seed_idx in range(seeds):
        seed = 42 + seed_idx
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        if seeds == 1:
            sp = base_path
        else:
            sp = base_path.parent / f"{base_path.stem}_seed{seed_idx}{base_path.suffix}"

        print(f"\n--- Seed {seed_idx + 1}/{seeds} (seed={seed}) ---")
        model = build_model(n_features, mcfg)
        train_model(model, train_ds, val_ds, tcfg, sp, quiet=(seeds > 3))

        ckpt = torch.load(sp, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models.append(model)

        loss, acc, tot = _evaluate(model, test_ds, batch_size)
        seed_results.append({"seed": seed, "loss": loss, "acc": acc, "n": tot})
        print(f"  Test: loss={loss:.6f}  dir_acc={acc:.1%} ({int(acc*tot)}/{tot})")

    # Summary table
    print(f"\n{'='*65}")
    print("  INDIVIDUAL SEED RESULTS")
    print(f"{'='*65}")
    print(f"  {'Seed':<8} {'Val Loss':<12} {'Test Loss':<12} {'Dir Acc':<12}")
    print(f"  {'-'*44}")
    for i, r in enumerate(seed_results):
        print(f"  {r['seed']:<8} {'---':<12} {r['loss']:<12.6f} {r['acc']:<12.1%}")

    accs = [r["acc"] for r in seed_results]
    losses = [r["loss"] for r in seed_results]
    print(f"\n  Mean dir_acc: {np.mean(accs):.1%}  |  Std: {np.std(accs):.1%}")
    print(f"  Mean loss:    {np.mean(losses):.6f}  |  Std: {np.std(losses):.6f}")

    # Ensemble evaluation
    e_loss, e_acc, e_tot = _ensemble_eval(models, test_ds, batch_size)
    print(f"\n  ENSEMBLE ({len(models)} models):")
    print(f"    Test loss: {e_loss:.6f}  |  Dir accuracy: {e_acc:.1%} ({int(e_acc*e_tot)}/{e_tot})")

    print(f"\n[TRAIN] Complete. {seeds} models saved to {base_path.parent}/")


if __name__ == "__main__":
    main()
