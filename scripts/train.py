"""CLI: Train models for each target ticker with multi-seed ensemble and walk-forward CV."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import click
import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (
    DEVICE, MODELS_DIR, ModelConfig, TrainConfig,
    TARGET_TICKERS, ticker_model_dir,
)
from src.data.storage import read_all, init_db
from src.data.preprocessor import (
    build_features, fit_scaler, save_scaler, save_feature_cols,
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


def _walk_forward_cv(merged, target_ticker, mcfg, tcfg, n_folds=4, seeds_per_fold=3):
    """Expanding-window walk-forward cross-validation for a single ticker."""
    feat_df, target_col = build_features(merged, target_ticker=target_ticker)
    feature_cols = [c for c in feat_df.columns if c != target_col]
    X_all = feat_df[feature_cols].values.astype(np.float32)
    y_all = feat_df[target_col].values.astype(np.float32)
    n = len(X_all)

    seq = mcfg.seq_len
    min_windows = max(5, seq // 10)
    min_split = seq + min_windows
    test_size = max(n // 8, min_split)
    val_size = max(n // 10, min_split)

    if n < seq + min_windows + val_size + test_size:
        print(f"\n  [CV] Skipping CV: not enough data ({n} samples) for seq_len={seq}")
        return []

    train_pool = n - val_size - test_size
    step = max(train_pool // max(n_folds, 1), min_split)
    results = []

    print(f"\n{'='*65}")
    print(f"  WALK-FORWARD CV  [{target_ticker}]  ({n_folds} folds, {seeds_per_fold} seeds each)")
    print(f"{'='*65}")

    for fold in range(n_folds):
        train_end = max(int(n * 0.15), min_split) + step * fold
        val_end = train_end + val_size
        test_end = val_end + test_size

        if train_end < min_split or test_end > n:
            continue
        if (val_end - train_end) < min_split or (test_end - val_end) < min_split:
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

        model_dir = ticker_model_dir(target_ticker)
        for s in range(seeds_per_fold):
            seed = 100 + fold * 10 + s
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

            m = build_model(n_feat, mcfg)
            sp = model_dir / f"cv_fold{fold}_s{s}.pt"
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
        if tot > 0:
            print(f"    Ensemble: loss={loss:.6f}  dir_acc={acc:.1%} ({int(acc*tot)}/{tot})")
        else:
            print(f"    Ensemble: no test windows for this fold")

    if results:
        accs = [r["acc"] for r in results if not np.isnan(r["acc"])]
        losses = [r["loss"] for r in results if not np.isnan(r["loss"])]
        print(f"\n  CV Summary ({len(results)} folds):")
        print(f"    Dir Accuracy:  mean={np.mean(accs):.1%}  std={np.std(accs):.1%}  "
              f"range=[{min(accs):.1%}, {max(accs):.1%}]")
        print(f"    Test Loss:     mean={np.mean(losses):.6f}  std={np.std(losses):.6f}")
    return results


def _train_single_ticker(target_ticker, merged, mcfg, tcfg, seeds, cv_folds, save_path):
    """Full training pipeline for one target ticker."""
    print(f"\n{'#'*65}")
    print(f"  TRAINING: {target_ticker}")
    print(f"{'#'*65}")

    if cv_folds > 0:
        _walk_forward_cv(merged, target_ticker, mcfg, tcfg, n_folds=cv_folds, seeds_per_fold=3)

    print(f"\n{'='*65}")
    print(f"  PRIMARY ENSEMBLE  [{target_ticker}]  ({seeds} seeds)")
    print(f"{'='*65}")

    print(f"\n[TRAIN] [{target_ticker}] Preparing datasets...")
    train_ds, val_ds, test_ds, scaler, feature_cols = prepare_datasets(
        merged, target_ticker=target_ticker, cfg=mcfg,
        train_ratio=tcfg.train_ratio, val_ratio=tcfg.val_ratio,
    )
    n_features = len(feature_cols)
    print(f"  Features: {n_features}  |  Seq len: {mcfg.seq_len}")

    model_dir = ticker_model_dir(target_ticker)
    if save_path:
        base_path = Path(save_path)
    else:
        base_path = model_dir / "best_model.pt"

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

        print(f"\n--- [{target_ticker}] Seed {seed_idx + 1}/{seeds} (seed={seed}) ---")
        model = build_model(n_features, mcfg)
        train_model(model, train_ds, val_ds, tcfg, sp, quiet=(seeds > 3))

        ckpt = torch.load(sp, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()
        models.append(model)

        loss, acc, tot = _evaluate(model, test_ds, tcfg.batch_size)
        seed_results.append({"seed": seed, "loss": loss, "acc": acc, "n": tot})
        if tot > 0:
            print(f"  Test: loss={loss:.6f}  dir_acc={acc:.1%} ({int(acc*tot)}/{tot})")
        else:
            print(f"  Test: no test windows (seq_len > test split size)")

    print(f"\n{'='*65}")
    print(f"  SEED RESULTS  [{target_ticker}]")
    print(f"{'='*65}")
    print(f"  {'Seed':<8} {'Test Loss':<12} {'Dir Acc':<12}")
    print(f"  {'-'*32}")
    for r in seed_results:
        print(f"  {r['seed']:<8} {r['loss']:<12.6f} {r['acc']:<12.1%}")

    accs = [r["acc"] for r in seed_results if not np.isnan(r["acc"])]
    losses = [r["loss"] for r in seed_results if not np.isnan(r["loss"])]
    if accs:
        print(f"\n  Mean dir_acc: {np.mean(accs):.1%}  |  Std: {np.std(accs):.1%}")
        print(f"  Mean loss:    {np.mean(losses):.6f}  |  Std: {np.std(losses):.6f}")
    else:
        print(f"\n  No test results available (test dataset was empty)")

    e_loss, e_acc, e_tot = _ensemble_eval(models, test_ds, tcfg.batch_size)
    print(f"\n  ENSEMBLE ({len(models)} models):")
    if e_tot > 0:
        print(f"    Test loss: {e_loss:.6f}  |  Dir accuracy: {e_acc:.1%} ({int(e_acc*e_tot)}/{e_tot})")
    else:
        print(f"    No test windows available for ensemble evaluation")

    print(f"\n[TRAIN] [{target_ticker}] Complete. {seeds} models saved to {base_path.parent}/")
    return n_features


@click.command()
@click.option("--model", "model_type", default=_MC.model_type, type=click.Choice(["lstm", "transformer"]))
@click.option("--epochs", default=_TC.epochs, help="Max training epochs.")
@click.option("--seq-len", default=_MC.seq_len, help="Lookback window length.")
@click.option("--batch-size", default=_TC.batch_size)
@click.option("--lr", default=_TC.learning_rate, help="Initial learning rate.")
@click.option("--patience", default=_TC.patience, help="Early stopping patience.")
@click.option("--seeds", default=10, help="Number of ensemble seeds to train per ticker.")
@click.option("--cv-folds", default=4, help="Walk-forward CV folds (0 to skip).")
@click.option("--ticker", default=None, help="Train a single ticker (e.g. UVXY). Default: all 4.")
@click.option("--save-path", default=None, help="Model save path (overrides per-ticker default).")
def main(model_type, epochs, seq_len, batch_size, lr, patience, seeds, cv_folds, ticker, save_path):
    """Train neural prediction models for UVXY, SPXU, SVIX, SPXL."""
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

    tickers_to_train = [ticker] if ticker else TARGET_TICKERS

    for tkr in tickers_to_train:
        if tkr not in TARGET_TICKERS:
            print(f"WARNING: {tkr} is not in TARGET_TICKERS, training anyway.")
        _train_single_ticker(tkr, merged, mcfg, tcfg, seeds, cv_folds, save_path)

    print(f"\n{'#'*65}")
    print(f"  ALL TRAINING COMPLETE")
    print(f"  Tickers: {', '.join(tickers_to_train)}")
    print(f"  Seeds per ticker: {seeds}")
    print(f"{'#'*65}")


if __name__ == "__main__":
    main()
