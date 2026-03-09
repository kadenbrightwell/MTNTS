"""Walk-forward backtesting engine with slippage and transaction costs.

Provides both a single-ticker BacktestEngine and a MultiTickerEngine
that manages independent long/flat positions across UVXY, SPXU, SVIX, SPXL.
"""

from __future__ import annotations

from typing import List, Tuple, Dict

import numpy as np
import pandas as pd

from config import BacktestConfig, TARGET_TICKERS


class BacktestEngine:
    """Simulates a long/flat strategy for a single ticker using model predictions."""

    def __init__(self, cfg: BacktestConfig | None = None):
        self.cfg = cfg or BacktestConfig()

    def run(
        self,
        predictions: np.ndarray,
        actual_returns: np.ndarray,
        prices: pd.Series,
    ) -> Tuple[pd.Series, List[float], pd.Series]:
        """Execute the backtest.

        Args:
            predictions: array of predicted next-period log-returns.
            actual_returns: array of realised next-period log-returns (aligned).
            prices: close prices (aligned, same length).

        Returns:
            (equity_curve, trade_returns, benchmark_curve)
        """
        n = len(predictions)
        capital = self.cfg.initial_capital
        equity = [capital]
        position = 0  # 0 = flat, 1 = long
        entry_price = 0.0
        trade_rets: List[float] = []

        slip = self.cfg.slippage_bps / 10_000
        comm = self.cfg.commission_bps / 10_000

        for i in range(n):
            pred = predictions[i]
            ret = actual_returns[i]
            price = prices.iloc[i] if isinstance(prices, pd.Series) else prices[i]

            if position == 1:
                pnl = capital * (np.exp(ret) - 1)
                capital += pnl

            if pred > self.cfg.long_threshold and position == 0:
                cost = capital * (slip + comm)
                capital -= cost
                position = 1
                entry_price = price

            elif pred < self.cfg.flat_threshold and position == 1:
                cost = capital * (slip + comm)
                capital -= cost
                trade_ret = (price / entry_price) - 1 if entry_price > 0 else 0.0
                trade_rets.append(trade_ret)
                position = 0
                entry_price = 0.0

            equity.append(capital)

        if position == 1 and entry_price > 0:
            last_price = prices.iloc[-1] if isinstance(prices, pd.Series) else prices[-1]
            trade_rets.append((last_price / entry_price) - 1)

        idx = prices.index if isinstance(prices, pd.Series) else range(n + 1)
        if isinstance(prices, pd.Series):
            eq_index = list(prices.index[:n]) + [prices.index[-1]]
            if len(eq_index) != len(equity):
                eq_index = list(range(len(equity)))
        else:
            eq_index = list(range(len(equity)))

        equity_curve = pd.Series(equity, index=eq_index, name="equity")

        bench_start = prices.iloc[0] if isinstance(prices, pd.Series) else prices[0]
        bench = prices / bench_start * self.cfg.initial_capital
        if isinstance(bench, pd.Series):
            bench.name = "benchmark"

        return equity_curve, trade_rets, bench


class MultiTickerEngine:
    """Runs independent BacktestEngine instances for each ticker and aggregates results.

    Capital is split equally (or by configurable weights) across tickers.
    Each ticker has its own long/flat position.
    """

    def __init__(
        self,
        tickers: List[str] | None = None,
        cfg: BacktestConfig | None = None,
        weights: Dict[str, float] | None = None,
    ):
        self.cfg = cfg or BacktestConfig()
        self.tickers = tickers or TARGET_TICKERS
        n = len(self.tickers)

        if weights:
            self.weights = weights
        else:
            self.weights = {t: 1.0 / n for t in self.tickers}

        self.engines: Dict[str, BacktestEngine] = {}
        for tkr in self.tickers:
            per_ticker_cfg = BacktestConfig(
                initial_capital=self.cfg.initial_capital * self.weights[tkr],
                slippage_bps=self.cfg.slippage_bps,
                commission_bps=self.cfg.commission_bps,
                long_threshold=self.cfg.long_threshold,
                flat_threshold=self.cfg.flat_threshold,
            )
            self.engines[tkr] = BacktestEngine(per_ticker_cfg)

    def run(
        self,
        predictions: Dict[str, np.ndarray],
        actual_returns: Dict[str, np.ndarray],
        prices: Dict[str, pd.Series],
    ) -> Dict[str, Tuple[pd.Series, List[float], pd.Series]]:
        """Run backtest for each ticker independently.

        Args:
            predictions: {ticker: predicted_returns}
            actual_returns: {ticker: actual_returns}
            prices: {ticker: close_prices}

        Returns:
            {ticker: (equity_curve, trade_returns, benchmark_curve)}
        """
        results = {}
        for tkr in self.tickers:
            if tkr in predictions and tkr in actual_returns and tkr in prices:
                results[tkr] = self.engines[tkr].run(
                    predictions[tkr], actual_returns[tkr], prices[tkr]
                )
        return results

    @staticmethod
    def aggregate_equity(
        results: Dict[str, Tuple[pd.Series, List[float], pd.Series]],
    ) -> pd.Series:
        """Sum per-ticker equity curves into a portfolio-level curve."""
        curves = []
        for tkr, (eq, _, _) in results.items():
            curves.append(eq.rename(tkr))

        if not curves:
            return pd.Series(dtype=float, name="portfolio_equity")

        combined = pd.concat(curves, axis=1).ffill().bfill()
        portfolio = combined.sum(axis=1)
        portfolio.name = "portfolio_equity"
        return portfolio
