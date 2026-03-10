"""Execution adapters for turning recommendations into orders.

Default behavior is DRY-RUN (prints/logs orders but does not place them).

Optional: Alpaca REST (paper or live) via environment variables:
  - ALPACA_KEY_ID
  - ALPACA_SECRET_KEY
  - ALPACA_BASE_URL (optional; default paper)
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any

import requests


Side = Literal["buy", "sell"]


@dataclass
class OrderRequest:
    symbol: str
    side: Side
    qty: int
    limit_price: float
    time_in_force: str = "day"


@dataclass
class OrderResult:
    ok: bool
    order_id: Optional[str] = None
    error: Optional[str] = None
    raw: Optional[Dict[str, Any]] = None


class ExecutionClient:
    """Abstract execution client."""

    name: str = "base"

    def place_limit_order(self, req: OrderRequest) -> OrderResult:  # pragma: no cover
        raise NotImplementedError


class DryRunExecutionClient(ExecutionClient):
    name = "dry-run"

    def place_limit_order(self, req: OrderRequest) -> OrderResult:
        print(
            f"[EXEC:{self.name}] {req.side.upper()} {req.qty} {req.symbol} "
            f"LIMIT ${req.limit_price:.2f} tif={req.time_in_force}"
        )
        return OrderResult(ok=True, order_id="dryrun")


class AlpacaExecutionClient(ExecutionClient):
    """Minimal Alpaca REST client for limit orders.

    Notes:
    - This only covers order placement. Portfolio sync/risk checks are handled separately.
    - Uses v2 trading endpoint.
    """

    name = "alpaca"

    def __init__(
        self,
        key_id: str,
        secret_key: str,
        base_url: str = "https://paper-api.alpaca.markets",
        timeout_s: float = 10.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self._headers = {
            "APCA-API-KEY-ID": key_id,
            "APCA-API-SECRET-KEY": secret_key,
            "Content-Type": "application/json",
        }

    @classmethod
    def from_env(cls) -> "AlpacaExecutionClient":
        key_id = os.getenv("ALPACA_KEY_ID", "").strip()
        secret = os.getenv("ALPACA_SECRET_KEY", "").strip()
        base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").strip()
        if not key_id or not secret:
            raise RuntimeError(
                "Missing Alpaca credentials. Set ALPACA_KEY_ID and ALPACA_SECRET_KEY."
            )
        return cls(key_id=key_id, secret_key=secret, base_url=base_url)

    def place_limit_order(self, req: OrderRequest) -> OrderResult:
        url = f"{self.base_url}/v2/orders"
        payload = {
            "symbol": req.symbol,
            "qty": str(int(req.qty)),
            "side": req.side,
            "type": "limit",
            "time_in_force": req.time_in_force,
            "limit_price": f"{float(req.limit_price):.2f}",
        }
        try:
            r = requests.post(url, json=payload, headers=self._headers, timeout=self.timeout_s)
            if r.status_code >= 200 and r.status_code < 300:
                data = r.json()
                return OrderResult(ok=True, order_id=data.get("id"), raw=data)
            return OrderResult(ok=False, error=f"{r.status_code}: {r.text}")
        except Exception as e:
            return OrderResult(ok=False, error=str(e))

