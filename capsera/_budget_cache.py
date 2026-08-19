"""
The cached budget snapshot: what the backend last told us about budget state.

Holds one immutable ``BudgetSnapshot``, swapped atomically. Readers on the hot
path take no lock and never see a half-applied update — they read a reference
to a frozen object, and a refresh replaces the reference.

Persistence
-----------
The snapshot is also written to disk, because a process that dies before it
ever refreshes never enforces anything. That is fine for a long-running
service (one cold call) and useless for a Lambda that makes three calls and
exits — without a warm start, *every* invocation is a cold start and the
budget is decorative. The disk copy makes the second call of the second
invocation enforce correctly.

Every disk operation is best-effort. A read-only filesystem, a full disk, a
corrupt file, a snapshot written by a different SDK version: all degrade to
"no cached snapshot", never to an exception on the caller's LLM call.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ._budget_rules import RULES_VERSION, BudgetRule, BudgetSnapshot, CustomerRule

logger = logging.getLogger("capsera")

CACHE_FILENAME_PREFIX = "capsera-budget-"
CACHE_SCHEMA = 1


class SnapshotCache:
    """Holds the current snapshot and, optionally, persists it."""

    def __init__(self, *, endpoint: str, api_key: str, persist: bool = True) -> None:
        self._snapshot: BudgetSnapshot | None = None
        self._lock = threading.Lock()
        self._persist = persist
        self._path = _cache_path(endpoint, api_key) if persist else None
        self._warned_write = False

    # ------------------------------------------------------------------ #
    # Hot path
    # ------------------------------------------------------------------ #

    def get(self) -> BudgetSnapshot | None:
        """The current snapshot, or None if we have never had one."""
        return self._snapshot

    def set(self, snapshot: BudgetSnapshot) -> None:
        """Install a fresh snapshot and persist it."""
        self._snapshot = snapshot
        if self._persist:
            self._save(snapshot)

    def age_seconds(self, now: datetime | None = None) -> float | None:
        """Seconds since the current snapshot was fetched, None if there is none."""
        snapshot = self._snapshot
        if snapshot is None or snapshot.fetched_at is None:
            return None
        return ((now or datetime.now(UTC)) - snapshot.fetched_at).total_seconds()

    # ------------------------------------------------------------------ #
    # Disk
    # ------------------------------------------------------------------ #

    def load(self) -> BudgetSnapshot | None:
        """Warm the cache from disk. Returns the snapshot if one was usable."""
        if not self._persist or self._path is None:
            return None
        try:
            raw = json.loads(self._path.read_text())
        except FileNotFoundError:
            return None
        except Exception:
            logger.debug("capsera: unreadable budget cache at %s", self._path, exc_info=True)
            return None

        try:
            if raw.get("schema") != CACHE_SCHEMA:
                return None
            snapshot = parse_snapshot(raw.get("snapshot") or {})
        except Exception:
            logger.debug("capsera: corrupt budget cache at %s", self._path, exc_info=True)
            return None

        with self._lock:
            if self._snapshot is None:
                self._snapshot = snapshot
        return snapshot

    def _save(self, snapshot: BudgetSnapshot) -> None:
        if self._path is None:
            return
        payload = {"schema": CACHE_SCHEMA, "snapshot": serialize_snapshot(snapshot)}
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            # Write-then-rename: a process killed mid-write must not leave a
            # truncated file that the next start reads as corrupt.
            fd, tmp = tempfile.mkstemp(dir=str(self._path.parent), prefix=".capsera-")
            try:
                with os.fdopen(fd, "w") as handle:
                    json.dump(payload, handle)
                os.replace(tmp, self._path)
            except Exception:
                _unlink(tmp)
                raise
        except Exception:
            if not self._warned_write:
                logger.debug(
                    "capsera: could not persist budget cache to %s", self._path, exc_info=True
                )
                self._warned_write = True


def _cache_path(endpoint: str, api_key: str) -> Path:
    """A per-workspace cache file in the system temp dir.

    Keyed by endpoint *and* api key so two workspaces sharing a host — or a
    rotated key — never read each other's budget state. The key is hashed, not
    stored: this file is world-readable on a shared temp dir.
    """
    digest = hashlib.sha256(f"{endpoint}|{api_key}".encode()).hexdigest()[:16]
    return Path(tempfile.gettempdir()) / f"{CACHE_FILENAME_PREFIX}{digest}.json"


def _unlink(path: str) -> None:
    try:
        os.unlink(path)
    except Exception:
        pass


# --------------------------------------------------------------------------- #
# Wire format
# --------------------------------------------------------------------------- #


def parse_snapshot(raw: dict[str, Any]) -> BudgetSnapshot:
    """Build a snapshot from the backend's JSON.

    Unknown or malformed individual rows are skipped rather than failing the
    whole snapshot: one bad budget row should not disable enforcement for
    every other budget in the workspace.
    """
    budgets: list[BudgetRule] = []
    for row in raw.get("budgets") or []:
        try:
            budgets.append(
                BudgetRule(
                    id=str(row["id"]),
                    name=str(row.get("name") or ""),
                    scope_type=str(row["scope_type"]),
                    scope_id=row.get("scope_id"),
                    action=str(row.get("action") or "alert"),
                    limit_usd=float(row.get("limit_usd") or 0.0),
                    current_spend_usd=float(row.get("current_spend_usd") or 0.0),
                    period_end=_parse_dt(row["period_end"]),
                    per_call_ceiling_usd=(
                        float(row["per_call_ceiling_usd"])
                        if row.get("per_call_ceiling_usd")
                        else None
                    ),
                )
            )
        except Exception:
            logger.debug("capsera: skipping malformed budget row %r", row, exc_info=True)

    customers: list[CustomerRule] = []
    for row in raw.get("customers") or []:
        try:
            customers.append(
                CustomerRule(
                    customer_id=str(row["customer_id"]),
                    name=str(row.get("name") or ""),
                    monthly_revenue_usd=float(row.get("monthly_revenue_usd") or 0.0),
                    margin_alert_threshold=float(row.get("margin_alert_threshold") or 0.0),
                    current_spend_usd=float(row.get("current_spend_usd") or 0.0),
                )
            )
        except Exception:
            logger.debug("capsera: skipping malformed customer row %r", row, exc_info=True)

    targets = raw.get("margin_targets") or {}
    fetched = raw.get("fetched_at")

    return BudgetSnapshot(
        budgets=tuple(budgets),
        customers=tuple(customers),
        margin_targets={str(k): str(v) for k, v in targets.items()} if targets else {},
        fetched_at=_parse_dt(fetched) if fetched else datetime.now(UTC),
        rules_version=int(raw.get("rules_version") or RULES_VERSION),
    )


def serialize_snapshot(snapshot: BudgetSnapshot) -> dict[str, Any]:
    """Inverse of ``parse_snapshot``, for the disk copy."""
    return {
        "rules_version": snapshot.rules_version,
        "fetched_at": snapshot.fetched_at.isoformat() if snapshot.fetched_at else None,
        "margin_targets": dict(snapshot.margin_targets),
        "budgets": [
            {
                "id": b.id,
                "name": b.name,
                "scope_type": b.scope_type,
                "scope_id": b.scope_id,
                "action": b.action,
                "limit_usd": b.limit_usd,
                "current_spend_usd": b.current_spend_usd,
                "period_end": b.period_end.isoformat(),
                "per_call_ceiling_usd": b.per_call_ceiling_usd,
            }
            for b in snapshot.budgets
        ],
        "customers": [
            {
                "customer_id": c.customer_id,
                "name": c.name,
                "monthly_revenue_usd": c.monthly_revenue_usd,
                "margin_alert_threshold": c.margin_alert_threshold,
                "current_spend_usd": c.current_spend_usd,
            }
            for c in snapshot.customers
        ],
    }


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    text = str(value).replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
