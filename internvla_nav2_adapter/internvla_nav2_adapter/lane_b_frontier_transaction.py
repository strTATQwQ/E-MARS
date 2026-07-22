"""One-shot identity-bound storage for the private Lane-B adviser transaction."""

from __future__ import annotations

import re
import secrets
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable


_SHA256_RE = re.compile(r"[0-9a-f]{64}")


class FrontierTransactionError(ValueError):
    """Raised when a prepare/commit transaction is not current and exact."""


@dataclass(frozen=True)
class PreparedFrontier:
    frontier_id: int
    discrete_action: int
    relative_x: float
    relative_z: float
    distance_m: float
    bearing_deg: float
    path_sha256: str
    path: Any

    def __post_init__(self) -> None:
        if isinstance(self.frontier_id, bool) or self.frontier_id < 0:
            raise FrontierTransactionError("frontier_id must be non-negative")
        if self.discrete_action not in {1, 2, 3}:
            raise FrontierTransactionError("frontier action must be MOVE-only")
        if not _SHA256_RE.fullmatch(self.path_sha256):
            raise FrontierTransactionError("path_sha256 is invalid")


@dataclass(frozen=True)
class PreparedTransaction:
    identity: tuple[str, int, int, str, str]
    token: str
    created_monotonic: float
    prepared_sim_ns: int
    frontiers: tuple[PreparedFrontier, ...]


class FrontierTransactionStore:
    """Keep at most one current, one-use transaction for this adapter."""

    def __init__(
        self,
        *,
        ttl_sec: float = 15.0,
        clock: Callable[[], float] = time.monotonic,
        token_factory: Callable[[], str] = lambda: secrets.token_hex(16),
    ) -> None:
        if not 12.0 <= float(ttl_sec) <= 30.0:
            raise FrontierTransactionError("transaction TTL is outside bounds")
        self.ttl_sec = float(ttl_sec)
        self.clock = clock
        self.token_factory = token_factory
        self._lock = threading.Lock()
        self._current: PreparedTransaction | None = None

    @property
    def current(self) -> PreparedTransaction | None:
        with self._lock:
            return self._current

    def prepare(
        self,
        identity: tuple[str, int, int, str, str],
        frontiers: Iterable[PreparedFrontier],
        *,
        prepared_sim_ns: int,
    ) -> PreparedTransaction:
        values = tuple(frontiers)
        ids = [value.frontier_id for value in values]
        if not values or len(ids) != len(set(ids)):
            raise FrontierTransactionError("prepared frontiers must be non-empty and unique")
        if isinstance(prepared_sim_ns, bool) or prepared_sim_ns <= 0:
            raise FrontierTransactionError("prepared simulation time is invalid")
        token = self.token_factory()
        if not re.fullmatch(r"[0-9a-f]{32}", token):
            raise FrontierTransactionError("transaction token is invalid")
        transaction = PreparedTransaction(
            identity=identity,
            token=token,
            created_monotonic=self.clock(),
            prepared_sim_ns=int(prepared_sim_ns),
            frontiers=values,
        )
        with self._lock:
            self._current = transaction
        return transaction

    def consume(
        self,
        *,
        identity: tuple[str, int, int, str, str],
        token: str,
        select_frontier: bool,
        frontier_id: int | None = None,
        expected_path_sha256: str = "",
    ) -> tuple[PreparedFrontier | None, int]:
        with self._lock:
            current = self._current
            self._current = None
        if current is None:
            raise FrontierTransactionError("no prepared frontier transaction")
        if self.clock() - current.created_monotonic > self.ttl_sec:
            raise FrontierTransactionError("prepared frontier transaction expired")
        if current.identity != identity or current.token != token:
            raise FrontierTransactionError("prepared frontier identity or token mismatch")
        if not select_frontier:
            if frontier_id not in {None, 0} or expected_path_sha256:
                raise FrontierTransactionError("release must not carry a frontier")
            return None, current.prepared_sim_ns
        if isinstance(frontier_id, bool) or frontier_id is None:
            raise FrontierTransactionError("selection requires frontier_id")
        selected = next(
            (value for value in current.frontiers if value.frontier_id == frontier_id),
            None,
        )
        if selected is None:
            raise FrontierTransactionError("frontier is not current")
        if selected.path_sha256 != expected_path_sha256:
            raise FrontierTransactionError("frontier path digest mismatch")
        return selected, current.prepared_sim_ns

    def invalidate(self) -> None:
        with self._lock:
            self._current = None
