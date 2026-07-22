"""Pure request-identity and simulator-stamp guards for observation transport."""

from __future__ import annotations

from dataclasses import dataclass

from .protocol import RequestIdentity


def describe_request_identity(identity: RequestIdentity) -> str:
    """Return every request-identity field for fail-closed diagnostics."""

    return (
        "{"
        f"episode_id={identity.episode_id!r}, "
        f"reset_generation={identity.reset_generation}, "
        f"sequence_id={identity.sequence_id}, "
        f"request_id={identity.request_id!r}"
        "}"
    )


@dataclass
class ObservationStampGate:
    """Prevent different request identities from sharing one simulator stamp."""

    last_identity: RequestIdentity | None = None
    last_stamp_ns: int = 0

    def accepts(self, identity: RequestIdentity, stamp_ns: int) -> bool:
        stamp_ns = int(stamp_ns)
        if stamp_ns <= 0:
            return False
        if self.last_identity is None:
            return True
        if identity == self.last_identity:
            # An idempotent retry may reuse its original simulator stamp, but
            # it must never move backwards.
            return stamp_ns >= self.last_stamp_ns
        return stamp_ns > self.last_stamp_ns

    def commit(self, identity: RequestIdentity, stamp_ns: int) -> None:
        if not self.accepts(identity, stamp_ns):
            raise ValueError("observation stamp does not satisfy the identity barrier")
        self.last_identity = identity
        self.last_stamp_ns = int(stamp_ns)
