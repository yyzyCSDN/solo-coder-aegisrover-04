"""Named, versioned checkpoints of resumable task state.

A checkpoint captures the *intermediate result* of a running task so the task can
be evicted from its executor (preemption) and later resumed from the exact same
point instead of restarting. State is serialised with a digest; a checkpoint that
fails verification is rejected loudly rather than silently restarting the task
from corrupted state.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

__all__ = ('Checkpoint', 'CheckpointError', 'CheckpointStore', 'create', 'verify')


class CheckpointError(RuntimeError):
    pass


def _canonical(state: Any) -> bytes:
    return json.dumps(state, sort_keys=True, separators=(',', ':')).encode()


def create(name, sequence, state):
    body = _canonical(state)
    return {'name': name, 'sequence': sequence, 'state': state,
            'digest': hashlib.sha256(body).hexdigest()}


def verify(c):
    expected = c['digest']
    actual = hashlib.sha256(_canonical(c['state'])).hexdigest()
    if expected != actual:
        raise CheckpointError(f"checkpoint {c.get('name')!r} digest mismatch")
    return True


@dataclass(frozen=True)
class Checkpoint:
    """Immutable snapshot of one task's progress."""

    name: str
    sequence: int
    state: Any
    digest: str
    created_at: float = 0.0

    @classmethod
    def capture(cls, name: str, sequence: int, state: Any, *, created_at: float = 0.0) -> 'Checkpoint':
        return cls(name=name, sequence=int(sequence), state=state,
                   digest=hashlib.sha256(_canonical(state)).hexdigest(),
                   created_at=float(created_at))

    @classmethod
    def from_dict(cls, payload: dict) -> 'Checkpoint':
        return cls(payload['name'], int(payload['sequence']), payload['state'],
                   payload['digest'], float(payload.get('created_at', 0.0)))

    def to_dict(self) -> dict:
        return {'name': self.name, 'sequence': self.sequence, 'state': self.state,
                'digest': self.digest, 'created_at': self.created_at}

    def verify(self) -> bool:
        return verify(self.to_dict())


class CheckpointStore:
    """Per-task checkpoint history with monotonically increasing sequence numbers.

    Saving verifies the digest before storing and rejects an out-of-order sequence,
    so a task can checkpoint repeatedly (each resume advances its own sequence) and
    an older snapshot can never overwrite a newer one after a racing save.
    """

    def __init__(self):
        self._items: dict[str, list[Checkpoint]] = {}

    def save(self, checkpoint: Checkpoint) -> Checkpoint:
        checkpoint.verify()
        history = self._items.setdefault(checkpoint.name, [])
        if history and checkpoint.sequence <= history[-1].sequence:
            raise CheckpointError(
                f"{checkpoint.name!r}: checkpoint sequence {checkpoint.sequence} "
                f"does not follow {history[-1].sequence}")
        history.append(checkpoint)
        return checkpoint

    def latest(self, name: str) -> Checkpoint:
        try:
            return self._items[name][-1]
        except (KeyError, IndexError):
            raise CheckpointError(f'no checkpoint for {name!r}') from None

    def load(self, name: str) -> Checkpoint:
        checkpoint = self.latest(name)
        checkpoint.verify()
        return checkpoint

    def discard(self, name: str) -> int:
        return len(self._items.pop(name, ()))

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._items))

    def __len__(self) -> int:
        return len(self._items)
