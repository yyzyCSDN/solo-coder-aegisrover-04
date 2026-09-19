"""Priority queues: static ordering and wait-time aging.

The static :class:`Queue` orders by fixed priority. Real dispatch needs one more
property: a low-priority item that has been waiting a long time must not stay at
the back forever behind a stream of fresh high-priority items. :class:`AgingQueue`
therefore grows every queued item's *effective* priority the longer it waits::

    effective = base_priority + aging_rate * waited

Waiting tasks, and only waiting tasks, age. Once dispatched the boost is frozen;
if the task is later preempted it is re-enqueued with a fresh base priority but it
keeps the time it already spent queued, so its accumulated fairness is not lost.

Selection evaluates the effective priority against the *current* time rather than
caching it at enqueue: ordering can change over time once a capped item stops
gaining (``max_boost``), so an enqueue-time heap key would go stale.
"""
from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from typing import Any

__all__ = ('Entry', 'Queue', 'AgingQueue', 'effective_priority')


@dataclass(frozen=True)
class Entry:
    name: str
    base_priority: int
    enqueued_at: float
    waited: float = 0.0
    effective_priority: float = 0.0


def effective_priority(base_priority: float, waited: float, aging_rate: float,
                       *, max_boost: float | None = None) -> float:
    boost = aging_rate * max(0.0, waited)
    if max_boost is not None:
        boost = min(boost, max_boost)
    return base_priority + boost

class Queue:
    """Static priority queue (higher priority popped first); no aging."""

    def __init__(self):
        self.q = []
        self.seq = 0

    def push(self, item, priority):
        self.seq += 1
        heapq.heappush(self.q, (-priority, self.seq, item))

    def pop(self):
        return heapq.heappop(self.q)[2] if self.q else None


@dataclass
class _Item:
    name: Any
    base: int
    enqueued_at: float
    seq: int
    wait_offset: float = 0.0


class AgingQueue:
    """Priority queue whose items gain priority while they wait.

    ``aging_rate`` is priority points per unit of wait time. ``max_boost`` caps the
    boost; leave it ``None`` so every item is eventually selected (starvation
    freedom). Effective priorities are evaluated lazily at selection time, so a
    queue sitting idle needs no background timer.

    Wait time is pure queue time: ``waited = wait_offset + (now - enqueued_at)``.
    A preempted task re-enters with the waiting time it already accumulated in
    ``wait_offset``; the time it actually spent running never counts.
    """

    def __init__(self, *, aging_rate: float = 1.0, max_boost: float | None = None):
        if aging_rate < 0:
            raise ValueError('aging_rate must not be negative')
        if max_boost is not None and max_boost < 0:
            raise ValueError('max_boost must not be negative')
        self.aging_rate = aging_rate
        self.max_boost = max_boost
        self._items: dict[Any, _Item] = {}
        self._counter = itertools.count()

    def __len__(self) -> int:
        return len(self._items)

    @property
    def depth(self) -> int:
        return len(self._items)

    def enqueue(self, name: Any, priority: int, *, now: float,
                enqueued_at: float | None = None, wait_offset: float = 0.0) -> Entry:
        """Add an item.

        ``enqueued_at`` anchors the current waiting segment; ``wait_offset`` carries
        waiting time accumulated during earlier segments (e.g. before preemption).
        """
        if name in self._items:
            raise ValueError(f'{name!r} is already queued')
        if wait_offset < 0:
            raise ValueError('wait_offset must not be negative')
        item = _Item(name, int(priority),
                     float(now if enqueued_at is None else enqueued_at),
                     next(self._counter), float(wait_offset))
        self._items[name] = item
        return self._entry(item, now)

    # Backwards-compatible alias.
    push = enqueue

    def head(self, *, now: float) -> Entry | None:
        item = self._select(now)
        return None if item is None else self._entry(item, now)

    def peek(self, *, now: float) -> Entry | None:
        return self.head(now=now)

    def pop(self, *, now: float) -> Entry | None:
        item = self._select(now)
        if item is None:
            return None
        del self._items[item.name]
        return self._entry(item, now)

    def remove(self, name: Any) -> bool:
        """Withdraw an item (cancellation or an out-of-band dispatch)."""
        return self._items.pop(name, None) is not None

    def waited(self, name: str, *, now: float) -> float:
        item = self._items[name]
        return self._waited(item, now)

    def entries(self, *, now: float) -> list[Entry]:
        return sorted((self._entry(item, now) for item in self._items.values()),
                      key=lambda e: (-e.effective_priority, e.waited, e.name))

    def _select(self, now: float) -> _Item | None:
        if not self._items:
            return None
        return min(self._items.values(),
                   key=lambda it: (-self._effective(it, now), -self._waited(it, now), it.seq))

    @staticmethod
    def _waited(item: _Item, now: float) -> float:
        return item.wait_offset + max(0.0, now - item.enqueued_at)

    def _effective(self, item: _Item, now: float) -> float:
        return effective_priority(item.base, self._waited(item, now), self.aging_rate,
                                  max_boost=self.max_boost)

    def _entry(self, item: _Item, now: float) -> Entry:
        waited = self._waited(item, now)
        return Entry(item.name, item.base, item.enqueued_at, waited,
                     effective_priority(item.base, waited, self.aging_rate,
                                        max_boost=self.max_boost))
