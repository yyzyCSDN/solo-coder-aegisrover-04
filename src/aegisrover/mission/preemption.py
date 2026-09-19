"""Preemptive single-executor scheduling with checkpoint/resume and priority aging.

Three failure modes are addressed here.

* A long task used to block the executor until it finished: an urgent job had to
  wait, and killing the long job threw away all its progress. Instead the
  executor *yields* the running task when a higher-priority task is ready; the
  running task saves a checkpoint and returns to the queue.
* Checkpoint/resume must lose nothing. A resumed task is rebuilt from the exact
  state captured at preemption time; a checkpoint whose digest no longer matches
  is refused rather than silently restarting.
* Low-priority tasks must not starve behind a stream of fresh high-priority work.
  Queued tasks gain effective priority the longer they wait (see
  :mod:`aegisrover.mission.priorities`), so an old enough low-priority task will
  eventually overtake a newly arrived one.

The scheduler is a discrete, clock-injected policy object: ``tick(now)`` advances
it deterministically (the same sequence of submissions and ticks always yields
the same dispatch and preemption order). A task only ever does a *step* of work
per dispatch, so preemption latency is bounded by one step: the running task
finishes its current step, checkpoints, and yields — it is never killed mid-step.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from aegisrover.mission.priorities import AgingQueue, Entry, effective_priority
from aegisrover.storage.checkpoint import Checkpoint, CheckpointStore

__all__ = (
    'ResumableTask', 'StepOutcome', 'PreemptionRecord', 'PreemptionError',
    'PreemptiveScheduler', 'CountingTask', 'WaypointTask',
)

RUNNING = 'running'
WAITING = 'waiting'
PREEMPTED = 'preempted'
COMPLETED = 'completed'
CANCELLED = 'cancelled'
FAILED = 'failed'
ACTIVE_STATES = (WAITING, PREEMPTED, RUNNING)


class PreemptionError(RuntimeError):
    pass


class ResumableTask:
    """Task contract: cooperative steps plus checkpoint/resume of intermediate state."""

    def __init__(self, name: str, *, priority: int = 0):
        self.name = name
        self.base_priority = int(priority)

    # -- work ------------------------------------------------------------------
    def step(self) -> bool:
        """Advance the task by one unit of work.

        Return ``True`` when the task is complete. ``step`` is the granularity of
        preemption: once it returns, the scheduler may suspend the task, so it must
        leave the object in a state :meth:`checkpoint_state` can fully describe.
        """
        raise NotImplementedError

    # -- persistence -----------------------------------------------------------
    def checkpoint_state(self) -> Any:
        """Serializable snapshot of all intermediate results."""
        raise NotImplementedError

    @classmethod
    def restore(cls, name: str, state: Any, priority: int = 0) -> 'ResumableTask':
        """Rebuild the task from a checkpoint produced by :meth:`checkpoint_state`."""
        raise NotImplementedError

    def progress(self) -> float:
        return 0.0


@dataclass(frozen=True)
class StepOutcome:
    name: str
    completed: bool
    state: str
    progress: float


@dataclass(frozen=True)
class PreemptionRecord:
    """Who yielded the executor to whom, and where the checkpoint was saved."""

    preempted: str
    successor: str
    at: float
    reason: str
    sequence: int
    checkpoint: dict


@dataclass
class _Registration:
    name: str
    factory: Callable[..., ResumableTask]
    initial_state: Any
    priority: int
    enqueued_at: float
    wait_offset: float = 0.0
    task: ResumableTask | None = None
    state: str = WAITING
    dispatched_age: float = 0.0
    checkpoint_seq: int = 0
    preemptions: int = 0


class PreemptiveScheduler:
    """One executor, cooperative preemption, checkpoints and wait-time aging."""

    def __init__(self, *, clock=lambda: 0.0, aging_rate: float = 1.0,
                 max_boost: float | None = None, preempt_gap: float = 0.0,
                 store: CheckpointStore | None = None):
        if aging_rate < 0:
            raise ValueError('aging_rate must not be negative')
        if preempt_gap < 0:
            raise ValueError('preempt_gap must not be negative')
        self._clock = clock
        self.aging_rate = aging_rate
        self.max_boost = max_boost
        self.preempt_gap = preempt_gap
        self.store = store if store is not None else CheckpointStore()
        self._queue = AgingQueue(aging_rate=aging_rate, max_boost=max_boost)
        self._tasks: dict[str, _Registration] = {}
        self._running: str | None = None
        self.history: list[PreemptionRecord] = []

    # -- submission ------------------------------------------------------------
    def submit(self, task: ResumableTask, *, now: float | None = None,
               enqueued_at: float | None = None) -> str:
        now = self._time(now)
        if task.name in self._tasks:
            raise PreemptionError(f'{task.name!r} is already known')
        at = now if enqueued_at is None else float(enqueued_at)
        self._tasks[task.name] = _Registration(
            name=task.name, factory=type(task).restore,
            initial_state=task.checkpoint_state(), priority=task.base_priority,
            enqueued_at=at)
        self._queue.enqueue(task.name, task.base_priority, now=now, enqueued_at=at)
        return task.name

    def yield_now(self, *, now: float | None = None) -> PreemptionRecord | None:
        """Voluntarily yield the running task so another job can use the executor.

        The yielded task keeps its checkpoint. With no contender it simply keeps
        running, so yielding with an empty queue is a no-op.
        """
        now = self._time(now)
        if self._running is None:
            return None
        head = self._queue.head(now=now)
        if head is None:
            return None
        return self._suspend(now, successor=head.name, reason='yield')

    def cancel(self, name: str, *, now: float | None = None) -> bool:
        """Cancel a waiting, preempted or running task.

        Cancellation does *not* delete the checkpoint: an operator can inspect or
        recover the intermediate results afterwards. Returns False for an unknown
        or already finished task.
        """
        now = self._time(now)
        reg = self._tasks.get(name)
        if reg is None or reg.state not in ACTIVE_STATES:
            return False
        if reg.state == RUNNING:
            self._running = None  # free the executor without re-enqueuing
        else:
            self._queue.remove(name)
        reg.state = CANCELLED
        reg.task = None
        return True

    # -- execution -------------------------------------------------------------
    def tick(self, *, now: float | None = None) -> StepOutcome | None:
        """Run one dispatch cycle and return the step that executed, if any."""
        now = self._time(now)
        name = self._dispatch(now)
        if name is None:
            self._running = None
            return None
        reg = self._tasks[name]
        task = reg.task
        try:
            completed = bool(task.step())
        except BaseException:
            # A failed step must not poison the executor or strand the queue.
            reg.state = FAILED
            reg.task = None
            self._running = None
            raise
        if completed:
            reg.state = COMPLETED
            reg.task = None
            self._running = None
            self.store.discard(name)
            return StepOutcome(name, True, COMPLETED, task.progress())
        self._running = name
        return StepOutcome(name, False, RUNNING, task.progress())

    def run_idle(self, *, max_steps: int = 10_000) -> list[StepOutcome]:
        """Drive ticks until no active work remains (test/driver convenience)."""
        out: list[StepOutcome] = []
        for _ in range(max_steps):
            if self.is_idle:
                break
            outcome = self.tick()
            if outcome is not None:
                out.append(outcome)
        return out

    # -- queries ---------------------------------------------------------------
    @property
    def is_idle(self) -> bool:
        return self._running is None and self._queue.depth == 0

    @property
    def running(self) -> str | None:
        return self._running

    def state(self, name: str) -> str:
        return self._tasks[name].state

    def effective_priority(self, name: str, *, now: float | None = None) -> float:
        now = self._time(now)
        reg = self._tasks[name]
        if reg.state == RUNNING and self._running == name:
            return reg.priority + self.aging_rate * reg.dispatched_age  # frozen
        waited = reg.wait_offset + max(0.0, now - reg.enqueued_at)
        return effective_priority(reg.priority, waited, self.aging_rate,
                                  max_boost=self.max_boost)

    def queue(self, *, now: float | None = None) -> list[Entry]:
        return self._queue.entries(now=self._time(now))

    def preemptions(self, name: str) -> int:
        return self._tasks[name].preemptions

    def checkpoint(self, name: str) -> Checkpoint:
        return self.store.latest(name)

    def stats(self, *, now: float | None = None) -> dict:
        now = self._time(now)

        def waited(reg: _Registration) -> float:
            return reg.wait_offset + max(0.0, now - reg.enqueued_at)

        regs = self._tasks.values()
        return {
            'running': self._running,
            'waiting': self._queue.depth,
            'completed': sum(1 for r in regs if r.state == COMPLETED),
            'cancelled': sum(1 for r in regs if r.state == CANCELLED),
            'failed': sum(1 for r in regs if r.state == FAILED),
            'preemptions': len(self.history),
            'oldest_wait': round(max((waited(r) for r in regs
                                      if r.state in (WAITING, PREEMPTED)), default=0.0), 6),
        }

    # -- internals -------------------------------------------------------------
    def _dispatch(self, now: float) -> str | None:
        """Choose who runs this cycle, preempting the current holder if needed."""
        head = self._queue.head(now=now)
        if self._running is None:
            return None if head is None else self._activate(head.name, now)
        if head is None:
            return self._running
        incumbent = self._tasks[self._running]
        frozen_priority = incumbent.priority + self.aging_rate * incumbent.dispatched_age
        if head.effective_priority >= frozen_priority + self.preempt_gap:
            # _suspend activates the successor before returning the record.
            self._suspend(now, successor=head.name, reason='preempt')
            return self._running
        return incumbent.name

    def _activate(self, name: str, now: float) -> str:
        reg = self._tasks[name]
        entry = self._queue.pop(now=now)
        reg.task = self._materialize(reg)
        reg.state = RUNNING
        # Freeze the age (total pure wait time) at dispatch: the task does not age
        # while it runs. wait_offset accumulates queue time segment by segment, so
        # wait already rewarded in an earlier dispatch is never counted twice.
        reg.wait_offset = entry.waited
        reg.dispatched_age = entry.waited
        reg.enqueued_at = now
        self._running = name
        return name

    def _suspend(self, now: float, *, successor: str | None, reason: str) -> PreemptionRecord:
        incumbent_name = self._running
        incumbent = self._tasks[incumbent_name]
        task = incumbent.task
        sequence = incumbent.checkpoint_seq + 1
        checkpoint = Checkpoint.capture(incumbent_name, sequence,
                                        task.checkpoint_state(), created_at=now)
        # save() verifies the digest. Refuse to give up the executor when the
        # intermediate results cannot be captured: evicting would lose progress.
        self.store.save(checkpoint)
        incumbent.checkpoint_seq = sequence
        incumbent.preemptions += 1
        incumbent.task = None
        incumbent.state = PREEMPTED
        # Re-enter the queue carrying the age frozen at dispatch (the pure wait
        # time accumulated before this run); the time spent running is excluded.
        incumbent.wait_offset = incumbent.dispatched_age
        incumbent.enqueued_at = now
        self._queue.enqueue(incumbent_name, incumbent.priority, now=now,
                            enqueued_at=now, wait_offset=incumbent.wait_offset)
        record = PreemptionRecord(
            preempted=incumbent_name, successor=successor or '', at=now, reason=reason,
            sequence=sequence, checkpoint=checkpoint.to_dict())
        self.history.append(record)
        self._running = None
        # Hand the executor over. With no contender the incumbent resumes itself
        # (used by voluntary yields that race with a draining queue).
        self._activate(successor or incumbent_name, now)
        return record

    def _materialize(self, reg: _Registration) -> ResumableTask:
        if reg.checkpoint_seq:
            state = self.store.load(reg.name).state
        else:
            state = reg.initial_state
        return reg.factory(reg.name, state, reg.priority)

    def _time(self, now: float | None) -> float:
        return float(self._clock() if now is None else now)


class CountingTask(ResumableTask):
    """Demo task: count from 0 to ``target`` one increment per step."""

    def __init__(self, name: str, target: int, *, priority: int = 0, value: int = 0):
        super().__init__(name, priority=priority)
        if target <= 0:
            raise ValueError('target must be positive')
        self.target = int(target)
        self.value = int(value)

    def step(self) -> bool:
        self.value += 1
        return self.value >= self.target

    def progress(self) -> float:
        return min(1.0, self.value / self.target)

    def checkpoint_state(self) -> dict:
        return {'target': self.target, 'value': self.value}

    @classmethod
    def restore(cls, name: str, state: dict, priority: int = 0) -> 'CountingTask':
        return cls(name, state['target'], priority=priority, value=state['value'])


class WaypointTask(ResumableTask):
    """Resumable waypoint traversal: the reached-waypoint index is the checkpoint.

    Each step represents reaching the next waypoint. On resume the task continues
    from the captured index rather than driving the already-covered route again.
    """

    def __init__(self, name: str, waypoints, *, priority: int = 0, index: int = 0):
        super().__init__(name, priority=priority)
        self.waypoints = tuple(tuple(point) for point in waypoints)
        if not self.waypoints:
            raise ValueError('a waypoint task needs at least one waypoint')
        if not 0 <= index <= len(self.waypoints):
            raise ValueError('index out of range')
        self.index = int(index)

    def step(self) -> bool:
        if self.index < len(self.waypoints):
            self.index += 1
        return self.index >= len(self.waypoints)

    def progress(self) -> float:
        return self.index / len(self.waypoints)

    def checkpoint_state(self) -> dict:
        return {'waypoints': [list(w) for w in self.waypoints], 'index': self.index}

    @classmethod
    def restore(cls, name: str, state: dict, priority: int = 0) -> 'WaypointTask':
        return cls(name, state['waypoints'], priority=priority, index=state['index'])
