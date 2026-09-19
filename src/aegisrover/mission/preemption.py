"""Cooperative preemption with checkpoint/resume and priority aging.

Two operational failures motivate this module. An urgent mission used to wait
behind a long-running mission because nothing could release the executor without
the operator cancelling the running one -- which discarded every partial result
and made it re-queue from scratch. And low-priority missions could wait forever
behind a steady stream of higher-priority submissions, because dispatch only
looked at the static ``priority`` field.

The design answers both:

* **Safe-point preemption.** A running *job* is asked to checkpoint at a safe
  point (``suspend``) before the executor is handed to a more urgent job. The
  checkpoint is stored with a sequence number and digest, so when the displaced
  job reaches the front again it resumes from exactly the state it gave up, not
  from zero. A job that reports ``preemptible is False`` cannot be evicted that
  tick -- progress is never taken at a moment where it would corrupt work.

* **Priority aging.** Effective priority is ``base + wait_credit`` while a job
  waits, capped at ``max_boost``. Every job therefore becomes the highest
  priority eventually (starvation freedom), but an urgent base priority still
  cuts in immediately without waiting. Preemption additionally requires a
  positive ``preempt_gap`` so two similar-priority jobs do not thrash the
  executor between every step.

Aging applies to *waiting* time, not running time: a job stops accruing boost
the moment it is dispatched, and resumes accruing (keeping the credit it earned
before) if it is later preempted back into the queue.
"""
from __future__ import annotations

import hashlib
import json
import itertools
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable

__all__ = (
    'JOB_STATES', 'PreemptionError', 'CheckpointError', 'UnknownJob',
    'JobAlreadyQueued', 'JobFinished',
    'PreemptibleJob', 'StepJob', 'ScheduledJob', 'CheckpointStore',
    'AgingQueue', 'PreemptiveScheduler', 'PreemptionEvent',
)

JOB_STATES = ('queued', 'running', 'completed', 'failed')


class PreemptionError(RuntimeError):
    pass


class CheckpointError(PreemptionError):
    pass


class UnknownJob(PreemptionError):
    def __init__(self, job_id: str):
        super().__init__(f'unknown job {job_id!r}')
        self.job_id = job_id


class JobAlreadyQueued(PreemptionError):
    def __init__(self, job_id: str):
        super().__init__(f'job {job_id!r} is already queued')
        self.job_id = job_id


class JobFinished(PreemptionError):
    def __init__(self, job_id: str):
        super().__init__(f'job {job_id!r} has already finished')
        self.job_id = job_id


@runtime_checkable
class PreemptibleJob(Protocol):
    """Unit of work the scheduler can suspend and resume.

    ``suspend`` is only ever called at a safe point chosen by the job. It must
    capture enough state for ``restore`` to continue without repeating or
    skipping committed work, and return ``False`` from ``preemptible`` if the
    current step cannot be interrupted safely.
    """

    job_id: str

    def step(self) -> None:
        """Advance the job by one incremental, checkpoint-able unit."""
        ...

    def done(self) -> bool: ...

    def preemptible(self) -> bool: ...

    def suspend(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of the resumable state."""
        ...

    def restore(self, state: dict[str, Any]) -> None:
        """Re-instate state previously produced by :meth:`suspend`."""
        ...


def _digest(state: dict[str, Any]) -> str:
    body = json.dumps(state, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(body).hexdigest()


@dataclass
class StepJob:
    """A simple counted workload, useful as the reference job and in tests.

    Each :meth:`step` performs one unit of ``work`` and appends its ordinal to
    ``artifacts``. Artifacts are the "intermediate results" that must survive a
    preemption: they live in the checkpoint, so after resume the job continues
    at ``progress + 1`` and never redoes a unit nor skips one.
    """

    job_id: str
    total: int
    progress: int = 0
    artifacts: list[int] = field(default_factory=list)
    allow_preempt: bool = True

    def step(self) -> None:
        if self.progress >= self.total:
            raise PreemptionError(f'{self.job_id}: stepping a finished job')
        self.progress += 1
        self.artifacts.append(self.progress)

    def done(self) -> bool:
        return self.progress >= self.total

    def preemptible(self) -> bool:
        return self.allow_preempt and not self.done()

    def suspend(self) -> dict[str, Any]:
        return {'progress': self.progress, 'artifacts': list(self.artifacts)}

    def restore(self, state: dict[str, Any]) -> None:
        progress = int(state['progress'])
        artifacts = list(state['artifacts'])
        if progress < 0 or progress > self.total:
            raise CheckpointError(f'{self.job_id}: checkpoint progress {progress} out of range')
        if len(artifacts) != progress:
            raise CheckpointError(f'{self.job_id}: checkpoint has {len(artifacts)} '
                                  f'artifacts for progress {progress}')
        self.progress = progress
        self.artifacts = artifacts


@dataclass
class Checkpoint:
    sequence: int
    state: dict[str, Any]
    digest: str

    def verify(self) -> bool:
        return self.digest == _digest(self.state)


class CheckpointStore:
    """Checkpoint storage keyed by job id, with sequence numbers and digests.

    A restore rejects a checkpoint whose state no longer matches its digest
    (tampering / corrupt persistence) or whose sequence went backwards -- the
    caller must never resume from an older snapshot than the latest one.
    """

    def __init__(self):
        self._items: dict[str, Checkpoint] = {}

    def save(self, job_id: str, sequence: int, state: dict[str, Any]) -> Checkpoint:
        previous = self._items.get(job_id)
        if previous is not None and sequence <= previous.sequence:
            raise CheckpointError(
                f'{job_id}: refusing to overwrite checkpoint seq {previous.sequence} '
                f'with older/equal seq {sequence}')
        checkpoint = Checkpoint(sequence=sequence, state=state, digest=_digest(state))
        self._items[job_id] = checkpoint
        return checkpoint

    def load(self, job_id: str) -> Checkpoint:
        try:
            checkpoint = self._items[job_id]
        except KeyError:
            raise CheckpointError(f'{job_id}: no checkpoint stored') from None
        if not checkpoint.verify():
            raise CheckpointError(f'{job_id}: checkpoint digest mismatch')
        return checkpoint

    def discard(self, job_id: str) -> None:
        self._items.pop(job_id, None)

    def __contains__(self, job_id: str) -> bool:
        return job_id in self._items

    def __len__(self) -> int:
        return len(self._items)


@dataclass
class ScheduledJob:
    """A job plus the bookkeeping the scheduler owns around it."""

    job: PreemptibleJob
    base_priority: int
    enqueued_at: float
    order: int
    wait_credit: float = 0.0
    state: str = 'queued'
    runs: int = 0
    preemptions: int = 0

    @property
    def job_id(self) -> str:
        return self.job.job_id

    def effective_priority(self, now: float, aging_rate: float, max_boost: float) -> float:
        """Static priority plus accumulated/accruing wait boost.

        Running jobs do not age; queued jobs age continuously, including wait
        time accumulated before an earlier dispatch (``wait_credit``).
        """
        boost = self.wait_credit
        if self.state == 'queued':
            boost += max(0.0, now - self.enqueued_at)
        boost *= aging_rate
        if max_boost is not None:
            boost = min(boost, max_boost)
        return float(self.base_priority) + boost

    def _requeue(self, now: float) -> None:
        """Move back to the queue.

        Wait time since the last dispatch is banked earlier by ``_dispatch`` /
        ``effective_priority`` accounting; here we only open a fresh queue stay.
        Adding ``now - enqueued_at`` again would count the running period as
        waiting and let preempted jobs age faster than genuinely queued ones.
        """
        if self.state == 'queued':
            return
        self.enqueued_at = now
        self.state = 'queued'


class AgingQueue:
    """Priority queue ordered by *effective* (aging) priority, FIFO as tiebreak.

    Effective priority changes with wall time, so a static heap ordered once at
    insertion goes stale. The candidate selection recomputes priorities over the
    waiting set on every :meth:`peek` -- fleet queues hold tens, not millions,
    of jobs, and the recompute is what lets an old low-priority job overtake a
    freshly arrived one the instant its aged priority crosses it.
    """

    def __init__(self, aging_rate: float = 1.0, max_boost: float | None = 32.0):
        if aging_rate < 0:
            raise ValueError('aging_rate must not be negative')
        if max_boost is not None and max_boost < 0:
            raise ValueError('max_boost must not be negative')
        self.aging_rate = aging_rate
        self.max_boost = max_boost
        self._waiting: dict[str, ScheduledJob] = {}

    def add(self, entry: ScheduledJob) -> None:
        if entry.job_id in self._waiting:
            raise JobAlreadyQueued(entry.job_id)
        self._waiting[entry.job_id] = entry

    def remove(self, job_id: str) -> ScheduledJob:
        try:
            return self._waiting.pop(job_id)
        except KeyError:
            raise UnknownJob(job_id) from None

    def peek(self, now: float) -> ScheduledJob | None:
        return self._best(now)

    def pop(self, now: float) -> ScheduledJob | None:
        best = self._best(now)
        if best is None:
            return None
        del self._waiting[best.job_id]
        return best

    def waiting(self) -> tuple[ScheduledJob, ...]:
        return tuple(self._waiting.values())

    def __len__(self) -> int:
        return len(self._waiting)

    def __contains__(self, job_id: str) -> bool:
        return job_id in self._waiting

    def _best(self, now: float) -> ScheduledJob | None:
        if not self._waiting:
            return None
        # Highest effective priority first; the earliest submission breaks ties
        # (the monotonic ``order`` keeps that comparison exact even when clocks
        # tie), so equal-priority jobs are strictly FIFO.
        return min(self._waiting.values(),
                   key=lambda e: (-e.effective_priority(now, self.aging_rate, self.max_boost),
                                  e.order))


@dataclass(frozen=True)
class PreemptionEvent:
    at: float
    kind: str
    job_id: str
    detail: dict[str, Any] = field(default_factory=dict)


class PreemptiveScheduler:
    """Single executor, cooperative preemption, aging priorities.

    Parameters
    ----------
    aging_rate:
        Effective-priority points gained per second spent waiting.
    max_boost:
        Cap on wait-derived boost so an ancient job cannot age without bound;
        ``None`` lifts the cap. Every job still reaches the cap eventually, so
        starvation freedom holds with the cap in place.
    preempt_gap:
        Minimum effective-priority lead a queued job must have over the running
        job to evict it. Prevents equal-priority neighbours from ping-ponging.
    min_hold:
        Minimum seconds a job keeps the executor per dispatch. Aging alone
        (which accrues only while waiting) would otherwise let a queued
        neighbour overtake the running job second by second and evict it in a
        leapfrog churn; the hold window bounds how often that can happen. A
        genuinely urgent job whose lead exceeds ``preempt_gap`` is *not* held
        back merely by the window -- only age-driven takeovers wait it out.
    """

    def __init__(self, *, aging_rate: float = 1.0, max_boost: float | None = 32.0,
                 preempt_gap: float = 0.0, min_hold: float = 0.0,
                 clock: Callable[[], float] = time.time,
                 checkpoint_store: CheckpointStore | None = None):
        if preempt_gap < 0:
            raise ValueError('preempt_gap must not be negative')
        if min_hold < 0:
            raise ValueError('min_hold must not be negative')
        self.aging_rate = aging_rate
        self.max_boost = max_boost
        self.preempt_gap = preempt_gap
        self.min_hold = min_hold
        self._clock = clock
        self.checkpoints = checkpoint_store or CheckpointStore()
        self.queue = AgingQueue(aging_rate=aging_rate, max_boost=max_boost)
        self._jobs: dict[str, ScheduledJob] = {}
        self._active: ScheduledJob | None = None
        self._dispatched_at: float | None = None
        self._seq = itertools.count(1)
        self._checkpoint_seq: dict[str, int] = {}
        self.events: list[PreemptionEvent] = []

    # -- admission -------------------------------------------------------------
    def submit(self, job: PreemptibleJob, priority: int = 0, *,
               now: float | None = None) -> ScheduledJob:
        now = self._now(now)
        existing = self._jobs.get(job.job_id)
        if existing is not None and existing.state in ('queued', 'running'):
            raise JobAlreadyQueued(job.job_id)
        if existing is not None and existing.state in ('completed', 'failed'):
            raise JobFinished(job.job_id)
        entry = ScheduledJob(job=job, base_priority=int(priority),
                             enqueued_at=now, order=next(self._seq))
        self._jobs[job.job_id] = entry
        self.queue.add(entry)
        self._emit(now, 'submit', job.job_id, {'priority': entry.base_priority})
        return entry

    def cancel(self, job_id: str, *, now: float | None = None) -> None:
        """Remove a queued job. Running jobs must be preempted or finish first."""
        now = self._now(now)
        entry = self._require(job_id)
        if entry.state != 'queued':
            raise PreemptionError(f'{job_id}: cannot cancel a {entry.state} job')
        self.queue.remove(job_id)
        entry.state = 'failed'
        self.checkpoints.discard(job_id)
        self._emit(now, 'cancel', job_id)

    # -- execution -------------------------------------------------------------
    def tick(self, *, now: float | None = None) -> ScheduledJob | None:
        """Run exactly one step of the job that should own the executor.

        Returns the entry that was stepped (still queued if it finished), or
        ``None`` when no job is runnable this tick.
        """
        now = self._now(now)
        candidate = self.queue.peek(now)

        if self._active is not None:
            if candidate is not None and self._should_preempt(self._active, candidate, now):
                if not self._preempt(self._active, now):
                    # Active job is at a non-preemptible point; it keeps the
                    # executor this tick and the urgent job waits one more step.
                    candidate = None
                else:
                    self._active = None
                    self._dispatched_at = None
                    candidate = self.queue.peek(now)
            if self._active is not None:
                return self._step_active(self._active, now)

        if candidate is None:
            return None
        entry = self.queue.pop(now)
        self._dispatch(entry, now)
        self._active = entry
        return self._step_active(entry, now)

    def run_until_idle(self, *, now: float | None = None,
                       max_steps: int | None = None) -> int:
        """Step until the queue and active job are drained. Returns step count."""
        steps = 0
        while True:
            previous = self.tick(now=now)
            if previous is None:
                return steps
            steps += 1
            if max_steps is not None and steps >= max_steps and not self.is_idle():
                raise PreemptionError('run_until_idle exceeded max_steps with jobs pending')

    # -- queries ---------------------------------------------------------------
    def is_idle(self) -> bool:
        return self._active is None and len(self.queue) == 0

    def active_job(self) -> ScheduledJob | None:
        return self._active

    def get(self, job_id: str) -> ScheduledJob:
        return self._require(job_id)

    def effective_priority(self, job_id: str, *, now: float | None = None) -> float:
        now = self._now(now)
        return self._require(job_id).effective_priority(now, self.aging_rate, self.max_boost)

    def wait_time(self, job_id: str, *, now: float | None = None) -> float:
        """Total seconds spent queued (banked + current queue stay)."""
        now = self._now(now)
        entry = self._require(job_id)
        waited = entry.wait_credit
        if entry.state == 'queued':
            waited += max(0.0, now - entry.enqueued_at)
        return waited

    def history(self, kind: str | None = None) -> tuple[PreemptionEvent, ...]:
        events = self.events if kind is None else [e for e in self.events if e.kind == kind]
        return tuple(events)

    # -- internals -------------------------------------------------------------
    def _dispatch(self, entry: ScheduledJob, now: float) -> None:
        # Bank wait time earned on this (possibly resumed) queue stay, then age
        # resets for the running period.
        entry.wait_credit += max(0.0, now - entry.enqueued_at)
        entry.enqueued_at = now
        entry.state = 'running'
        entry.runs += 1
        self._dispatched_at = now
        if entry.job_id in self.checkpoints:
            checkpoint = self.checkpoints.load(entry.job_id)
            entry.job.restore(checkpoint.state)
            self._emit(now, 'resume', entry.job_id,
                       {'checkpoint_seq': checkpoint.sequence,
                        'effective_priority': self._priority(entry, now)})
        else:
            self._emit(now, 'dispatch', entry.job_id,
                       {'effective_priority': self._priority(entry, now)})

    def _step_active(self, entry: ScheduledJob, now: float) -> ScheduledJob:
        try:
            entry.job.step()
        except Exception:
            entry.state = 'failed'
            self._active = None
            self._dispatched_at = None
            self._emit(now, 'fail', entry.job_id)
            raise
        self._emit(now, 'step', entry.job_id)
        if entry.job.done():
            entry.state = 'completed'
            self._active = None
            self._dispatched_at = None
            self.checkpoints.discard(entry.job_id)
            self._checkpoint_seq.pop(entry.job_id, None)
            self._emit(now, 'complete', entry.job_id)
        return entry

    def _should_preempt(self, active: ScheduledJob, candidate: ScheduledJob, now: float) -> bool:
        gap = self._priority(candidate, now) - self._priority(active, now)
        if gap <= self.preempt_gap:
            return False
        # A takeover driven only by aging waits out the hold window; a lead
        # larger than the job's static-priority span means genuine urgency and
        # cuts in immediately regardless of hold.
        base_gap = candidate.base_priority - active.base_priority
        held = (self._dispatched_at is not None
                and now - self._dispatched_at < self.min_hold)
        if held and base_gap <= self.preempt_gap:
            return False
        return True

    def _preempt(self, entry: ScheduledJob, now: float) -> bool:
        """Ask the active job to yield at a safe point. False means 'not now'."""
        if not entry.job.preemptible():
            self._emit(now, 'preempt_deferred', entry.job_id)
            return False
        state = entry.job.suspend()
        sequence = self._checkpoint_seq.get(entry.job_id, 0) + 1
        self.checkpoints.save(entry.job_id, sequence, state)
        self._checkpoint_seq[entry.job_id] = sequence
        entry.preemptions += 1
        entry._requeue(now)
        self.queue.add(entry)
        self._emit(now, 'preempt', entry.job_id,
                   {'checkpoint_seq': sequence,
                    'wait_credit': round(entry.wait_credit, 9)})
        return True

    def _priority(self, entry: ScheduledJob, now: float) -> float:
        return entry.effective_priority(now, self.aging_rate, self.max_boost)

    def _require(self, job_id: str) -> ScheduledJob:
        try:
            return self._jobs[job_id]
        except KeyError:
            raise UnknownJob(job_id) from None

    def _now(self, now: float | None) -> float:
        return self._clock() if now is None else now

    def _emit(self, now: float, kind: str, job_id: str, detail: dict | None = None) -> None:
        self.events.append(PreemptionEvent(at=now, kind=kind, job_id=job_id,
                                           detail=dict(detail or {})))
