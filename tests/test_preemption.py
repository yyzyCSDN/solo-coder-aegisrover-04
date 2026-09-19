"""Acceptance tests for preemptive scheduling: yield without losing intermediate
results, checkpoint/resume across preemption, and wait-time priority aging."""
import pytest

from aegisrover.mission.preemption import (
    CountingTask, PreemptionError, PreemptiveScheduler, WaypointTask,
)
from aegisrover.mission.priorities import AgingQueue, Queue, effective_priority
from aegisrover.storage.checkpoint import Checkpoint, CheckpointError, CheckpointStore


class Clock:
    def __init__(self, now=0.0):
        self.now = now

    def __call__(self):
        return self.now

    def tick(self, delta):
        self.now += delta
        return self.now


# ------------------------------------------------------------- static priority queue
def test_static_queue_orders_by_priority_then_arrival():
    q = Queue()
    q.push('late-low', 1)
    q.push('high', 9)
    q.push('high-again', 9)
    assert [q.pop(), q.pop(), q.pop()] == ['high', 'high-again', 'late-low']
    assert q.pop() is None


# ------------------------------------------------------------------------ aging queue
def test_aging_grows_with_wait_time_and_carryover_offset():
    q = AgingQueue(aging_rate=1.0)
    q.enqueue('low', 0, now=0.0)
    # Effective priority grows the longer the item waits.
    assert q.head(now=0.0).effective_priority == 0.0
    assert q.head(now=4.0).effective_priority == 4.0
    assert q.waited('low', now=4.0) == pytest.approx(4.0)
    # Simulate preemption: the task ran for 100 s and re-enters carrying the 4 s
    # it had waited. The running time must never enter its effective priority.
    q.remove('low')
    q.enqueue('low', 0, now=104.0, enqueued_at=104.0, wait_offset=4.0)
    assert q.waited('low', now=104.0) == pytest.approx(4.0)
    assert q.head(now=105.0).effective_priority == 5.0  # offset 4 + 1 more second


def test_aging_fresh_high_priority_beats_unaged_low():
    q = AgingQueue(aging_rate=1.0)
    q.enqueue('low', 0, now=0.0)
    q.enqueue('high', 5, now=1.0)
    assert q.head(now=1.0).name == 'high'


def test_aging_tie_breaks_to_longer_waiting_then_fifo():
    q = AgingQueue(aging_rate=1.0)
    q.enqueue('a', 0, now=0.0)
    q.enqueue('b', 0, now=1.0)
    head = q.pop(now=5.0)
    assert head.name == 'a' and head.waited == 5.0


def test_aging_caps_boost_and_rejects_bad_config():
    q = AgingQueue(aging_rate=2.0, max_boost=4.0)
    q.enqueue('x', 1, now=0.0)
    assert q.head(now=100.0).effective_priority == 5.0
    with pytest.raises(ValueError):
        AgingQueue(aging_rate=-1)
    with pytest.raises(ValueError):
        AgingQueue(max_boost=-1)


def test_aging_wait_offset_excludes_running_time():
    # A task that waited 3 s, ran 100 s and re-enters must not inherit the run time.
    q = AgingQueue(aging_rate=1.0)
    q.enqueue('p', 0, now=103.0, enqueued_at=103.0, wait_offset=3.0)
    assert q.head(now=103.0).waited == 3.0
    assert q.head(now=104.0).effective_priority == 4.0


def test_aging_queue_remove_and_duplicate_guard():
    q = AgingQueue()
    q.enqueue('a', 0, now=0.0)
    with pytest.raises(ValueError):
        q.enqueue('a', 1, now=0.0)
    assert q.remove('a') is True
    assert q.remove('a') is False
    assert q.depth == 0


# ------------------------------------------------------------------------- checkpoints
def test_checkpoint_store_versions_and_verifies():
    store = CheckpointStore()
    store.save(Checkpoint.capture('t', 1, {'step': 1}, created_at=0.0))
    store.save(Checkpoint.capture('t', 2, {'step': 2}, created_at=1.0))
    loaded = store.load('t')
    assert loaded.sequence == 2 and loaded.state == {'step': 2}
    with pytest.raises(CheckpointError):
        store.save(Checkpoint.capture('t', 2, {'step': 9}, created_at=2.0))
    assert store.names() == ('t',)


def test_tampered_checkpoint_is_refused():
    store = CheckpointStore()
    checkpoint = Checkpoint.capture('t', 1, {'step': 1})
    tampered = Checkpoint.from_dict({**checkpoint.to_dict(), 'state': {'step': 99}})
    with pytest.raises(CheckpointError):
        store.save(tampered)


# ------------------------------------------------------------- preemption + checkpoint
def test_urgent_task_preempts_and_long_task_resumes_from_checkpoint():
    clock = Clock(0.0)
    sch = PreemptiveScheduler(clock=clock, aging_rate=0.0)
    sch.submit(CountingTask('long', target=100), now=0.0)

    sch.tick(now=0.0)
    assert sch.running == 'long'
    # The long task does several steps of real work...
    for _ in range(9):
        sch.tick(now=0.0)
    assert sch.state('long') == 'running'

    # ...then an urgent job arrives. The executor yields at the step boundary.
    sch.submit(CountingTask('urgent', target=2, priority=10), now=1.0)
    outcome = sch.tick(now=1.0)
    assert outcome.name == 'urgent'
    assert sch.state('long') == 'preempted'

    record = sch.history[-1]
    assert (record.preempted, record.successor, record.reason) == ('long', 'urgent', 'preempt')
    saved = sch.checkpoint('long')
    assert saved.state == {'target': 100, 'value': 10}  # nothing lost

    # Finish the urgent job; the long job resumes from 10, not from 0.
    sch.tick(now=2.0)  # urgent step -> completes (target 2)
    resumed = sch.tick(now=2.0)
    assert resumed.name == 'long' and resumed.progress == pytest.approx(11 / 100)
    assert sch.state('urgent') == 'completed'
    assert sch.preemptions('long') == 1


def test_voluntary_yield_hands_executor_over_and_preserves_progress():
    clock = Clock(0.0)
    sch = PreemptiveScheduler(clock=clock, aging_rate=0.0)
    sch.submit(CountingTask('a', target=50), now=0.0)
    sch.tick(now=0.0)
    for _ in range(4):
        sch.tick(now=0.0)
    sch.submit(CountingTask('b', target=1, priority=2), now=1.0)

    record = sch.yield_now(now=1.0)
    assert record is not None and record.preempted == 'a' and record.successor == 'b'
    assert sch.running == 'b'
    sch.tick(now=1.0)  # b completes
    next_outcome = sch.tick(now=2.0)
    assert next_outcome.name == 'a' and next_outcome.progress == pytest.approx(6 / 50)

    # Yielding with nobody waiting is a no-op.
    assert sch.yield_now(now=2.0) is None


def test_waypoint_task_resumes_at_checkpointed_index():
    clock = Clock(0.0)
    sch = PreemptiveScheduler(clock=clock, aging_rate=0.0)
    route = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
    sch.submit(WaypointTask('patrol', route), now=0.0)
    sch.tick(now=0.0)
    sch.tick(now=0.0)  # reached waypoint index 2
    sch.submit(CountingTask('alarm', target=1, priority=3), now=1.0)
    sch.tick(now=1.0)  # alarm preempts patrol
    assert sch.checkpoint('patrol').state['index'] == 2
    sch.tick(now=2.0)  # alarm completes; next dispatch activates patrol -> index 3
    final = sch.tick(now=3.0)  # patrol advances to index 4 and completes
    assert final.name == 'patrol' and final.completed


def test_preemption_only_above_priority_gap_avoids_thrashing():
    sch = PreemptiveScheduler(aging_rate=0.0, preempt_gap=2.0)
    sch.submit(CountingTask('runner', target=10, priority=5), now=0.0)
    sch.tick(now=0.0)
    # Equal or marginally higher priority must not preempt the running task.
    sch.submit(CountingTask('same', target=1, priority=5), now=0.0)
    sch.submit(CountingTask('slightly', target=1, priority=6), now=0.0)
    assert sch.tick(now=0.0).name == 'runner'
    sch.submit(CountingTask('clearly', target=1, priority=8), now=0.0)
    assert sch.tick(now=0.0).name == 'clearly'


def test_aging_guarantees_old_low_priority_eventually_runs():
    clock = Clock(0.0)
    sch = PreemptiveScheduler(clock=clock, aging_rate=1.0)
    # 'low' has already been queued for 5 s before anything dispatches it: that is
    # exactly the "stuck waiting behind a busy executor" situation aging addresses.
    sch.submit(CountingTask('low', target=100, priority=0), now=5.0, enqueued_at=0.0)
    sch.tick(now=5.0)
    assert sch.effective_priority('low', now=5.0) == pytest.approx(5.0)  # frozen at 5

    # A fresh p=4 urgent job arrives. It cannot preempt: low is already frozen at 5
    # purely because it waited five seconds, even though its static priority is 0.
    sch.submit(CountingTask('hi', target=1, priority=4), now=5.0)
    assert sch.tick(now=5.0).name == 'low'
    assert sch.state('hi') == 'waiting'
    assert sch.history == []  # no preemption: the aged task was not pushed aside

    # A genuinely higher p=6 emergency still can preempt; low keeps its 5 s age.
    sch.submit(CountingTask('sos', target=1, priority=6), now=6.0)
    assert sch.tick(now=6.0).name == 'sos'
    assert sch.checkpoint('low').state['value'] == 2
    assert sch.effective_priority('low', now=6.0) == pytest.approx(5.0)  # carries age
    sch.tick(now=6.0)  # sos completes; low resumes from its checkpoint
    assert sch.running == 'low'
    assert any(r.successor == 'sos' and r.preempted == 'low' for r in sch.history)


def test_unbounded_aging_eventually_overtakes_any_fresh_priority():
    q = AgingQueue(aging_rate=1.0)  # no max_boost -> starvation freedom
    q.enqueue('old', 0, now=0.0, wait_offset=1000.0)
    q.enqueue('fresh', 999, now=1000.0)
    assert q.head(now=1000.0).name == 'old'  # 1000 vs 999


def test_non_unit_aging_rate_scales_both_queue_and_frozen_priority():
    sch = PreemptiveScheduler(aging_rate=2.0)
    # 5 s in the queue at rate 2 lifts a p=0 task to an effective priority of 10.
    sch.submit(CountingTask('old', target=5), now=5.0, enqueued_at=0.0)
    sch.tick(now=5.0)
    assert sch.effective_priority('old', now=5.0) == pytest.approx(10.0)
    sch.submit(CountingTask('urgent', target=1, priority=9), now=5.0)
    assert sch.tick(now=5.0).name == 'old'  # frozen 10 beats fresh 9
    sch.submit(CountingTask('emergency', target=1, priority=11), now=5.0)
    assert sch.tick(now=5.0).name == 'emergency'  # 11 beats the frozen 10


def test_cancel_frees_executor_but_keeps_checkpoint():
    sch = PreemptiveScheduler(aging_rate=0.0)
    sch.submit(CountingTask('a', target=10), now=0.0)
    sch.tick(now=0.0)
    for _ in range(3):
        sch.tick(now=0.0)
    sch.submit(CountingTask('b', target=1, priority=9), now=1.0)
    sch.tick(now=1.0)  # a preempted, checkpoint at value 4
    assert sch.cancel('a') is True
    assert sch.state('a') == 'cancelled'
    checkpoint = sch.checkpoint('a')  # intermediate result still recoverable
    assert checkpoint.state['value'] == 4
    sch.tick(now=2.0)
    assert sch.state('b') == 'completed'
    # Cancelling an unknown/finished task is not an error.
    assert sch.cancel('a') is False
    assert sch.cancel('ghost') is False


def test_completed_task_drops_checkpoint_and_scheduler_goes_idle():
    sch = PreemptiveScheduler(aging_rate=0.0)
    sch.submit(CountingTask('a', target=2), now=0.0)
    sch.tick(now=0.0)
    sch.tick(now=0.0)
    assert sch.is_idle
    with pytest.raises(CheckpointError):
        sch.checkpoint('a')
    assert sch.stats()['completed'] == 1


def test_duplicate_submission_and_step_failure():
    sch = PreemptiveScheduler(aging_rate=0.0)
    sch.submit(CountingTask('a', target=3), now=0.0)
    with pytest.raises(PreemptionError):
        sch.submit(CountingTask('a', target=3), now=0.0)

    class Boom(CountingTask):
        def step(self):
            raise RuntimeError('actuator offline')

    sch2 = PreemptiveScheduler(aging_rate=0.0)
    sch2.submit(Boom('x', target=3), now=0.0)
    with pytest.raises(RuntimeError):
        sch2.tick(now=0.0)
    assert sch2.state('x') == 'failed' and sch2.running is None


def test_effective_priority_helper():
    assert effective_priority(2, 3, 1.0) == pytest.approx(5.0)
    assert effective_priority(2, 3, 1.0, max_boost=2.0) == pytest.approx(4.0)
