"""Preemption, checkpoint/resume and priority-aging acceptance tests."""
import pytest

from aegisrover.mission.execution import MissionExecution, WaypointRunner
from aegisrover.mission.lifecycle import InvalidTransition, MissionService
from aegisrover.mission.preemption import (
    CheckpointError, CheckpointStore, JobAlreadyQueued, JobFinished,
    PreemptiveScheduler, StepJob, UnknownJob,
)
from aegisrover.mission.preemption_bridge import MissionExecutionJob
from aegisrover.storage.repository import Repository


class ManualClock:
    def __init__(self, start=0.0):
        self.now = start

    def __call__(self):
        return self.now

    def advance(self, dt):
        self.now += dt
        return self.now


# ------------------------------------------------------------------ checkpoints
def test_checkpoint_rejects_tampered_and_stale_state():
    store = CheckpointStore()
    store.save('j', 1, {'progress': 3})
    checkpoint = store.load('j')
    assert checkpoint.sequence == 1 and checkpoint.verify()
    checkpoint.state['progress'] = 99
    assert not checkpoint.verify()
    with pytest.raises(CheckpointError):
        store.load('j')
    with pytest.raises(CheckpointError):
        store.save('j', 1, {'progress': 4})  # sequence must advance


def test_step_job_restores_to_exact_progress():
    job = StepJob('j', total=5)
    for _ in range(3):
        job.step()
    state = job.suspend()
    fresh = StepJob('j', total=5)
    fresh.restore(state)
    fresh.step()
    assert fresh.progress == 4 and fresh.artifacts == [1, 2, 3, 4]
    with pytest.raises(CheckpointError):
        fresh.restore({'progress': 2, 'artifacts': [1]})  # mismatch


# ------------------------------------------------------------------------ aging
def test_aging_lifts_waiting_priority_and_caps_boost():
    clock = ManualClock()
    sched = PreemptiveScheduler(aging_rate=1.0, max_boost=10.0, clock=clock)
    sched.submit(StepJob('low', total=10), priority=0)
    clock.advance(4)
    assert sched.effective_priority('low') == pytest.approx(4.0)
    clock.advance(100)
    assert sched.effective_priority('low') == pytest.approx(10.0)  # capped


def test_running_job_stops_aging_but_keeps_wait_credit_when_requeued():
    clock = ManualClock()
    sched = PreemptiveScheduler(aging_rate=1.0, max_boost=None, clock=clock)
    sched.submit(StepJob('a', total=5), priority=0)
    clock.advance(3)
    sched.tick()  # dispatch at effective priority 3
    clock.advance(50)  # long run: running time must NOT age the job
    # urgent job preempts; a gets requeued with its 3s of banked wait only
    sched.submit(StepJob('urgent', total=1), priority=9)
    sched.tick()
    assert sched.get('a').state == 'queued'
    assert sched.wait_time('a') == pytest.approx(3.0)


def test_aging_is_fifo_among_equal_priorities():
    clock = ManualClock()
    sched = PreemptiveScheduler(aging_rate=1.0, clock=clock)
    sched.submit(StepJob('first', total=2), priority=1)
    clock.advance(1)
    sched.submit(StepJob('second', total=2), priority=1)
    assert sched.queue.peek(clock()).job_id == 'first'


# ------------------------------------------------------------------ preemption
def test_urgent_job_preempts_and_evicted_job_resumes_without_rework():
    clock = ManualClock()
    sched = PreemptiveScheduler(aging_rate=1.0, clock=clock)
    long = sched.submit(StepJob('long', total=10), priority=0)
    sched.tick()                      # long starts, progress 1
    assert sched.active_job().job_id == 'long'
    urgent = sched.submit(StepJob('urgent', total=2), priority=5)
    sched.tick()                      # long preempts at safe point, urgent steps
    assert sched.active_job().job_id == 'urgent'
    assert long.state == 'queued' and long.preemptions == 1
    assert sched.checkpoints.load('long').state == {'progress': 1, 'artifacts': [1]}
    kinds = [e.kind for e in sched.history()]
    assert 'preempt' in kinds and 'resume' not in kinds
    sched.tick()                      # urgent completes (progress 2)
    assert sched.active_job() is None
    stepped = sched.tick()            # long resumes from checkpoint
    assert stepped.job_id == 'long'
    assert stepped.job.progress == 2 and stepped.job.artifacts == [1, 2]
    resume_events = sched.history('resume')
    assert resume_events and resume_events[0].job_id == 'long'


def test_preemption_requires_priority_gap():
    clock = ManualClock()
    sched = PreemptiveScheduler(aging_rate=1.0, preempt_gap=2.0, clock=clock)
    sched.submit(StepJob('running', total=5), priority=0)
    sched.tick()
    sched.submit(StepJob('mild', total=2), priority=1)  # lead 1 < gap 2
    sched.tick()
    assert sched.active_job().job_id == 'running'
    assert sched.get('mild').state == 'queued'


def test_min_hold_defers_aging_takeover_but_not_genuine_urgency():
    clock = ManualClock()
    sched = PreemptiveScheduler(aging_rate=1.0, max_boost=100.0, min_hold=5.0,
                                preempt_gap=0.5, clock=clock)
    sched.submit(StepJob('a', total=10), priority=0)
    sched.tick()
    # A same-base-priority neighbour ages ahead, but the hold window blocks the
    # age-driven eviction.
    sched.submit(StepJob('b', total=10), priority=0)
    clock.advance(4)
    sched.tick()
    assert sched.active_job().job_id == 'a'
    clock.advance(2)  # past the hold window: the aged neighbour may now take over
    sched.tick()
    assert sched.active_job().job_id == 'b'
    assert sched.get('a').state == 'queued'


def test_genuinely_urgent_job_preempts_within_hold_window():
    clock = ManualClock()
    sched = PreemptiveScheduler(aging_rate=1.0, max_boost=100.0, min_hold=5.0,
                                preempt_gap=0.5, clock=clock)
    sched.submit(StepJob('a', total=10), priority=0)
    sched.tick()
    clock.advance(1)
    sched.submit(StepJob('urgent', total=2), priority=9)
    sched.tick()  # base-priority lead 9 > gap: hold window does not protect a
    assert sched.active_job().job_id == 'urgent'


def test_non_preemptible_job_defers_eviction_until_safe_point():
    clock = ManualClock()
    sched = PreemptiveScheduler(aging_rate=0.0, clock=clock)
    long = StepJob('long', total=5, allow_preempt=False)
    sched.submit(long, priority=0)
    sched.tick()
    sched.submit(StepJob('urgent', total=2), priority=9)
    sched.tick()  # long cannot yield during its critical step
    assert sched.active_job().job_id == 'long'
    assert any(e.kind == 'preempt_deferred' for e in sched.history())
    long.allow_preempt = True
    sched.tick()  # next tick the urgent job takes over
    assert sched.active_job().job_id == 'urgent'


def test_no_starvation_old_low_priority_beats_fresh_high_priority():
    clock = ManualClock()
    # aging 1/s, cap high enough that an old base-0 job overtakes a new base-5.
    sched = PreemptiveScheduler(aging_rate=1.0, max_boost=100.0, clock=clock)
    old = sched.submit(StepJob('old', total=8), priority=0)
    clock.advance(6)
    fresh = sched.submit(StepJob('fresh', total=8), priority=5)
    # old: 0 + 6 = 6 > fresh: 5 + 0
    assert sched.queue.peek(clock()).job_id == 'old'
    assert old.effective_priority(clock(), 1.0, 100.0) > \
        fresh.effective_priority(clock(), 1.0, 100.0)


def test_preempted_job_keeps_seniority_and_eventually_runs_again():
    clock = ManualClock()
    sched = PreemptiveScheduler(aging_rate=1.0, max_boost=100.0, clock=clock)
    a = sched.submit(StepJob('a', total=4), priority=0)
    sched.tick()                       # a: progress 1
    sched.submit(StepJob('u1', total=1), priority=10)
    sched.tick()                       # preempt a, run u1
    sched.tick()                       # u1 completes; a resumes (progress 2)
    sched.submit(StepJob('u2', total=1), priority=10)
    sched.tick()                       # preempt a again
    assert a.preemptions == 2
    sched.tick()                       # u2 completes
    sched.run_until_idle()
    assert a.state == 'completed' and a.job.artifacts == [1, 2, 3, 4]
    assert 'a' not in sched.checkpoints  # dropped on completion


def test_cancel_and_resubmit_guards():
    clock = ManualClock()
    sched = PreemptiveScheduler(clock=clock)
    sched.submit(StepJob('q', total=3), priority=0)
    sched.cancel('q')
    with pytest.raises(JobFinished):
        sched.submit(StepJob('q', total=3), priority=0)  # finished id not reused
    sched.submit(StepJob('r', total=3), priority=0)
    with pytest.raises(JobAlreadyQueued):
        sched.submit(StepJob('r', total=3), priority=0)  # still queued
    with pytest.raises(UnknownJob):
        sched.cancel('ghost')


def test_failed_step_marks_job_and_frees_executor():
    class Boom(StepJob):
        def step(self):
            raise RuntimeError('actuator fault')

    clock = ManualClock()
    sched = PreemptiveScheduler(clock=clock)
    sched.submit(Boom('boom', total=2), priority=0)
    with pytest.raises(RuntimeError):
        sched.tick()
    assert sched.get('boom').state == 'failed' and sched.active_job() is None


# ---------------------------------------------------------------- lifecycle
@pytest.fixture()
def repo():
    r = Repository(':memory:', clock=ManualClock(1000.0))
    yield r
    r.close()


def test_requeue_is_guarded_transition_and_releases_assignee(repo):
    service = MissionService(repo, clock=ManualClock())
    service.create('m1', [(0, 0), (5, 0)], priority=2)
    service.command('m1', 'queue')
    service.command('m1', 'assign', assignee='robot-1')
    service.command('m1', 'start')
    result = service.command('m1', 'requeue', reason='urgent insertion')
    assert result['applied'] is True
    mission = service.get('m1')
    assert mission.state == 'queued' and mission.assigned_to is None
    # Re-dispatch works without losing the record, and draft cannot requeue.
    service.command('m1', 'assign', assignee='robot-1')
    service.create('m2', [(0, 0)])
    with pytest.raises(InvalidTransition):
        service.command('m2', 'requeue')


# ------------------------------------------------------------- end-to-end bridge
class PositionPlayer:
    """Walk through waypoints one at a time for a single mission."""

    def __init__(self, waypoints, start=(0.0, 0.0)):
        self.points = list(waypoints)
        self.index = 0
        self.start = start

    def __call__(self):
        if self.index < len(self.points):
            pos = self.points[self.index]
            self.index += 1
            return pos
        return self.points[-1]


def make_mission_job(service, mission_id, waypoints, actor='robot-1'):
    runner = WaypointRunner(waypoints, tolerance=0.1)
    execution = MissionExecution(mission_id, runner)
    player = PositionPlayer(waypoints)
    return MissionExecutionJob(execution, service, player,
                               start_position=(0.0, 0.0), actor=actor)


def test_waypoint_mission_preempts_and_resumes_with_lifecycle_audit(repo):
    clock = ManualClock()
    service = MissionService(repo, clock=clock)
    waypoints = [(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]
    service.create('survey', waypoints, priority=0)
    service.command('survey', 'queue')
    service.command('survey', 'assign', assignee='robot-1')
    survey = make_mission_job(service, 'survey', waypoints)

    sched = PreemptiveScheduler(aging_rate=0.0, clock=clock)
    sched.submit(survey, priority=0)
    sched.tick()  # visits (1,0): runner index 1
    assert survey.execution.runner.index == 1

    # Emergency inspection arrives; its mission lifecycle is managed separately.
    emergency = StepJob('emergency', total=2)
    sched.submit(emergency, priority=9)
    sched.tick()  # survey suspends at safe point and requeues in the service
    assert service.get('survey').state == 'queued'
    assert service.get('survey').assigned_to is None
    assert sched.checkpoints.load('survey').state['runner']['index'] == 1
    assert sched.active_job().job_id == 'emergency'

    # Emergency finishes its work; survey is then re-assigned to the worker and
    # resumes without revisiting waypoint 1.
    sched.tick()  # emergency completes
    assert sched.active_job() is None
    service.command('survey', 'assign', assignee='robot-1')
    sched.tick()  # restore -> resume -> visits (2,0)
    assert survey.execution.runner.index == 2
    history = [h['command'] for h in service.get('survey').history]
    assert history.count('requeue') == 1
    assert 'suspend' in [e['event'] for e in survey.execution.events]
