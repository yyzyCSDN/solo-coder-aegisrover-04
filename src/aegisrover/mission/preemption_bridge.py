"""Adapt waypoint missions to the preemptive scheduler.

:class:`PreemptiveScheduler` only knows the small :class:`PreemptibleJob`
protocol. This module binds that protocol to a real
:class:`~aegisrover.mission.execution.MissionExecution` (the movable work) and
:class:`~aegisrover.mission.lifecycle.MissionService` (the audited record), so
that:

* ``step`` advances one tick of the waypoint runner;
* preemption suspends the execution at a waypoint safe point **and** moves the
  mission record back to ``queued`` with a revision bump and an audit row;
* dispatch after a wait re-issues the mission to the worker and restores the
  checkpoint so visited waypoints are not traversed again.
"""
from __future__ import annotations

from typing import Any, Callable

from .execution import MissionExecution
from .lifecycle import MissionService

__all__ = ('MissionExecutionJob',)


class MissionExecutionJob:
    """Drive a :class:`MissionExecution` under a :class:`PreemptiveScheduler`.

    ``positions`` yields the robot position observed at each tick; the job is
    done once the underlying execution reports a terminal state.
    """

    def __init__(self, execution: MissionExecution, service: MissionService,
                 positions: Callable[[], tuple[float, float]], *,
                 start_position: tuple[float, float], actor: str = 'scheduler'):
        self.execution = execution
        self._service = service
        self._positions = positions
        self._actor = actor
        self._terminal = False
        self.execution.start(start_position)
        # Lifecycle: assigned -> running. The mission is expected to already be
        # assigned to this worker by claim_next/assign.
        self._service.command(self.job_id, 'start', actor=actor)

    @property
    def job_id(self) -> str:
        return self.execution.mission_id

    def step(self) -> None:
        state = self.execution.tick(self._positions())
        if state == 'completed':
            self._service.command(self.job_id, 'complete', actor=self._actor)
            self._terminal = True
        elif state == 'aborted':
            self._service.command(self.job_id, 'fail', actor=self._actor,
                                  reason=self.execution.abort_reason or 'aborted')
            self._terminal = True

    def done(self) -> bool:
        return self._terminal

    def preemptible(self) -> bool:
        return self.execution.preemptible()

    def suspend(self) -> dict[str, Any]:
        checkpoint = self.execution.suspend()
        # running -> queued, releases the assignee, revision + audit preserved.
        self._service.command(self.job_id, 'requeue', actor=self._actor,
                              reason='preempted by higher-priority mission')
        return checkpoint

    def restore(self, state: dict[str, Any]) -> None:
        # The mission comes back out of the queue; claim/assign/start must have
        # re-issued it to this worker before the scheduler steps it again.
        self.execution.restore(state)
        self.execution.resume()
        self._service.command(self.job_id, 'start', actor=self._actor)
