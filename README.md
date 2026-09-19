# AegisRover

Autonomous mobile robot simulation, navigation, estimation, control, power, mission, protocol and runtime platform.

## Preemptive task scheduling

`aegisrover.mission.preemption` provides a single-executor scheduler for the case
where urgent work cannot wait behind a long-running task:

- **Cooperative preemption** — a task runs in steps; when a higher-priority task is
  ready the executor yields at a step boundary instead of killing the running job
  (`PreemptiveScheduler.tick`, `yield_now`).
- **Checkpoint / resume** — the preempted task captures its intermediate results to
  a digest-verified `CheckpointStore` and is rebuilt from that exact state when it
  resumes, so no progress is lost. Cancellation keeps the last checkpoint for
  recovery.
- **Wait-time priority aging** — queued tasks gain effective priority the longer
  they wait (`AgingQueue`, `effective_priority = base + aging_rate * waited`). With
  no boost cap an old task eventually overtakes fresh high-priority arrivals, so
  low-priority work cannot starve; running time is excluded from aging and the
  dispatched priority is frozen for the run.

