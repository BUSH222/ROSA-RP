from __future__ import annotations

from enum import StrEnum


class JobState(StrEnum):
    PENDING = "pending"
    RECORDING = "recording"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TRANSITIONS: dict[JobState, set[JobState]] = {
    JobState.PENDING: {JobState.RECORDING, JobState.CANCELLED, JobState.FAILED},
    JobState.RECORDING: {JobState.COMPLETED, JobState.FAILED, JobState.CANCELLED},
    JobState.COMPLETED: set(),
    JobState.FAILED: set(),
    JobState.CANCELLED: set(),
}


class IllegalTransition(Exception):
    pass


def validate_transition(current: str, new: str) -> None:
    try:
        current_s, new_s = JobState(current), JobState(new)
    except ValueError as e:
        raise IllegalTransition(f"Unknown state: {e}") from e
    if new_s not in _TRANSITIONS[current_s]:
        raise IllegalTransition(f"Cannot transition job from {current_s} to {new_s}")


def is_terminal(state: str) -> bool:
    return not _TRANSITIONS[JobState(state)]
