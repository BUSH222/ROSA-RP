from __future__ import annotations

import asyncio
import inspect
import logging
from collections import defaultdict
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum

from core.models import Job

logger = logging.getLogger(__name__)

HookFn = Callable[[Job, asyncio.Event], Awaitable[None] | None]


class Event(StrEnum):
    OBSERVATION_START = "observation_start"
    POST_OBSERVATION = "post_observation"


@dataclass
class _Registration:
    fn: HookFn
    applies_to: frozenset[int] | None  # None = every satellite

    def matches(self, norad_id: int) -> bool:
        return self.applies_to is None or norad_id in self.applies_to


class HookRegistry:
    def __init__(self):
        self._hooks: dict[Event, list[_Registration]] = defaultdict(list)

    def register(self, event: Event, fn: HookFn, applies_to: Iterable[int] | None = None):
        frozen = frozenset(applies_to) if applies_to is not None else None
        self._hooks[event].append(_Registration(fn=fn, applies_to=frozen))
        return fn

    def on(self, event: Event, applies_to: Iterable[int] | None = None):
        """@registry.on(Event.POST_OBSERVATION, applies_to={59051, 57166})"""

        def wrapper(fn):
            self.register(event, fn, applies_to=applies_to)
            return fn

        return wrapper

    def emit(self, event: Event, job: Job, stop_event: asyncio.Event) -> list[asyncio.Task]:
        applicable = [r for r in self._hooks[event] if r.matches(job.norad_id)]
        return [asyncio.create_task(self._run_one(r.fn, job, stop_event)) for r in applicable]

    @staticmethod
    async def _run_one(fn: HookFn, job: Job, stop_event: asyncio.Event):
        try:
            if inspect.iscoroutinefunction(fn):
                await fn(job, stop_event)
            else:
                await asyncio.to_thread(fn, job, stop_event)
        except Exception:
            logger.exception("Hook %s raised for job %s", getattr(fn, "__name__", fn), job.id)


registry = HookRegistry()
