"""Minimal stage runner: each stage persists outputs and is skipped when cached."""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.common.logger import get_logger

log = get_logger(__name__)


@dataclass
class Stage:
    name: str
    group: str
    run: Callable[[], None]
    outputs: tuple[Path, ...] = ()

    def done(self) -> bool:
        return bool(self.outputs) and all(p.exists() for p in self.outputs)


def run_pipeline(
    stages: list[Stage],
    force: bool = False,
    groups: set[str] | None = None,
) -> None:
    for stage in stages:
        if groups and stage.group not in groups:
            log.info("skip (filtered): %s", stage.name)
            continue
        if stage.done() and not force:
            log.info("skip (cached): %s", stage.name)
            continue
        log.info("run: %s", stage.name)
        t0 = time.perf_counter()
        stage.run()
        log.info("done: %s (%.1fs)", stage.name, time.perf_counter() - t0)


def stage_status(stages: list[Stage]) -> list[tuple[str, bool, list[Path]]]:
    return [(s.name, s.done(), list(s.outputs)) for s in stages]
