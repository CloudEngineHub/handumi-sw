from __future__ import annotations

import os

import pytest

from handumi.dataset.parallel import (
    MIN_EPISODES_PER_JOB,
    describe_jobs,
    iter_chunks,
    map_episodes,
    resolve_jobs,
)


def _setup(**kwargs):
    return {"scale": kwargs.get("scale", 1)}


def _task(episode: int, state: dict) -> int:
    return episode * state["scale"]


def test_sequential_path_runs_in_the_calling_process() -> None:
    """`jobs=1` is the loop the command has always run, not a pool of one.

    Keeping it in-process is what makes the sequential result impossible to
    drift from by accident.
    """
    seen: list[int] = []
    results = map_episodes(
        [3, 1, 2],
        setup=_setup,
        setup_kwargs={"scale": 10},
        task=_task,
        jobs=1,
        on_result=lambda episode, _value: seen.append(episode),
    )
    assert results == [30, 10, 20]
    # Order follows the episodes as given, not their numeric order.
    assert seen == [3, 1, 2]


def test_jobs_default_leaves_room_for_the_solver_threads() -> None:
    cores = os.cpu_count() or 1
    resolved = resolve_jobs(None, 1000)
    assert resolved == max(1, cores // 3)


def test_jobs_never_exceed_the_work_or_the_machine() -> None:
    cores = os.cpu_count() or 1
    assert resolve_jobs(1000, 1000) == cores
    assert resolve_jobs(8, 40) <= 8


def test_small_datasets_stay_sequential() -> None:
    """Each worker pays a JIT compilation before its first episode."""
    assert resolve_jobs(8, 1) == 1
    assert resolve_jobs(8, MIN_EPISODES_PER_JOB - 1) == 1
    assert resolve_jobs(8, MIN_EPISODES_PER_JOB * 2) == 2


def test_no_episodes_needs_no_workers() -> None:
    assert resolve_jobs(8, 0) == 1


def test_jobs_must_be_positive() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        resolve_jobs(0, 10)


def test_describe_jobs_says_what_will_happen() -> None:
    assert describe_jobs(1, 12) == "12 episodes, one at a time"
    assert describe_jobs(4, 12) == "12 episodes across 4 workers"


def test_chunks_cover_every_episode_in_order() -> None:
    assert iter_chunks(range(7), 3) == [[0, 1, 2], [3, 4, 5], [6]]
    assert iter_chunks([], 3) == []
    with pytest.raises(ValueError, match="at least 1"):
        iter_chunks([1], 0)
