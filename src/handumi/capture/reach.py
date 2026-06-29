"""Reach-feasibility features used during HandUMI recording."""

from __future__ import annotations

import numpy as np

REACH_BUDGETS_M = {
    "piper": 0.45,
    "openarm": 0.55,
}


def empty_reach_features(*, feasible: bool = False) -> dict:
    value = np.array([1 if feasible else 0], dtype=np.int64)
    features = {
        "observation.reach.any_episode_feasible": value.copy(),
    }
    for robot in REACH_BUDGETS_M:
        features[f"observation.reach.{robot}_left_ratio"] = np.zeros((1,), dtype=np.float32)
        features[f"observation.reach.{robot}_right_ratio"] = np.zeros((1,), dtype=np.float32)
        features[f"observation.reach.{robot}_max_ratio"] = np.zeros((1,), dtype=np.float32)
        features[f"observation.reach.{robot}_frame_feasible"] = value.copy()
        features[f"observation.reach.{robot}_episode_feasible"] = value.copy()
    return features


def compute_reach_features(
    pico_frame: dict,
    anchor_l: np.ndarray,
    anchor_r: np.ndarray,
) -> tuple[dict, dict]:
    left = np.asarray(pico_frame["observation.pico.left_controller_pose"], dtype=np.float32)[:3]
    right = np.asarray(pico_frame["observation.pico.right_controller_pose"], dtype=np.float32)[:3]
    disp_l = float(np.linalg.norm(left - anchor_l))
    disp_r = float(np.linalg.norm(right - anchor_r))

    features: dict = {}
    metrics: dict = {}
    for robot, budget in REACH_BUDGETS_M.items():
        left_ratio = disp_l / budget
        right_ratio = disp_r / budget
        max_ratio = max(left_ratio, right_ratio)
        feasible = max_ratio <= 1.0
        metrics[robot] = {
            "left_ratio": left_ratio,
            "right_ratio": right_ratio,
            "max_ratio": max_ratio,
            "feasible": feasible,
        }
        features[f"observation.reach.{robot}_left_ratio"] = np.array([left_ratio], dtype=np.float32)
        features[f"observation.reach.{robot}_right_ratio"] = np.array([right_ratio], dtype=np.float32)
        features[f"observation.reach.{robot}_max_ratio"] = np.array([max_ratio], dtype=np.float32)
        features[f"observation.reach.{robot}_frame_feasible"] = np.array(
            [1 if feasible else 0], dtype=np.int64
        )
        features[f"observation.reach.{robot}_episode_feasible"] = np.zeros((1,), dtype=np.int64)

    features["observation.reach.any_episode_feasible"] = np.zeros((1,), dtype=np.int64)
    return features, metrics


def update_episode_reach_flags(dataset, *, save_unreachable: bool) -> tuple[bool, dict[str, bool]]:
    buf = dataset.writer.episode_buffer
    size = int(buf["size"])
    episode_feasible: dict[str, bool] = {}
    for robot in REACH_BUDGETS_M:
        frame_key = f"observation.reach.{robot}_frame_feasible"
        ep_key = f"observation.reach.{robot}_episode_feasible"
        frame_values = [int(np.asarray(v).reshape(-1)[0]) for v in buf[frame_key]]
        feasible = bool(size > 0 and all(frame_values))
        episode_feasible[robot] = feasible
        buf[ep_key] = [np.array([1 if feasible else 0], dtype=np.int64) for _ in range(size)]

    any_feasible = any(episode_feasible.values())
    buf["observation.reach.any_episode_feasible"] = [
        np.array([1 if any_feasible else 0], dtype=np.int64) for _ in range(size)
    ]
    return any_feasible or save_unreachable, episode_feasible

