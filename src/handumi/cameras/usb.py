"""USB camera setup helpers for HandUMI recording."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import yaml

log = logging.getLogger("handumi.record")


def build_camera_specs(
    cam_ids: list[int | str],
    *,
    laptop_camera: bool,
    laptop_cam_id: int,
    laptop_cam_name: str,
) -> tuple[list[dict[str, Any]], str | None]:
    names = ["left_wrist", "right_wrist"]
    specs = []
    for i, cam_id in enumerate(cam_ids):
        name = names[i] if i < len(names) else f"cam_{i}"
        specs.append({"id": cam_id, "name": name, "is_laptop": False})
    resolved_laptop_name = laptop_cam_name if laptop_camera else None
    if laptop_camera:
        for spec in specs:
            if spec["name"] == laptop_cam_name:
                spec["is_laptop"] = True
                spec["id"] = laptop_cam_id
                break
        else:
            specs.append(
                {"id": laptop_cam_id, "name": laptop_cam_name, "is_laptop": True}
            )
    return specs, resolved_laptop_name


def resolve_camera_ids(
    cam_ids: list[int | str] | None,
    camera_config: Path,
) -> list[int | str]:
    if cam_ids is not None:
        return cam_ids
    if not camera_config.exists():
        return [0, 2]
    with camera_config.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    return [
        _read_camera_value(data, "left_wrist", 0),
        _read_camera_value(data, "right_wrist", 2),
    ]


def connect_cameras(
    camera_specs: list[dict[str, Any]],
    *,
    fps: int,
    width: int,
    height: int,
    zero_non_laptop: bool,
) -> list:
    from lerobot.cameras.opencv import OpenCVCamera
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig

    cameras: list = []
    for spec in camera_specs:
        cam_id = spec["id"]
        name = spec["name"]
        should_zero = zero_non_laptop and not spec["is_laptop"]
        if should_zero:
            cameras.append(None)
            log.info(f"Camera '{name}' will be zero-filled.")
        else:
            cfg = OpenCVCameraConfig(
                index_or_path=cam_id,
                fps=fps,
                width=width,
                height=height,
            )
            cam = OpenCVCamera(cfg)
            cam.connect()
            cameras.append(cam)
            label = " laptop overlay" if spec["is_laptop"] else ""
            log.info(f"Camera '{name}' (index {cam_id}) connected.{label}")
    return cameras


def read_camera_frames(
    cameras: list,
    cam_names: list[str],
    *,
    width: int,
    height: int,
) -> dict:
    frames: dict = {}
    for cam, name in zip(cameras, cam_names):
        frame = (
            np.zeros((height, width, 3), dtype=np.uint8)
            if cam is None
            else cam.async_read()
        )
        frames[f"observation.images.{name}"] = frame
    return frames


def disconnect_cameras(cameras: list) -> None:
    for cam in cameras:
        try:
            cam.disconnect()
        except Exception:
            pass


def _read_camera_value(data: dict[str, Any], key: str, default: int) -> int | str:
    section = data.get(key) or {}
    value = section.get("index_or_path", default)
    if isinstance(value, int):
        return value
    text = str(value)
    return int(text) if text.isdigit() else text
