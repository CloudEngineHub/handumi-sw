"""Live cameras and calibrated gripper openings in Rerun, without VR tracking."""

from __future__ import annotations

import argparse
import logging
import time
from contextlib import ExitStack
from pathlib import Path

from handumi.cameras import build_camera_specs, make_camera_device, resolve_camera_ids
from handumi.config import DEFAULT_RIG_CONFIG
from handumi.feetech.calibration import assert_calibrated, load_config
from handumi.feetech.gripper import FeetechGripperPair
from handumi.visualization import LEFT_COLOR, RIGHT_COLOR


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rig-config", type=Path, default=DEFAULT_RIG_CONFIG)
    parser.add_argument(
        "--cameras",
        default="left_wrist,right_wrist",
        help="Comma-separated camera names from the rig config.",
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--duration-s",
        type=float,
        default=0.0,
        help="Stop after this many seconds; 0 runs until Ctrl+C.",
    )
    args = parser.parse_args(argv)
    args.cameras = [name.strip() for name in args.cameras.split(",") if name.strip()]
    if not args.cameras or args.fps <= 0 or args.duration_s < 0:
        parser.error(
            "Select at least one camera, positive FPS and nonnegative duration."
        )
    return args


def run(args: argparse.Namespace) -> None:
    import rerun as rr
    import rerun.blueprint as rrb

    config = load_config(args.rig_config)
    assert_calibrated(config)
    ids = resolve_camera_ids(None, args.rig_config, camera_names=args.cameras)
    if len(set(ids)) != len(ids):
        raise SystemExit(
            "Selected cameras must use distinct devices in the rig config."
        )
    specs, _ = build_camera_specs(
        ids,
        camera_names=args.cameras,
        rig_config=args.rig_config,
        laptop_camera=False,
        laptop_cam_id=0,
        laptop_cam_name="laptop",
        default_width=640,
        default_height=480,
        default_fps=args.fps,
    )
    with ExitStack() as stack:
        grippers = stack.enter_context(FeetechGripperPair(config))
        cameras = []
        for spec in specs:
            camera = make_camera_device(spec)
            stack.callback(camera.disconnect)
            camera.connect()
            cameras.append(camera)

        rr.init("handumi_visualize", spawn=True)
        stack.callback(rr.rerun_shutdown)
        width_paths = [
            f"/observation.feetech.{side}_width_mm" for side in ("left", "right")
        ]
        for path, color in zip(width_paths, (LEFT_COLOR, RIGHT_COLOR), strict=True):
            rr.log(
                path, rr.SeriesLines(colors=[[*color, 255]], widths=[2.5]), static=True
            )
        rr.send_blueprint(
            rrb.Blueprint(
                rrb.Vertical(
                    rrb.Horizontal(
                        *[
                            rrb.Spatial2DView(
                                origin=f"/observation.images.{name}", name=name
                            )
                            for name in args.cameras
                        ]
                    ),
                    rrb.TimeSeriesView(
                        origin="/",
                        contents=width_paths,
                        name="gripper_width_mm",
                        axis_y=rrb.ScalarAxis(
                            range=(
                                0.0,
                                max(
                                    float(config.left.max_width_mm or 0),
                                    float(config.right.max_width_mm or 0),
                                )
                                * 1.1,
                            )
                        ),
                    ),
                    row_shares=[3, 2],
                ),
                rrb.BlueprintPanel(state="collapsed"),
                rrb.SelectionPanel(state="collapsed"),
                rrb.TimePanel(state="collapsed"),
            ),
            make_active=True,
            make_default=True,
        )
        print("Rerun: cameras and gripper openings (mm). Ctrl+C to stop.", flush=True)
        started = time.monotonic()
        while not args.duration_s or time.monotonic() - started < args.duration_s:
            frame_start = time.monotonic()
            widths = grippers.read_normalized_widths()
            samples = [camera.sample_at() for camera in cameras]
            rr.set_time("sensor_time", duration=frame_start - started)
            for name, sample in zip(args.cameras, samples, strict=True):
                rr.log(
                    f"observation.images.{name}",
                    rr.Image(sample.image).compress(jpeg_quality=75),
                )
            rr.log(width_paths[0], rr.Scalars(widths.left_mm))
            rr.log(width_paths[1], rr.Scalars(widths.right_mm))
            time.sleep(max(0.0, 1 / args.fps - (time.monotonic() - frame_start)))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    try:
        run(parse_args())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
