import argparse
import logging
import time
from pathlib import Path

import numpy as np

from handumi.config import DEFAULT_RIG_CONFIG
from handumi.real.can_setup import ensure_can_interfaces_ready
from handumi.real.piper.driver import (
    PiperGripperFeedback,
    load_piper_can_settings,
)
from handumi.robots.registry import load_embodiment, resolve_home_q

log = logging.getLogger("handumi.piper_joint_monitor")

ARM_JOINT_COUNT = 6


def format_arm_joints(side: str, mdeg: np.ndarray) -> str:
    """Test-only formatter: one ``{side}_joint{i}`` entry per arm joint, in deg."""
    degrees = np.asarray(mdeg, dtype=np.float64) / 1000.0
    return " ".join(
        f"{side}_joint{i}={deg:+8.3f}deg"
        for i, deg in enumerate(degrees[:ARM_JOINT_COUNT], start=1)
    )


def format_gripper_joint(
    side: str,
    feedback: PiperGripperFeedback | None,
) -> str:
    """Test-only formatter: ``{side}_joint7`` reports gripper opening in meters."""
    if feedback is None:
        return f"{side}_joint7=SIN_FEEDBACK"
    return f"{side}_joint7={feedback.calibrated_mm / 1000.0:+.4f}m"


def format_side_line(
    side: str,
    mdeg: np.ndarray,
    feedback: PiperGripperFeedback | None,
) -> str:
    """Test-only formatter combining all 7 joints of one side into one line."""
    return f"{side}: {format_arm_joints(side, mdeg)} {format_gripper_joint(side, feedback)}"


def test_format_arm_joints_lists_each_joint_in_degrees() -> None:
    mdeg = np.array([1000, -2000, 3000, 0, 45000, -90000])

    line = format_arm_joints("left", mdeg)

    assert "left_joint1=  +1.000deg" in line
    assert "left_joint2=  -2.000deg" in line
    assert "left_joint5= +45.000deg" in line
    assert "left_joint6= -90.000deg" in line


def test_format_gripper_joint_reports_calibrated_opening_in_meters() -> None:
    from handumi.real.piper.driver import PiperGripperRange

    feedback = PiperGripperFeedback(
        measured_microm=500,
        opening=0.2,
        gripper_range=PiperGripperRange(-3_000, 66_000, "test"),
    )

    line = format_gripper_joint("right", feedback)

    assert line == "right_joint7=+0.0035m"


def test_format_gripper_joint_explains_missing_feedback() -> None:
    assert format_gripper_joint("left", None) == "left_joint7=SIN_FEEDBACK"


def _build_hardware_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Test manual: home the Piper arms, then only read joint angles "
            "while the operator moves them by hand. No further joint or "
            "gripper commands are sent after homing."
        )
    )
    parser.add_argument("--rig-config", type=Path, default=DEFAULT_RIG_CONFIG)
    parser.add_argument(
        "--side", choices=("left", "right", "both"), default="both"
    )
    parser.add_argument("--log-hz", type=float, default=5.0)
    parser.add_argument("--skip-can-repair", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the opt-in hardware monitor; pytest never calls this function."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_hardware_parser().parse_args(argv)
    if args.log_hz <= 0.0:
        raise SystemExit("--log-hz must be > 0")
    sides = ("left", "right") if args.side == "both" else (args.side,)

    runtime = load_embodiment("piper")
    _, home_q = resolve_home_q(runtime, rig_config=args.rig_config)
    settings = load_piper_can_settings(args.rig_config, runtime.config.real)
    fallback_width_mm = runtime.config.gripper_max_width_m * 1000.0
    fallback_widths = {
        side: float(
            getattr(settings, f"{side}_gripper_max_width_mm") or fallback_width_mm
        )
        for side in sides
    }
    ensure_can_interfaces_ready(
        [settings.left_port, settings.right_port],
        bitrate=settings.bitrate,
        restart_ms=settings.restart_ms,
        repair=not args.skip_can_repair,
    )

    # Imported lazily: PiperCanEnvironment/q_to_piper_mdeg pull in the piper_sdk
    # backend, which the pure formatting tests above must not require.
    from handumi.real.piper.driver import PiperCanEnvironment, q_to_piper_mdeg

    environment = PiperCanEnvironment(settings)
    try:
        environment.connect()
        log.info("Enviando ambos brazos a home ...")
        environment.home(q_to_piper_mdeg(home_q, runtime.joint_names))
        log.info("Home alcanzado. Liberando motores para mover a mano ...")
        for side in sides:
            environment.disable_arm(side)
            environment.disable_gripper(side)
        log.info(
            "Monitor de solo lectura activo para %s. Mueve los brazos a mano. "
            "Ctrl+C para terminar.",
            "/".join(sides),
        )

        log_period_s = 1.0 / args.log_hz
        while True:
            started_s = time.monotonic()
            feedback_mdeg = environment.feedback_mdeg()
            gripper_feedback = environment.gripper_feedback(
                fallback_max_width_mm=fallback_widths
            )
            for side in sides:
                log.info(
                    "%s",
                    format_side_line(
                        side, feedback_mdeg[side], gripper_feedback.get(side)
                    ),
                )
            remaining_s = log_period_s - (time.monotonic() - started_s)
            if remaining_s > 0.0:
                time.sleep(remaining_s)
    except KeyboardInterrupt:
        log.info("Monitor detenido por el usuario.")
    finally:
        environment.close()


if __name__ == "__main__":
    main()
