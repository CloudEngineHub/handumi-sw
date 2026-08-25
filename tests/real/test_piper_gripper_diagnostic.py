import argparse
import logging
import time
from pathlib import Path

from handumi.config import DEFAULT_RIG_CONFIG
from handumi.feetech import (
    FeetechGripperPair,
    FeetechGripperSampler,
    assert_calibrated,
    load_config,
    user_calibration_path,
)
from handumi.real.can_setup import ensure_can_interfaces_ready
from handumi.real.piper.driver import (
    PiperCanEnvironment,
    PiperGripperFeedback,
    PiperGripperRange,
    load_piper_can_settings,
)
from handumi.robots.registry import load_embodiment
from handumi.teleop.standby import GRIPPER_FULLY_CLOSED, GRIPPER_REOPENED

log = logging.getLogger("handumi.piper_gripper_diagnostic")


def classify_opening(opening: float) -> str:
    """Test-only presentation of the thresholds used by parking."""
    if opening <= GRIPPER_FULLY_CLOSED:
        return "CERRADO"
    if opening >= GRIPPER_REOPENED:
        return "ABIERTO"
    return "INTERMEDIO"


def format_sample(
    side: str,
    handumi_opening: float,
    feedback: PiperGripperFeedback | None,
) -> str:
    """Test-only formatter retained for inspecting captured samples."""
    if feedback is None:
        return (
            f"{side}: HandUMI={handumi_opening:.4f} | Piper=SIN_FEEDBACK "
            "(el contador de home se reinicia)"
        )
    return (
        f"{side}: HandUMI={handumi_opening:.4f} | "
        f"Piper={feedback.opening:.6f} (calibrado={feedback.calibrated_mm:.3f} mm, "
        f"raw={feedback.measured_microm} um) | "
        f"estado={classify_opening(feedback.opening)}"
    )


def _feedback(opening: float, measured_microm: int = 500) -> PiperGripperFeedback:
    return PiperGripperFeedback(
        measured_microm=measured_microm,
        opening=opening,
        gripper_range=PiperGripperRange(0, 66_000, "test"),
    )


def test_classification_matches_parking_thresholds() -> None:
    assert classify_opening(0.0) == "CERRADO"
    assert classify_opening(0.005) == "CERRADO"
    assert classify_opening(0.00501) == "INTERMEDIO"
    assert classify_opening(0.149) == "INTERMEDIO"
    assert classify_opening(0.15) == "ABIERTO"


def test_format_sample_reports_raw_normalized_and_state() -> None:
    line = format_sample("left", 0.0, _feedback(0.0075))

    assert "HandUMI=0.0000" in line
    assert "Piper=0.007500" in line
    assert "calibrado=0.500 mm" in line
    assert "raw=500 um" in line
    assert "estado=INTERMEDIO" in line


def test_format_sample_explains_missing_feedback() -> None:
    line = format_sample("right", 1.0, None)

    assert "SIN_FEEDBACK" in line
    assert "contador de home se reinicia" in line


def _build_hardware_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Test manual: move only the Piper grippers from HandUMI input and "
            "print physical feedback. Arm joints hold their current position."
        )
    )
    parser.add_argument("--rig-config", type=Path, default=DEFAULT_RIG_CONFIG)
    parser.add_argument(
        "--side", choices=("left", "right", "both"), default="both"
    )
    parser.add_argument("--log-hz", type=float, default=2.0)
    parser.add_argument("--command-hz", type=float, default=30.0)
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
    if args.log_hz <= 0.0 or args.command_hz <= 0.0:
        raise SystemExit("--log-hz and --command-hz must be > 0")
    sides = ("left", "right") if args.side == "both" else (args.side,)

    feetech_config = load_config(args.rig_config)
    assert_calibrated(feetech_config, source=user_calibration_path())
    runtime = load_embodiment("piper")
    settings = load_piper_can_settings(args.rig_config, runtime.config.real)
    fallback_width_mm = runtime.config.gripper_max_width_m * 1000.0
    fallback_widths = {
        side: float(
            getattr(settings, f"{side}_gripper_max_width_mm")
            or fallback_width_mm
        )
        for side in sides
    }
    ensure_can_interfaces_ready(
        [settings.left_port, settings.right_port],
        bitrate=settings.bitrate,
        restart_ms=settings.restart_ms,
        repair=not args.skip_can_repair,
    )

    gripper_pair = FeetechGripperPair(feetech_config)
    sampler = FeetechGripperSampler(
        gripper_pair, sample_hz=max(100.0, args.command_hz)
    )
    environment = PiperCanEnvironment(settings)
    minima: dict[str, PiperGripperFeedback] = {}
    maxima: dict[str, PiperGripperFeedback] = {}
    try:
        gripper_pair.open()
        sampler.start()
        environment.connect()
        environment.start_streaming_current_pose()
        log.info(
            "Monitor manual activo; articulaciones inmoviles y solo grippers %s. "
            "Ctrl+C para terminar.",
            "/".join(sides),
        )
        log.info(
            "CERRADO <= %.4f; ABIERTO >= %.2f; entre ambos = INTERMEDIO.",
            GRIPPER_FULLY_CLOSED,
            GRIPPER_REOPENED,
        )

        command_period_s = 1.0 / args.command_hz
        log_period_s = 1.0 / args.log_hz
        next_log_s = 0.0
        while True:
            started_s = time.monotonic()
            latest = sampler.latest()
            if latest is None:
                raise RuntimeError("No hay muestras Feetech disponibles")
            all_openings = {
                "left": float(latest.widths.left_normalized),
                "right": float(latest.widths.right_normalized),
            }
            openings = {side: all_openings[side] for side in sides}
            environment.set_gripper_openings(
                openings, fallback_max_width_mm=fallback_widths
            )
            feedback = environment.gripper_feedback(
                fallback_max_width_mm=fallback_widths
            )
            environment.raise_if_failed()
            for side in sides:
                sample = feedback.get(side)
                if sample is None:
                    continue
                if side not in minima or sample.opening < minima[side].opening:
                    minima[side] = sample
                if side not in maxima or sample.opening > maxima[side].opening:
                    maxima[side] = sample

            if started_s >= next_log_s:
                for side in sides:
                    log.info(
                        "%s",
                        format_sample(side, openings[side], feedback.get(side)),
                    )
                next_log_s = started_s + log_period_s

            remaining_s = command_period_s - (time.monotonic() - started_s)
            if remaining_s > 0.0:
                time.sleep(remaining_s)
    except KeyboardInterrupt:
        log.info("Monitor detenido por el usuario.")
    finally:
        try:
            environment.close()
        finally:
            sampler.stop()
            gripper_pair.close()

        for side in sides:
            if side not in minima or side not in maxima:
                log.info("%s: no se recibio feedback Piper valido.", side)
                continue
            log.info(
                "%s resumen: minimo=%.6f (%+.3f mm), "
                "maximo=%.6f (%+.3f mm).",
                side,
                minima[side].opening,
                minima[side].measured_mm,
                maxima[side].opening,
                maxima[side].measured_mm,
            )


if __name__ == "__main__":
    main()
