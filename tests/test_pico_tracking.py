from __future__ import annotations

from subprocess import CompletedProcess

from handumi.tracking.pico import (
    PICO_AUDIO_SERVICE_PORT,
    PICO_SERVICE_PORT,
    setup_adb_reverse,
)


def test_setup_adb_reverse_configures_tracking_and_audio_ports() -> None:
    commands: list[list[str]] = []

    def runner(command, **kwargs):
        commands.append(command)
        return CompletedProcess(command, 0, stdout="", stderr="")

    assert setup_adb_reverse(runner=runner)
    assert commands == [
        [
            "adb",
            "reverse",
            f"tcp:{PICO_SERVICE_PORT}",
            f"tcp:{PICO_SERVICE_PORT}",
        ],
        [
            "adb",
            "reverse",
            f"tcp:{PICO_AUDIO_SERVICE_PORT}",
            f"tcp:{PICO_AUDIO_SERVICE_PORT}",
        ],
    ]


def test_setup_adb_reverse_attempts_both_ports_when_one_fails() -> None:
    commands: list[list[str]] = []

    def runner(command, **kwargs):
        commands.append(command)
        port = command[2]
        return CompletedProcess(
            command,
            1 if port == f"tcp:{PICO_SERVICE_PORT}" else 0,
            stdout="",
            stderr="failed",
        )

    assert not setup_adb_reverse(runner=runner)
    assert [command[2] for command in commands] == [
        f"tcp:{PICO_SERVICE_PORT}",
        f"tcp:{PICO_AUDIO_SERVICE_PORT}",
    ]
