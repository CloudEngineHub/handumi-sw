import subprocess
from unittest import mock

import pytest

from handumi.real.backends.setup import _check_openarm_motors


def test_openarm_motor_check_uses_read_only_parameter_queries():
    output = "\n".join(f"MOTOR ID: 0x{motor:x}" for motor in range(1, 9))
    with mock.patch(
        "handumi.real.backends.setup.subprocess.run",
        return_value=subprocess.CompletedProcess([], 0, stdout=output, stderr=""),
    ) as run:
        _check_openarm_motors("right", "can0")

    command = run.call_args.args[0]
    assert "show_param" in command
    assert "monitor" not in command
    assert command[-1] == "1,2,3,4,5,6,7,8"


def test_openarm_motor_check_rejects_any_missing_response():
    output = "\n".join(f"MOTOR ID: 0x{motor:x}" for motor in range(1, 9))
    output += "\n[!] NO RESPONSE FROM MOTOR"
    with (
        mock.patch(
            "handumi.real.backends.setup.subprocess.run",
            return_value=subprocess.CompletedProcess([], 0, stdout=output, stderr=""),
        ),
        pytest.raises(SystemExit, match="diagnostic failed"),
    ):
        _check_openarm_motors("left", "can1")
