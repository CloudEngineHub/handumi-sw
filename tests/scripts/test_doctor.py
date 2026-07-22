from pathlib import Path
from unittest import mock

from handumi.scripts import doctor


def test_doctor_reports_missing_rig_as_failure(tmp_path: Path):
    with (
        mock.patch.object(
            doctor,
            "_encoder_check",
            return_value=doctor.DoctorCheck("video encoder", "pass", "h264"),
        ),
        mock.patch.object(
            doctor,
            "_python_stack_check",
            return_value=doctor.DoctorCheck("Python stack", "pass", "ok"),
        ),
    ):
        checks = doctor.collect_doctor_checks(tmp_path / "missing.yaml")

    assert any(check.name == "rig.yaml" and check.status == "fail" for check in checks)


def test_run_doctor_returns_false_for_failed_check(capsys):
    checks = [doctor.DoctorCheck("rig.yaml", "fail", "missing")]
    with mock.patch.object(doctor, "collect_doctor_checks", return_value=checks):
        assert not doctor.run_doctor(Path("rig.yaml"))

    assert "[FAIL] rig.yaml" in capsys.readouterr().out
