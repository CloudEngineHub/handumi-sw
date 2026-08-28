from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from handumi.calibration.deployment import (
    load_deployment_calibration,
    resolve_deployment_calibration,
)
from handumi.dataset.raw import (
    HANDUMI_CAPTURE_SCHEMA,
    HANDUMI_STATE_SEMANTICS,
    HANDUMI_TRACKING_SCHEMA,
)
from handumi.scripts.replay import replay_in_sim


def _write_calibration(path: Path, *, scope: str, position: list[float]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "kind": "handumi_robot_table_calibration",
                "robot": "piper",
                "scope": scope,
                "source": "test",
                "verified": scope == "physical",
                "calibration": {
                    "robot_from_table": {
                        "position": position,
                        "quaternion": [0.0, 0.0, 0.0, 1.0],
                    }
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_auto_prefers_lab_local_calibration(tmp_path: Path) -> None:
    local = _write_calibration(
        tmp_path / "lab" / "piper.yaml",
        scope="physical",
        position=[0.4, 0.1, 0.2],
    )
    rig = tmp_path / "rig.yaml"
    rig.write_text(
        yaml.safe_dump(
            {
                "deployment": {
                    "lab": "test_lab",
                    "table_calibrations": {"piper": str(local)},
                }
            }
        ),
        encoding="utf-8",
    )

    selection = resolve_deployment_calibration("piper", rig_config=rig)

    assert selection.profile == "local"
    assert selection.scope == "physical"
    assert selection.verified
    assert selection.lab == "test_lab"
    np.testing.assert_allclose(selection.pose7[:3], [0.4, 0.1, 0.2])


def test_auto_discovers_conventional_lab_local_calibration(tmp_path: Path) -> None:
    local = _write_calibration(
        tmp_path / "calibration" / "table" / "local" / "piper.yaml",
        scope="physical",
        position=[0.5, 0.0, 0.2],
    )
    payload = yaml.safe_load(local.read_text())
    payload["lab"] = "test_lab"
    local.write_text(yaml.safe_dump(payload), encoding="utf-8")
    rig = tmp_path / "rig.yaml"
    rig.write_text(
        yaml.safe_dump({"deployment": {"lab": "test_lab"}}),
        encoding="utf-8",
    )

    selection = resolve_deployment_calibration("piper", rig_config=rig)

    assert selection.path == local
    assert selection.profile == "local"
    assert selection.lab == "test_lab"


def test_sim_profile_is_separate_from_lab_config(tmp_path: Path) -> None:
    local = _write_calibration(
        tmp_path / "piper.yaml",
        scope="physical",
        position=[9.0, 9.0, 9.0],
    )
    rig = tmp_path / "rig.yaml"
    rig.write_text(
        yaml.safe_dump(
            {
                "deployment": {
                    "lab": "test_lab",
                    "table_calibrations": {"piper": str(local)},
                }
            }
        ),
        encoding="utf-8",
    )

    selection = resolve_deployment_calibration(
        "piper",
        profile="sim",
        rig_config=rig,
    )

    assert selection.profile == "sim"
    assert selection.scope == "simulation"
    assert selection.path.as_posix().endswith("table/sim/piper.yaml")
    np.testing.assert_allclose(selection.pose7[:3], [0.30, 0.0, 0.0])


def test_lab_local_calibration_requires_lab_identity(tmp_path: Path) -> None:
    local = _write_calibration(
        tmp_path / "piper.yaml",
        scope="physical",
        position=[0.4, 0.1, 0.2],
    )
    rig = tmp_path / "rig.yaml"
    rig.write_text(
        yaml.safe_dump(
            {"deployment": {"table_calibrations": {"piper": str(local)}}}
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="Missing deployment.lab"):
        resolve_deployment_calibration("piper", rig_config=rig)


def test_explicit_calibration_overrides_profile(tmp_path: Path) -> None:
    explicit = _write_calibration(
        tmp_path / "explicit.yaml",
        scope="simulation",
        position=[0.7, 0.0, 0.1],
    )

    selection = resolve_deployment_calibration(
        "piper",
        explicit_path=explicit,
        profile="local",
        rig_config=tmp_path / "missing.yaml",
    )

    assert selection.profile == "explicit"
    np.testing.assert_allclose(selection.pose7[:3], [0.7, 0.0, 0.1])


def test_legacy_file_without_scope_gets_actionable_error(tmp_path: Path) -> None:
    path = _write_calibration(
        tmp_path / "legacy.yaml",
        scope="physical",
        position=[0.3, 0.0, 0.0],
    )
    payload = yaml.safe_load(path.read_text())
    del payload["scope"]
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(SystemExit, match="missing or invalid scope"):
        load_deployment_calibration(path, expected_robot="piper")


def test_non_normalized_quaternion_is_rejected(tmp_path: Path) -> None:
    path = _write_calibration(
        tmp_path / "badquat.yaml",
        scope="simulation",
        position=[0.3, 0.0, 0.0],
    )
    payload = yaml.safe_load(path.read_text())
    payload["calibration"]["robot_from_table"]["quaternion"] = [1.0, 1.0, 1.0, 1.0]
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(SystemExit, match="not normalized"):
        load_deployment_calibration(path, expected_robot="piper")


def test_auto_with_local_file_but_missing_rig_gives_deployment_error(
    tmp_path: Path,
) -> None:
    rig = tmp_path / "rig.yaml"
    _write_calibration(
        tmp_path / "calibration" / "table" / "local" / "piper.yaml",
        scope="physical",
        position=[0.4, 0.0, 0.2],
    )

    with pytest.raises(SystemExit, match="does not exist"):
        resolve_deployment_calibration("piper", rig_config=rig)


def test_sim_profile_ignores_malformed_rig(tmp_path: Path) -> None:
    rig = tmp_path / "rig.yaml"
    rig.write_text("deployment: not_a_mapping\n", encoding="utf-8")

    selection = resolve_deployment_calibration("piper", profile="sim", rig_config=rig)

    assert selection.profile == "sim"


def test_simulation_calibration_cannot_claim_physical_verification(
    tmp_path: Path,
) -> None:
    path = _write_calibration(
        tmp_path / "invalid.yaml",
        scope="simulation",
        position=[0.3, 0.0, 0.0],
    )
    payload = yaml.safe_load(path.read_text())
    payload["verified"] = True
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(SystemExit, match="cannot be physically verified"):
        load_deployment_calibration(path, expected_robot="piper")


def test_replay_reads_parquet_columns_without_decoding_video(
    monkeypatch,
    tmp_path: Path,
) -> None:
    info = {
        "handumi": {
            "tracking_schema": HANDUMI_TRACKING_SCHEMA,
            "capture_schema": HANDUMI_CAPTURE_SCHEMA,
            "state_semantics": HANDUMI_STATE_SEMANTICS,
        }
    }
    state = np.zeros((2, 16), dtype=np.float32)
    state[:, 6] = 1.0
    state[:, 13] = 1.0

    class Table:
        column_names = [
            "observation.state",
            *replay_in_sim.GRIPPER_NORMALIZED_KEYS,
        ]

        def __getitem__(self, key: str):
            if key == "observation.state":
                return state
            return np.asarray([[0.25], [0.75]], dtype=np.float32)

    observed: dict[str, object] = {}

    def fake_open_dataset(ref, **kwargs):
        observed.update(kwargs)
        return SimpleNamespace(fps=30, hf_dataset=Table())

    monkeypatch.setattr(replay_in_sim, "ensure_metadata", lambda ref: info)
    monkeypatch.setattr(replay_in_sim, "open_dataset", fake_open_dataset)
    args = Namespace(
        repo_id="local/test",
        dataset_root=tmp_path,
        revision="main",
        episode=0,
        source="observation.state",
    )

    loaded, fps, _, grippers = replay_in_sim.load_episode_states(args)

    np.testing.assert_array_equal(loaded, state)
    assert fps == 30
    assert observed["download_videos"] is False
    assert grippers is not None
    np.testing.assert_allclose(grippers[:, 0], [0.25, 0.75])
