"""Recoverable native-rate tracking sidecars for HandUMI datasets."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections import deque
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from handumi.tracking.packet import TrackingPacket, tracking_packet_record

SIDECAR_SCHEMA = "handumi_tracking_sidecar_v1"

_ARROW_SCHEMA = pa.schema(
    [
        ("sidecar_schema", pa.string()),
        ("episode_index", pa.int64()),
        ("attempt_id", pa.string()),
        ("capture_status", pa.string()),
        ("schema", pa.string()),
        ("source_schema_version", pa.int64()),
        ("source", pa.string()),
        ("source_sequence", pa.int64()),
        ("receive_sequence", pa.int64()),
        ("coordinate_space", pa.string()),
        ("source_time_ns", pa.int64()),
        ("source_time_domain", pa.string()),
        ("mapped_pc_monotonic_ns", pa.int64()),
        ("receive_time_ns", pa.int64()),
        ("clock_offset_ns", pa.int64()),
        ("rtt_ns", pa.int64()),
        ("uncertainty_ns", pa.int64()),
        ("timestamp_quality", pa.string()),
        ("body_active", pa.bool_()),
        ("body_requested_joint_set", pa.string()),
        ("body_active_joint_set", pa.string()),
        ("body_joint_count", pa.int64()),
        ("body_confidence", pa.float64()),
        ("body_calibration_state", pa.string()),
        ("body_fidelity", pa.string()),
        ("body_skeleton_revision", pa.int64()),
        ("body_source_time_ns", pa.int64()),
        ("body_source_time_domain", pa.string()),
        ("body_observation_sequence", pa.int64()),
        ("body_is_new_observation", pa.int64()),
        ("body_joint_indices", pa.list_(pa.int64())),
        ("body_joint_names", pa.list_(pa.string())),
        ("body_joint_poses", pa.list_(pa.list_(pa.float64(), 7))),
        ("body_joint_location_flags", pa.list_(pa.int64())),
        ("body_joint_tracking_states", pa.list_(pa.string())),
        ("body_joint_confidences", pa.list_(pa.float64())),
        ("body_joint_provenance", pa.list_(pa.string())),
        ("raw_json", pa.string()),
    ]
)


def _packet_time_ns(packet: TrackingPacket) -> int:
    return int(
        packet.timestamps.mapped_pc_monotonic_ns
        or packet.timestamps.receive_time_ns
    )


def _sidecar_row(
    packet: TrackingPacket,
    *,
    episode_index: int,
    attempt_id: str,
    capture_status: str,
) -> dict[str, Any]:
    envelope = tracking_packet_record(packet)
    body = packet.body
    joints = () if body is None else body.joints
    return {
        "sidecar_schema": SIDECAR_SCHEMA,
        "episode_index": int(episode_index),
        "attempt_id": attempt_id,
        "capture_status": capture_status,
        "schema": packet.schema,
        "source_schema_version": packet.source_schema_version,
        "source": packet.source,
        "source_sequence": -1 if packet.sequence is None else packet.sequence,
        "receive_sequence": packet.receive_sequence,
        "coordinate_space": packet.coordinate_space,
        "source_time_ns": packet.timestamps.source_time_ns,
        "source_time_domain": packet.timestamps.source_time_domain,
        "mapped_pc_monotonic_ns": packet.timestamps.mapped_pc_monotonic_ns,
        "receive_time_ns": packet.timestamps.receive_time_ns,
        "clock_offset_ns": packet.timestamps.clock_offset_ns,
        "rtt_ns": packet.timestamps.rtt_ns,
        "uncertainty_ns": packet.timestamps.uncertainty_ns,
        "timestamp_quality": packet.timestamps.quality.value,
        "body_active": False if body is None else body.active,
        "body_requested_joint_set": "" if body is None else body.requested_joint_set,
        "body_active_joint_set": "" if body is None else body.active_joint_set,
        "body_joint_count": 0 if body is None else body.joint_count,
        "body_confidence": float("nan") if body is None else body.confidence,
        "body_calibration_state": "" if body is None else body.calibration_state,
        "body_fidelity": "" if body is None else body.fidelity,
        "body_skeleton_revision": 0 if body is None else body.skeleton_revision,
        "body_source_time_ns": 0 if body is None else body.source_time_ns,
        "body_source_time_domain": "" if body is None else body.source_time_domain,
        "body_observation_sequence": (
            -1
            if body is None or body.observation_sequence is None
            else body.observation_sequence
        ),
        "body_is_new_observation": (
            -1
            if body is None or body.is_new_observation is None
            else int(body.is_new_observation)
        ),
        "body_joint_indices": [joint.index for joint in joints],
        "body_joint_names": [joint.name for joint in joints],
        "body_joint_poses": [list(joint.pose) for joint in joints],
        "body_joint_location_flags": [joint.location_flags for joint in joints],
        "body_joint_tracking_states": [joint.tracking_state.value for joint in joints],
        "body_joint_confidences": [joint.confidence for joint in joints],
        "body_joint_provenance": [joint.provenance.value for joint in joints],
        "raw_json": json.dumps(
            envelope["packet"], separators=(",", ":"), allow_nan=True
        ),
    }


def load_tracking_sidecar(path: str | Path) -> list[dict[str, Any]]:
    """Load lossless source mappings from a native tracking Parquet sidecar."""
    table = pd.read_parquet(path)
    return [json.loads(value) for value in table["raw_json"].tolist()]


def discover_tracking_sidecars(
    dataset_root: str | Path, *, episode_index: int | None = None
) -> tuple[Path, ...]:
    root = Path(dataset_root) / "raw" / "tracking"
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return ()
    manifest = json.loads(manifest_path.read_text())
    files: list[Path] = []
    for record in manifest.get("files", []):
        if (
            episode_index is not None
            and int(record.get("episode_index", -1)) != episode_index
        ):
            continue
        path = Path(dataset_root) / str(record["path"])
        if path.exists():
            files.append(path)
    return tuple(files)


class TrackingSidecarWriter:
    """Append packets to a crash journal, then atomically publish Parquet.

    Discarded and interrupted attempts remain evidence in dedicated directories;
    only successfully saved episodes use the canonical chunk/file path.
    """

    def __init__(self, dataset_root: str | Path, *, chunks_size: int = 1000) -> None:
        self.dataset_root = Path(dataset_root)
        self.root = self.dataset_root / "raw" / "tracking"
        self.chunks_size = int(chunks_size)
        self.root.mkdir(parents=True, exist_ok=True)
        self._journal = None
        self._journal_path: Path | None = None
        self._episode_index: int | None = None
        self._attempt_id: str | None = None
        self._recent: deque[TrackingPacket] = deque(maxlen=4096)
        self._manifest_hashes: set[str] = set()
        manifest_records = self.root / "session_manifests.jsonl"
        if manifest_records.exists():
            for line in manifest_records.read_text().splitlines():
                if line.strip():
                    self._manifest_hashes.add(
                        hashlib.sha256(line.strip().encode()).hexdigest()
                    )
        self.recover_interrupted()

    def start_episode(self, episode_index: int) -> str:
        if self._journal is not None:
            raise RuntimeError("A tracking sidecar episode is already active")
        self._episode_index = int(episode_index)
        self._attempt_id = uuid.uuid4().hex
        inprogress = self.root / "inprogress"
        inprogress.mkdir(parents=True, exist_ok=True)
        self._journal_path = (
            inprogress
            / f"episode-{self._episode_index:06d}-{self._attempt_id}.jsonl.inprogress"
        )
        self._journal = self._journal_path.open("a", encoding="utf-8")
        self._recent.clear()
        return self._attempt_id

    def append_packets(self, packets: Iterable[TrackingPacket]) -> int:
        if self._journal is None or self._episode_index is None or self._attempt_id is None:
            raise RuntimeError("start_episode() must be called before appending packets")
        count = 0
        for packet in packets:
            self._recent.append(packet)
            record = _sidecar_row(
                packet,
                episode_index=self._episode_index,
                attempt_id=self._attempt_id,
                capture_status="inprogress",
            )
            self._journal.write(
                json.dumps(record, separators=(",", ":"), allow_nan=True) + "\n"
            )
            count += 1
        if count:
            self._journal.flush()
            os.fsync(self._journal.fileno())
        return count

    def drain_provider(self, provider: object) -> int:
        self._capture_provider_manifest(provider)
        drain = getattr(provider, "drain_packets", None)
        return self.append_packets(drain()) if callable(drain) else 0

    def _capture_provider_manifest(self, provider: object) -> None:
        accessor = getattr(provider, "session_manifest", None)
        manifest = accessor() if callable(accessor) else accessor
        if not isinstance(manifest, dict):
            return
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        if digest in self._manifest_hashes:
            return
        path = self.root / "session_manifests.jsonl"
        with path.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self._manifest_hashes.add(digest)

    def nearest_packet(self, target_time_ns: int) -> TrackingPacket | None:
        if not self._recent:
            return None
        return min(
            self._recent,
            key=lambda packet: abs(_packet_time_ns(packet) - int(target_time_ns)),
        )

    def finish_episode(self, *, status: str, provider: object | None = None) -> Path | None:
        if self._journal is None or self._journal_path is None:
            return None
        if provider is not None:
            self.drain_provider(provider)
        self._journal.flush()
        os.fsync(self._journal.fileno())
        self._journal.close()
        self._journal = None
        path = self._finalize_journal(self._journal_path, status=status)
        self._journal_path = None
        self._episode_index = None
        self._attempt_id = None
        self._recent.clear()
        return path

    def close(self) -> Path | None:
        return self.finish_episode(status="interrupted")

    def recover_interrupted(self) -> tuple[Path, ...]:
        recovered = []
        for journal in sorted((self.root / "inprogress").glob("*.jsonl.inprogress")):
            result = self._finalize_journal(journal, status="interrupted")
            if result is not None:
                recovered.append(result)
        return tuple(recovered)

    def _finalize_journal(self, journal: Path, *, status: str) -> Path | None:
        first: dict[str, Any] | None = None
        last: dict[str, Any] | None = None
        count = 0
        truncated_lines = 0
        chunk: list[dict[str, Any]] = []
        parquet_writer: pq.ParquetWriter | None = None
        output: Path | None = None
        temporary: Path | None = None
        with journal.open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # A process may die midway through its final append. Earlier
                    # fsynced records remain publishable; the truncated tail is
                    # explicitly counted in the sidecar manifest.
                    truncated_lines += 1
                    break
                record["capture_status"] = status
                if first is None:
                    first = record
                    episode_index = int(record["episode_index"])
                    attempt_id = str(record["attempt_id"])
                    if status == "recorded":
                        chunk_index = episode_index // self.chunks_size
                        file_index = episode_index % self.chunks_size
                        output = (
                            self.root
                            / f"chunk-{chunk_index:03d}"
                            / f"file-{file_index:03d}.parquet"
                        )
                    else:
                        output = (
                            self.root
                            / status
                            / f"episode-{episode_index:06d}-{attempt_id}.parquet"
                        )
                    output.parent.mkdir(parents=True, exist_ok=True)
                    temporary = output.with_suffix(output.suffix + ".tmp")
                    parquet_writer = pq.ParquetWriter(temporary, _ARROW_SCHEMA)
                last = record
                count += 1
                chunk.append(record)
                if len(chunk) >= 1024:
                    assert parquet_writer is not None
                    parquet_writer.write_table(
                        pa.Table.from_pylist(chunk, schema=_ARROW_SCHEMA)
                    )
                    chunk.clear()
        if first is None:
            if truncated_lines:
                corrupt = self.root / "corrupt" / (journal.name + ".corrupt")
                corrupt.parent.mkdir(parents=True, exist_ok=True)
                journal.replace(corrupt)
            else:
                journal.unlink(missing_ok=True)
            return None
        assert parquet_writer is not None and temporary is not None and output is not None
        if chunk:
            parquet_writer.write_table(pa.Table.from_pylist(chunk, schema=_ARROW_SCHEMA))
        parquet_writer.close()
        temporary.replace(output)
        journal.unlink()
        assert last is not None
        self._update_manifest(
            output,
            first,
            last,
            count=count,
            status=status,
            truncated_lines=truncated_lines,
        )
        return output

    def _update_manifest(
        self,
        output: Path,
        first: dict[str, Any],
        last: dict[str, Any],
        *,
        count: int,
        status: str,
        truncated_lines: int,
    ) -> None:
        path = self.root / "manifest.json"
        manifest = (
            json.loads(path.read_text())
            if path.exists()
            else {"schema": SIDECAR_SCHEMA, "files": []}
        )
        relative = output.relative_to(self.dataset_root).as_posix()
        manifest["files"] = [
            item for item in manifest.get("files", []) if item.get("path") != relative
        ]
        manifest["files"].append(
            {
                "path": relative,
                "episode_index": int(first["episode_index"]),
                "attempt_id": str(first["attempt_id"]),
                "status": status,
                "packet_count": count,
                "first_receive_sequence": int(first["receive_sequence"]),
                "last_receive_sequence": int(last["receive_sequence"]),
                "recovery_truncated_lines": truncated_lines,
            }
        )
        if (self.root / "session_manifests.jsonl").exists():
            manifest["session_manifests"] = "raw/tracking/session_manifests.jsonl"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(path)


__all__ = [
    "SIDECAR_SCHEMA",
    "TrackingSidecarWriter",
    "discover_tracking_sidecars",
    "load_tracking_sidecar",
]
