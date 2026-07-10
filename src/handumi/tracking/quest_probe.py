"""Capture and analyze raw Quest tracking packets for platform qualification.

This is investigation tooling, not the production body schema. Sender payloads
are preserved unchanged so runtime evidence can be reanalyzed after the wire
contract evolves. All Quest poses, including body joints, are runtime estimates.
This module does not estimate or report anatomical center of mass.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TextIO

from handumi.tracking.meta_quest import MetaQuestConfig, MetaQuestReceiver, QuestFrame

PROBE_SCHEMA = "handumi_quest_probe_v1"


def _finite_number(value: object) -> int | float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value if math.isfinite(float(value)) else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _nested_int(packet: Mapping[str, Any], paths: tuple[tuple[str, ...], ...]) -> int | None:
    for path in paths:
        value: object = packet
        for key in path:
            if not isinstance(value, Mapping) or key not in value:
                break
            value = value[key]
        else:
            number = _finite_number(value)
            if number is not None:
                return int(number)
    return None


def _body_sample_time_ns(packet: Mapping[str, Any]) -> int | None:
    """Read diagnostic body time without declaring a production wire schema."""
    return _nested_int(
        packet,
        (
            ("body", "sampleTimeNs"),
            ("body", "xrTime"),
            ("bodySampleTimeNs",),
            ("bodyXrTime",),
        ),
    )


@dataclass
class ProbeCapture:
    """Append-only JSONL writer called from ``MetaQuestReceiver``'s RX thread."""

    stream: TextIO
    metrics_provider: Callable[[], Mapping[str, Any]] = field(default=lambda: {})
    flush_every: int = 1
    count: int = 0

    def record(self, frame: QuestFrame) -> None:
        metrics = self.metrics_provider()
        envelope = {
            "probe_schema": PROBE_SCHEMA,
            "capture_index": self.count,
            "pc_receive_sequence": int(frame.receive_sequence),
            "pc_receive_time_ns": int(frame.pc_monotonic_ns),
            "sync": {
                "clock_synced": metrics.get("rtt_ns") is not None,
                "clock_offset_ns": _finite_number(metrics.get("offset_ns")),
                "rtt_ns": _finite_number(metrics.get("rtt_ns")),
            },
            "packet": frame.raw,
        }
        self.stream.write(json.dumps(envelope, separators=(",", ":"), allow_nan=False))
        self.stream.write("\n")
        self.count += 1
        if self.flush_every > 0 and self.count % self.flush_every == 0:
            self.stream.flush()


def iter_probe_records(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed probe JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Probe record at {path}:{line_number} is not an object")
            yield record


def _rate_hz(times_ns: list[int]) -> float | None:
    if len(times_ns) < 2 or times_ns[-1] <= times_ns[0]:
        return None
    return (len(times_ns) - 1) * 1e9 / (times_ns[-1] - times_ns[0])


def analyze_probe_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    receive_times: list[int] = []
    device_times: list[int] = []
    body_times: list[int] = []
    offsets: list[float] = []
    rtts: list[float] = []
    sample_ages: list[float] = []
    source_sequences: list[int] = []
    packet_count = 0

    for record in records:
        packet = record.get("packet")
        if not isinstance(packet, Mapping):
            continue
        packet_count += 1
        receive_time = _finite_number(record.get("pc_receive_time_ns"))
        if receive_time is not None:
            receive_times.append(int(receive_time))
        device_time = _nested_int(packet, (("ovrTimeNs",), ("deviceTimeNs",)))
        if device_time is not None and device_time > 0:
            device_times.append(device_time)
        body_time = _body_sample_time_ns(packet)
        if body_time is not None and body_time > 0 and (
            not body_times or body_time != body_times[-1]
        ):
            body_times.append(body_time)
        if "seq" in packet:
            sequence = _finite_number(packet.get("seq"))
            if sequence is not None:
                source_sequences.append(int(sequence))
        sync = record.get("sync")
        if isinstance(sync, Mapping):
            offset = _finite_number(sync.get("clock_offset_ns"))
            rtt = _finite_number(sync.get("rtt_ns"))
            clock_synced = sync.get("clock_synced") is True
            if clock_synced and offset is not None:
                offsets.append(float(offset))
            if clock_synced and rtt is not None:
                rtts.append(float(rtt))
            if (
                clock_synced
                and receive_time is not None
                and device_time is not None
                and offset is not None
            ):
                sample_ages.append(float(receive_time - (device_time + offset)))

    gaps = duplicates = resets_or_out_of_order = 0
    for previous, current in zip(source_sequences, source_sequences[1:], strict=False):
        delta = current - previous
        if delta > 1:
            gaps += delta - 1
        elif delta == 0:
            duplicates += 1
        elif delta < 0:
            resets_or_out_of_order += 1

    expected = len(source_sequences) + gaps
    loss_fraction = gaps / expected if expected else None
    interarrival = [
        float(current - previous)
        for previous, current in zip(receive_times, receive_times[1:], strict=False)
        if current > previous
    ]
    return {
        "probe_schema": PROBE_SCHEMA,
        "packet_count": packet_count,
        "receive_rate_hz": _rate_hz(receive_times),
        "device_rate_hz": _rate_hz(device_times),
        "body_update_rate_hz": _rate_hz(body_times),
        "receive_interarrival_ns": {
            "sample_count": len(interarrival),
            "median": statistics.median(interarrival) if interarrival else None,
            "p95": _percentile(interarrival, 0.95),
            "standard_deviation": (
                statistics.pstdev(interarrival) if len(interarrival) > 1 else None
            ),
            "maximum": max(interarrival) if interarrival else None,
        },
        "source_sequence": {
            "available": bool(source_sequences),
            "sample_count": len(source_sequences),
            "missing_packets": gaps if source_sequences else None,
            "loss_fraction": loss_fraction,
            "duplicates": duplicates if source_sequences else None,
            "resets_or_out_of_order": (
                resets_or_out_of_order if source_sequences else None
            ),
        },
        "clock_offset_ns": {
            "sample_count": len(offsets),
            "median": statistics.median(offsets) if offsets else None,
            "standard_deviation": statistics.pstdev(offsets) if len(offsets) > 1 else None,
            "minimum": min(offsets) if offsets else None,
            "maximum": max(offsets) if offsets else None,
        },
        "rtt_ns": {
            "sample_count": len(rtts),
            "median": statistics.median(rtts) if rtts else None,
            "p95": _percentile(rtts, 0.95),
            "maximum": max(rtts) if rtts else None,
        },
        "mapped_sample_age_ns": {
            "sample_count": len(sample_ages),
            "median": statistics.median(sample_ages) if sample_ages else None,
            "p95": _percentile(sample_ages, 0.95),
            "minimum": min(sample_ages) if sample_ages else None,
            "maximum": max(sample_ages) if sample_ages else None,
        },
        "limitations": [
            "Quest poses are platform-provided estimates, not direct measurements.",
            "Packet timing and loss statistics do not establish pose accuracy.",
            "Anatomical center of mass is not measured or estimated by this probe.",
        ],
    }


def analyze_probe_file(path: str | Path) -> dict[str, Any]:
    return analyze_probe_records(iter_probe_records(path))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def capture(args: argparse.Namespace) -> int:
    if args.duration_s <= 0:
        raise SystemExit("--duration-s must be greater than zero")
    if args.flush_every < 0:
        raise SystemExit("--flush-every must be zero or greater")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "quest_packets.jsonl"

    config = MetaQuestConfig.from_yaml(args.config)
    with raw_path.open("w", encoding="utf-8") as stream:
        session = ProbeCapture(stream=stream, flush_every=args.flush_every)
        receiver = MetaQuestReceiver(config, on_frame=session.record)
        session.metrics_provider = receiver.metrics
        started = datetime.now(UTC)
        receiver.start()
        try:
            deadline = time.monotonic() + args.duration_s
            while time.monotonic() < deadline:
                time.sleep(min(0.25, max(0.0, deadline - time.monotonic())))
        except KeyboardInterrupt:
            pass
        finally:
            receiver.stop()

    context = {
        "probe_schema": PROBE_SCHEMA,
        "started_at": started.isoformat(),
        "finished_at": datetime.now(UTC).isoformat(),
        "configured_duration_s": args.duration_s,
        "quest_ip": config.quest_ip,
        "tcp_port": config.tcp_port,
        "sync_port": config.sync_port,
        "raw_packets": raw_path.name,
    }
    summary = analyze_probe_file(raw_path)
    _write_json(output_dir / "capture_context.json", context)
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


def analyze(args: argparse.Namespace) -> int:
    summary = analyze_probe_file(args.input)
    if args.output is not None:
        _write_json(Path(args.output), summary)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    capture_parser = commands.add_parser("capture", help="Capture raw Quest packets")
    capture_parser.add_argument(
        "--config", type=Path, default=Path("configs/tracking_meta_quest.yaml")
    )
    capture_parser.add_argument("--duration-s", type=float, default=60.0)
    capture_parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New or existing directory for JSONL and summaries",
    )
    capture_parser.add_argument(
        "--flush-every",
        type=int,
        default=1,
        help="Flush after this many packets; zero flushes only when closed",
    )
    capture_parser.set_defaults(func=capture)

    analyze_parser = commands.add_parser("analyze", help="Analyze captured JSONL")
    analyze_parser.add_argument("input", type=Path)
    analyze_parser.add_argument("--output", type=Path)
    analyze_parser.set_defaults(func=analyze)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
