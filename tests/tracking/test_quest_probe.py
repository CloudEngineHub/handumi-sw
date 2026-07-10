import io
import json

from handumi.tracking.meta_quest import parse_frame
from handumi.tracking.quest_probe import PROBE_SCHEMA, ProbeCapture, analyze_probe_records


def _record(seq: int, receive_ns: int, device_ns: int, *, offset: int = 10, rtt: int = 4):
    return {
        "probe_schema": PROBE_SCHEMA,
        "pc_receive_time_ns": receive_ns,
        "sync": {
            "clock_synced": True,
            "clock_offset_ns": offset,
            "rtt_ns": rtt,
        },
        "packet": {"seq": seq, "ovrTimeNs": device_ns},
    }


def test_capture_preserves_raw_packet_and_adds_host_diagnostics():
    stream = io.StringIO()
    raw = {"seq": 7, "body": {"xrTime": 123, "joints": [{"flags": 3}]}}
    frame = parse_frame(raw, pc_monotonic_ns=456)
    capture = ProbeCapture(
        stream=stream,
        metrics_provider=lambda: {"offset_ns": 20, "rtt_ns": 8},
    )

    capture.record(frame)

    envelope = json.loads(stream.getvalue())
    assert envelope["packet"] == raw
    assert envelope["pc_receive_time_ns"] == 456
    assert envelope["sync"] == {
        "clock_synced": True,
        "clock_offset_ns": 20,
        "rtt_ns": 8,
    }


def test_analysis_reports_sequence_loss_and_timing_rates():
    records = [
        _record(10, 1_000_000_000, 2_000_000_000),
        _record(11, 1_010_000_000, 2_010_000_000),
        _record(13, 1_020_000_000, 2_020_000_000),
    ]

    summary = analyze_probe_records(records)

    assert summary["packet_count"] == 3
    assert summary["receive_rate_hz"] == 100.0
    assert summary["device_rate_hz"] == 100.0
    assert summary["source_sequence"]["missing_packets"] == 1
    assert summary["source_sequence"]["loss_fraction"] == 0.25
    assert summary["receive_interarrival_ns"]["standard_deviation"] == 0.0
    assert summary["mapped_sample_age_ns"]["median"] == -1_000_000_010


def test_analysis_does_not_claim_loss_measurement_without_sender_sequence():
    summary = analyze_probe_records(
        [
            {
                "pc_receive_time_ns": 1,
                "sync": {},
                "packet": {"ovrTimeNs": 2},
            }
        ]
    )

    assert summary["source_sequence"]["available"] is False
    assert summary["source_sequence"]["missing_packets"] is None
    assert summary["source_sequence"]["loss_fraction"] is None


def test_analysis_excludes_offsets_until_clock_is_synced():
    summary = analyze_probe_records(
        [
            {
                "pc_receive_time_ns": 100,
                "sync": {
                    "clock_synced": False,
                    "clock_offset_ns": 0,
                    "rtt_ns": None,
                },
                "packet": {"ovrTimeNs": 50},
            }
        ]
    )

    assert summary["clock_offset_ns"]["sample_count"] == 0
    assert summary["mapped_sample_age_ns"]["sample_count"] == 0


def test_analysis_reports_body_sample_rate_when_diagnostic_field_is_present():
    records = [
        {"packet": {"body": {"xrTime": 1_000_000_000}}},
        {"packet": {"body": {"xrTime": 1_000_000_000}}},
        {"packet": {"body": {"xrTime": 1_020_000_000}}},
        {"packet": {"body": {"xrTime": 1_040_000_000}}},
    ]

    summary = analyze_probe_records(records)

    assert summary["body_update_rate_hz"] == 50.0
