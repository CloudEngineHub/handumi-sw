import sys
import unittest
from types import SimpleNamespace
from unittest import mock

from handumi.feetech.bus import FeetechBus, _install_reliable_packet_timeout


class _FakePacket:
    def __init__(self, *, reads=None, writes=None, pings=None):
        self.reads = list(reads or [])
        self.writes = list(writes or [])
        self.pings = list(pings or [])
        self.write_calls = 0

    def read2ByteTxRx(self, *_args):
        response = self.reads.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def write1ByteTxRx(self, *_args):
        self.write_calls += 1
        response = self.writes.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response

    def ping(self, *_args):
        response = self.pings.pop(0)
        if isinstance(response, BaseException):
            raise response
        return response


def _bus_with(packet: _FakePacket) -> FeetechBus:
    bus = FeetechBus("/dev/test")
    bus._sdk = SimpleNamespace(COMM_SUCCESS=0)
    bus._port_handler = object()
    bus._packet_handler = packet
    return bus


class FeetechBusRetryTest(unittest.TestCase):
    def test_open_configures_serial_port_exactly_once(self):
        class _Port:
            tx_time_per_byte = 0.01

            def __init__(self, name):
                self.name = name
                self.baud_calls = []

            def setBaudRate(self, baudrate):
                self.baud_calls.append(baudrate)
                return True

            def closePort(self):
                pass

        port = _Port("/dev/test")
        sdk = SimpleNamespace(
            PortHandler=lambda _name: port,
            PacketHandler=lambda _protocol: object(),
        )
        bus = FeetechBus("/dev/test", baudrate=1_000_000)

        with mock.patch.dict(sys.modules, {"scservo_sdk": sdk}):
            bus.open()

        self.assertEqual(port.baud_calls, [1_000_000])

    def test_reliable_timeout_includes_usb_scheduler_margin(self):
        class _Port:
            tx_time_per_byte = 0.01

            def getCurrentTime(self):
                return 123.0

        port = _Port()
        _install_reliable_packet_timeout(port)

        port.setPacketTimeout(8)

        self.assertEqual(port.packet_start_time, 123.0)
        self.assertAlmostEqual(port.packet_timeout, 50.11)

    def test_read_position_retries_serial_io_errors(self):
        packet = _FakePacket(
            reads=[
                OSError("device reports readiness to read but returned no data"),
                (1234, 0, 0),
            ]
        )
        bus = _bus_with(packet)

        self.assertEqual(bus.read_position(1, retry_delay_s=0), 1234)

    def test_read_position_recovers_sdk_port_after_truncated_response(self):
        class _Port:
            is_using = True

        packet = _FakePacket(reads=[IndexError(), (1234, 0, 0)])
        bus = _bus_with(packet)
        bus._port_handler = _Port()

        self.assertEqual(bus.read_position(1, retry_delay_s=0), 1234)
        self.assertFalse(bus._port_handler.is_using)

    def test_read_position_reports_last_retry_failure(self):
        packet = _FakePacket(reads=[OSError("no data"), OSError("still no data")])
        bus = _bus_with(packet)

        with self.assertRaisesRegex(RuntimeError, "OSError: still no data"):
            bus.read_position(1, retries=1, retry_delay_s=0)

    def test_write_retries_serial_io_errors(self):
        packet = _FakePacket(writes=[OSError("no data"), (0, 0)])
        bus = _bus_with(packet)

        bus._write_1_byte(1, 40, 0, "Torque_Enable", retry_delay_s=0)

        self.assertEqual(packet.write_calls, 2)

    def test_ping_treats_serial_io_errors_as_no_response(self):
        packet = _FakePacket(pings=[OSError("no data")])
        bus = _bus_with(packet)

        self.assertFalse(bus.ping(1))

    def test_ping_model_releases_port_before_retry(self):
        class _Port:
            is_using = True

            def clearPort(self):
                self.is_using = False

        packet = _FakePacket(pings=[IndexError(), (777, 0, 0)])
        bus = _bus_with(packet)
        bus._port_handler = _Port()

        self.assertEqual(bus.ping_model(1, retries=1), 777)
        self.assertFalse(bus._port_handler.is_using)


if __name__ == "__main__":
    unittest.main()
