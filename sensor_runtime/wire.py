"""Nonblocking capacity-one UNIX transport for complete sensor batches."""

from __future__ import annotations

import json
import math
import os
import socket
import struct
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from .core import LatestOnlySlot, SensorBatch, SensorFault


WIRE_SCHEMA = 3
MAX_PACKET_BYTES = 64 * 1024 * 1024
MAX_HEADER_BYTES = 1024 * 1024
MAGIC = b"I01RW3\x00\x00"
ALLOWED_DTYPES = frozenset({"|u1", "|i1", "|b1", "<u2", "<i2", "<u4", "<i4", "<u8", "<i8", "<f4", "<f8"})
MAX_ARRAY_ELEMENTS = 32 * 1024 * 1024


def _pack_value(value: Any, segments: list[bytes], offset: list[int]) -> Any:
    import numpy as np

    if isinstance(value, np.ndarray):
        dtype = value.dtype
        if dtype.byteorder == ">" or (dtype.byteorder == "=" and not np.little_endian):
            value = value.astype(dtype.newbyteorder("<"), copy=False)
        array = np.ascontiguousarray(value)
        dtype_name = array.dtype.str
        if dtype_name not in ALLOWED_DTYPES:
            raise TypeError(f"unsupported wire ndarray dtype: {dtype_name}")
        if array.ndim <= 0 or array.ndim > 6 or array.size <= 0 or array.size > MAX_ARRAY_ELEMENTS:
            raise ValueError("wire ndarray dimensions/elements are invalid")
        raw = array.tobytes(order="C")
        descriptor = {
            "dtype": dtype_name,
            "shape": [int(part) for part in array.shape],
            "offset": offset[0],
            "nbytes": len(raw),
        }
        segments.append(raw)
        offset[0] += len(raw)
        return {"$array": descriptor}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        descriptor = {"offset": offset[0], "nbytes": len(value)}
        segments.append(value)
        offset[0] += len(value)
        return {"$bytes": descriptor}
    if isinstance(value, Mapping):
        return {str(key): _pack_value(item, segments, offset) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_pack_value(item, segments, offset) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported wire value: {type(value).__name__}")


def _descriptor(value: Any, *, array: bool) -> tuple[int, int, Any, Any]:
    import numpy as np

    expected = {"dtype", "shape", "offset", "nbytes"} if array else {"offset", "nbytes"}
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("wire segment descriptor has missing/unexpected fields")
    offset, nbytes = value["offset"], value["nbytes"]
    if isinstance(offset, bool) or isinstance(nbytes, bool) or not isinstance(offset, int) or not isinstance(nbytes, int) or offset < 0 or nbytes < 0:
        raise ValueError("wire segment offset/nbytes is invalid")
    if not array:
        return offset, nbytes, None, None
    dtype_name, shape = value["dtype"], value["shape"]
    if dtype_name not in ALLOWED_DTYPES:
        raise ValueError("wire array dtype is not allowed")
    if not isinstance(shape, list) or not 1 <= len(shape) <= 6 or any(
        isinstance(part, bool) or not isinstance(part, int) or part <= 0 for part in shape
    ):
        raise ValueError("wire array shape is invalid")
    elements = math.prod(shape)
    dtype = np.dtype(dtype_name)
    if elements > MAX_ARRAY_ELEMENTS or nbytes != elements * dtype.itemsize:
        raise ValueError("wire array shape/dtype does not match nbytes")
    return offset, nbytes, dtype, tuple(shape)


def _collect_descriptors(value: Any, output: list[tuple[int, int]]) -> None:
    if isinstance(value, dict) and set(value) == {"$array"}:
        offset, nbytes, _dtype, _shape = _descriptor(value["$array"], array=True)
        output.append((offset, nbytes))
    elif isinstance(value, dict) and set(value) == {"$bytes"}:
        offset, nbytes, _dtype, _shape = _descriptor(value["$bytes"], array=False)
        output.append((offset, nbytes))
    elif isinstance(value, dict):
        for item in value.values():
            _collect_descriptors(item, output)
    elif isinstance(value, list):
        for item in value:
            _collect_descriptors(item, output)


def _unpack_value(value: Any, raw: memoryview) -> Any:
    import numpy as np

    if isinstance(value, dict) and set(value) == {"$array"}:
        offset, nbytes, dtype, shape = _descriptor(value["$array"], array=True)
        return np.frombuffer(raw[offset : offset + nbytes], dtype=dtype).reshape(shape).copy()
    if isinstance(value, dict) and set(value) == {"$bytes"}:
        offset, nbytes, _dtype, _shape = _descriptor(value["$bytes"], array=False)
        return bytes(raw[offset : offset + nbytes])
    if isinstance(value, dict):
        return {key: _unpack_value(item, raw) for key, item in value.items()}
    if isinstance(value, list):
        return [_unpack_value(item, raw) for item in value]
    return value


def encode_batch(batch: SensorBatch) -> bytes:
    segments: list[bytes] = []
    offset = [0]
    body = {
        "schema_version": WIRE_SCHEMA,
        "stamp_ns": batch.stamp_ns,
        "generation": batch.generation,
        "sequence": batch.sequence,
        "payloads": _pack_value(batch.payloads, segments, offset),
        "stream_stamps_ns": dict(batch.stream_stamps_ns),
        "safe_stop": _pack_value(batch.safe_stop, segments, offset),
        "reset_reason": batch.reset_reason,
    }
    header = json.dumps(body, separators=(",", ":"), sort_keys=True, allow_nan=False).encode("utf-8")
    if not header or len(header) > MAX_HEADER_BYTES:
        raise ValueError("wire metadata header size is invalid")
    data = MAGIC + struct.pack("!I", len(header)) + header + b"".join(segments)
    if len(data) > MAX_PACKET_BYTES:
        raise ValueError("wire packet size is invalid")
    return data


def decode_batch(data: bytes) -> SensorBatch:
    if len(data) < len(MAGIC) + 4 or len(data) > MAX_PACKET_BYTES or not data.startswith(MAGIC):
        raise ValueError("wire packet size is invalid")
    header_size = struct.unpack("!I", data[len(MAGIC) : len(MAGIC) + 4])[0]
    header_start = len(MAGIC) + 4
    raw_start = header_start + header_size
    if header_size <= 0 or header_size > MAX_HEADER_BYTES or raw_start > len(data):
        raise ValueError("wire metadata header size is invalid")
    def reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite JSON constant is forbidden: {value}")

    body = json.loads(
        data[header_start:raw_start].decode("utf-8"), parse_constant=reject_constant
    )
    if not isinstance(body, dict):
        raise ValueError("wire batch metadata must be a JSON object")
    if body.get("schema_version") != WIRE_SCHEMA:
        raise ValueError("unsupported wire schema")
    if set(body) != {"schema_version", "stamp_ns", "generation", "sequence", "payloads", "stream_stamps_ns", "safe_stop", "reset_reason"}:
        raise ValueError("wire batch metadata has missing/unexpected fields")
    for name in ("stamp_ns", "generation", "sequence"):
        if isinstance(body[name], bool) or not isinstance(body[name], int):
            raise ValueError(f"wire {name} must be a non-boolean integer")
    raw = memoryview(data)[raw_start:]
    descriptors: list[tuple[int, int]] = []
    _collect_descriptors(body["payloads"], descriptors)
    _collect_descriptors(body["safe_stop"], descriptors)
    expected_offset = 0
    for offset, nbytes in sorted(descriptors):
        if offset != expected_offset:
            raise ValueError("wire raw segments overlap or contain a hole")
        expected_offset += nbytes
    if expected_offset != len(raw):
        raise ValueError("wire raw segment total does not match packet length")
    return SensorBatch(
        stamp_ns=int(body["stamp_ns"]),
        generation=int(body["generation"]),
        sequence=int(body["sequence"]),
        payloads=_unpack_value(body["payloads"], raw),
        stream_stamps_ns=body["stream_stamps_ns"],
        safe_stop=_unpack_value(body["safe_stop"], raw),
        reset_reason=body.get("reset_reason"),
    )


def _read_exact(connection: socket.socket, size: int) -> bytes | None:
    chunks: list[bytes] = []
    while size:
        chunk = connection.recv(size)
        if not chunk:
            return None
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


class LatestBatchEmitter:
    """Simulation thread submits only to a capacity-one slot; I/O is separate."""

    def __init__(self, socket_path: Path, *, send_timeout_sec: float = 0.2) -> None:
        if (
            isinstance(send_timeout_sec, bool)
            or not isinstance(send_timeout_sec, (int, float))
            or not math.isfinite(float(send_timeout_sec))
            or not 0.0 < float(send_timeout_sec) <= 5.0
        ):
            raise ValueError("sensor wire send timeout must be within (0, 5] seconds")
        self.socket_path = socket_path
        self.send_timeout_sec = float(send_timeout_sec)
        self.slot = LatestOnlySlot()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="sensor-wire-emitter", daemon=True)
        self._connection_lock = threading.Lock()
        self._connection: socket.socket | None = None
        self._started = False
        self.fault: str | None = None
        self._last_offered: tuple[int, int] | None = None
        self._last_submitted: tuple[int, int] | None = None
        self._last_sent: tuple[int, int] | None = None
        self.sent_count = 0
        self.connection_failures = 0
        self.serialization_failures = 0

    def start(self) -> None:
        self._started = True
        self._thread.start()

    def reset(self, generation: int) -> None:
        self.slot.reset(generation)

    def submit(self, batch: SensorBatch) -> None:
        self.raise_if_failed()
        outcome = self.slot.offer(batch)
        self._last_offered = (batch.generation, batch.sequence)
        if outcome != "barrier_dropped":
            self._last_submitted = (batch.generation, batch.sequence)

    def raise_if_failed(self) -> None:
        if self.fault:
            raise RuntimeError(self.fault)
        if self._started and not self._thread.is_alive() and not self._stop.is_set():
            raise RuntimeError("sensor emitter thread died")

    def _connect(self) -> socket.socket:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        # Socket discovery remains frozen at the strict 0.2 s bound.  Only an
        # established connection may use the explicit profile-bound send wait.
        connection.settimeout(0.2)
        try:
            connection.connect(str(self.socket_path))
            connection.settimeout(self.send_timeout_sec)
        except BaseException:
            connection.close()
            raise
        return connection

    def _run(self) -> None:
        connection: socket.socket | None = None
        try:
            while not self._stop.is_set():
                batch = self.slot.take(0.05)
                if batch is None:
                    continue
                try:
                    packet = encode_batch(batch)
                except BaseException as exc:
                    self.serialization_failures += 1
                    self.fault = f"serialization_failed: {type(exc).__name__}: {exc}"
                    return
                try:
                    if connection is None:
                        connection = self._connect()
                        with self._connection_lock:
                            self._connection = connection
                    connection.sendall(struct.pack("!I", len(packet)) + packet)
                    self.sent_count += 1
                    self._last_sent = (batch.generation, batch.sequence)
                except OSError as exc:
                    self.connection_failures += 1
                    if connection is not None:
                        connection.close()
                    connection = None
                    with self._connection_lock:
                        self._connection = None
                    self.fault = f"bounded_socket_send_failed: {type(exc).__name__}: {exc}"
                    return
        except BaseException as exc:
            if not self._stop.is_set():
                self.fault = f"emitter_thread_failed: {type(exc).__name__}: {exc}"
        finally:
            if connection is not None:
                connection.close()
            with self._connection_lock:
                self._connection = None

    def flush(self, timeout_sec: float = 2.0) -> dict[str, Any]:
        deadline = time.monotonic() + timeout_sec
        while self._last_sent != self._last_submitted:
            self.raise_if_failed()
            if time.monotonic() >= deadline:
                raise TimeoutError("latest sensor batch did not receive a send acknowledgement")
            time.sleep(0.005)
        return {
            "send_timeout_sec": self.send_timeout_sec,
            "accepted_count": self.slot.accepted_count,
            "overwrite_count": self.slot.overwrite_count,
            "reset_clear_count": self.slot.reset_clear_count,
            "barrier_drop_count": self.slot.barrier_drop_count,
            "sent_count": self.sent_count,
            "last_offered": self._last_offered,
            "last_submitted": self._last_submitted,
            "last_sent": self._last_sent,
            "thread_alive": self._thread.is_alive(),
            "fault": self.fault,
        }

    def close(self) -> None:
        self._stop.set()
        with self._connection_lock:
            if self._connection is not None:
                try:
                    self._connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        self.slot.close()
        if self._started:
            self._thread.join(2.0)
            if self._thread.is_alive():
                raise RuntimeError("sensor emitter did not stop within 2 seconds")
            if self.fault:
                raise RuntimeError(self.fault)


class LatestBatchServer:
    """One reconnecting producer feeding a guarded capacity-one receive slot."""

    def __init__(self, socket_path: Path) -> None:
        self.socket_path = socket_path
        self.slot = LatestOnlySlot()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="sensor-wire-server", daemon=True)
        self.received_count = 0
        self.fault: str | None = None
        self._connection_lock = threading.Lock()
        self._connection: socket.socket | None = None

    def start(self) -> None:
        if self.socket_path.exists():
            raise FileExistsError(f"refusing stale sensor socket: {self.socket_path}")
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self._thread.start()
        if not self._ready.wait(2.0):
            raise TimeoutError("sensor socket did not bind")
        if self.fault:
            raise RuntimeError(self.fault)

    def _accept_batch(self, batch: SensorBatch) -> None:
        current = self.slot.generation
        if current < 0:
            if batch.generation != 0 or batch.sequence != 0:
                raise SensorFault("stream_did_not_begin_at_generation_zero")
            self.slot.reset(0)
        elif batch.generation > current:
            if batch.generation != current + 1 or batch.sequence != 0:
                raise SensorFault("generation_jump_or_missing_reset_boundary")
            self.slot.reset(batch.generation)
        self.slot.offer(batch)

    def _run(self) -> None:
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            listener.bind(str(self.socket_path))
            os.chmod(self.socket_path, 0o600)
            listener.listen(1)
            listener.settimeout(0.1)
            self._ready.set()
            while not self._stop.is_set():
                try:
                    connection, _ = listener.accept()
                except socket.timeout:
                    continue
                with connection:
                    # Reads may legitimately pause at the terminal snapshot
                    # boundary.  close() interrupts this blocking receive with
                    # shutdown(), so no receive timeout is needed for bounded
                    # teardown and no partial packet can be discarded.
                    connection.settimeout(None)
                    with self._connection_lock:
                        self._connection = connection
                    while not self._stop.is_set():
                        header = _read_exact(connection, 4)
                        if header is None:
                            break
                        size = struct.unpack("!I", header)[0]
                        if size <= 0 or size > MAX_PACKET_BYTES:
                            raise ValueError("invalid length-prefixed packet")
                        packet = _read_exact(connection, size)
                        if packet is None:
                            break
                        self._accept_batch(decode_batch(packet))
                        self.received_count += 1
                    with self._connection_lock:
                        self._connection = None
        except BaseException as exc:
            if not self._stop.is_set():
                self.fault = f"{type(exc).__name__}: {exc}"
            self._ready.set()
        finally:
            listener.close()

    def close(self) -> None:
        self._stop.set()
        with self._connection_lock:
            if self._connection is not None:
                try:
                    self._connection.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass
        try:
            wake = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            wake.connect(str(self.socket_path))
            wake.close()
        except OSError:
            pass
        self._thread.join(2.0)
        if self._thread.is_alive():
            raise RuntimeError("sensor socket server did not stop within 2 seconds")
        self.slot.close()
        try:
            self.socket_path.unlink()
        except FileNotFoundError:
            pass
