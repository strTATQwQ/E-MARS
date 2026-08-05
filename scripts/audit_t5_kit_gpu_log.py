#!/usr/bin/env python3
"""Fail closed unless every Kit GPU table selects exactly the expected GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path


ROW = re.compile(
    r"^\|\s*(?P<index>\d+)\s*\|\s*(?P<name>[^|]+?)\s*\|"
    r"\s*(?P<active>[^|]*?)\s*\|"
)


def _uuid_prefix(value: str) -> str:
    normalized = value.removeprefix("GPU-").lower()
    return normalized.split("-", 1)[0]


def _bus_component(value: str) -> str:
    # nvidia-smi reports e.g. 00000000:81:00.0; Kit's compact table reports 81.
    fields = value.lower().split(":")
    if len(fields) < 2:
        raise ValueError(f"invalid PCI bus id: {value!r}")
    return fields[-2]


def _identity_cells(line: str) -> tuple[str | None, str | None]:
    """Extract Kit's compact UUID prefix and PCI bus from any table row.

    Kit versions differ on whether identity is printed on the primary GPU row
    or one/two indented continuation rows.  Match values by shape instead of
    relying on a fixed column offset while still refusing ambiguous identity.
    """

    cells = [cell.strip() for cell in line.split("|")[1:-1]]
    uuid_prefix: str | None = None
    bus_component: str | None = None
    for cell in cells:
        compact = cell.removesuffix("..").removeprefix("GPU-").lower()
        if re.fullmatch(r"[0-9a-f]{8,32}(?:-[0-9a-f-]+)?", compact):
            uuid_prefix = compact.split("-", 1)[0]
        if re.fullmatch(r"[0-9a-f]{1,2}", compact):
            # Some Kit builds omit the leading zero ("1" for PCI bus 01).
            bus_component = compact.zfill(2)
        elif re.fullmatch(
            r"[0-9a-f]{8}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-9]", compact
        ):
            bus_component = _bus_component(compact)
    return uuid_prefix, bus_component


def audit(
    log_path: Path,
    expected: int,
    expected_uuid: str,
    expected_pci_bus_id: str,
) -> dict[str, object]:
    if not log_path.is_file():
        raise FileNotFoundError(log_path)
    tables: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] | None = None
    with log_path.open("r", encoding="utf-8", errors="replace") as stream:
        for raw in stream:
            line = raw.rstrip("\r\n")
            if "| GPU |" in line and "| Active |" in line:
                if current is not None:
                    tables.append(current)
                current = []
                continue
            if current is None:
                continue
            match = ROW.match(line)
            if match:
                active_text = match.group("active").strip()
                compact_uuid, compact_bus = _identity_cells(line)
                current.append(
                    {
                        "index": int(match.group("index")),
                        "name": match.group("name").strip(),
                        "active_text": active_text,
                        "active": active_text.startswith("Yes"),
                        "uuid_prefix": compact_uuid,
                        "pci_bus_component": compact_bus,
                    }
                )
            elif current and line.startswith("|="):
                tables.append(current)
                current = None
            elif current and line.startswith("|"):
                compact_uuid, compact_bus = _identity_cells(line)
                if compact_uuid is not None:
                    current[-1]["uuid_prefix"] = compact_uuid
                if compact_bus is not None:
                    current[-1]["pci_bus_component"] = compact_bus
    if current is not None:
        tables.append(current)

    active_by_table = [
        [int(row["index"]) for row in table if bool(row["active"])]
        for table in tables
    ]
    expected_uuid_prefix = _uuid_prefix(expected_uuid)
    expected_bus_component = _bus_component(expected_pci_bus_id)
    active_rows = [
        [row for row in table if bool(row["active"])] for table in tables
    ]
    checks = {
        "gpu_table_present": bool(tables),
        "every_table_has_rows": bool(tables) and all(tables),
        "exact_expected_gpu_active": bool(tables)
        and all(active == [expected] for active in active_by_table),
        "no_other_gpu_active": bool(tables)
        and all(
            not bool(row["active"]) or int(row["index"]) == expected
            for table in tables
            for row in table
        ),
        "active_gpu_uuid_matches": bool(tables)
        and all(
            len(rows) == 1
            and rows[0].get("uuid_prefix") == expected_uuid_prefix
            for rows in active_rows
        ),
        "active_gpu_pci_bus_matches": bool(tables)
        and all(
            len(rows) == 1
            and rows[0].get("pci_bus_component") == expected_bus_component
            for rows in active_rows
        ),
    }
    digest = hashlib.sha256()
    with log_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "schema_version": 1,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "expected_active_gpu": expected,
        "expected_physical_gpu_uuid": expected_uuid,
        "expected_pci_bus_id": expected_pci_bus_id,
        "log_path": str(log_path.resolve()),
        "log_sha256": digest.hexdigest(),
        "table_count": len(tables),
        "tables": tables,
        "active_gpu_indices_by_table": active_by_table,
        "checks": checks,
        "recorded_unix": time.time(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--expected-active-gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--expected-gpu-uuid", required=True)
    parser.add_argument("--expected-pci-bus-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(
        args.log,
        args.expected_active_gpu,
        args.expected_gpu_uuid,
        args.expected_pci_bus_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
