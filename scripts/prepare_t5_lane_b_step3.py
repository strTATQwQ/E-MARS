#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Mapping


MODEL_VARIANT = "step3_vl_10b_bf16"
MODEL_REVISION = "5026053b0c2f5dfaa08fc2d149384162c3c8bca1"
TRANSFORMERS_VERSION = "4.57.6"
EXPECTED_PARAMETER_COUNT = 10_171_750_144
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REMOTE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")
_REMOTE_PATH_RE = re.compile(r"^/[A-Za-z0-9._/-]+$")
_HF_METADATA_PARENT = ".cache/huggingface/download"


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_files(model_path: Path) -> Iterable[Path]:
    for path in sorted(model_path.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_file():
            yield path


def _hf_metadata_provenance(model_path: Path) -> dict[str, Any]:
    metadata_paths = sorted(
        model_path.joinpath(_HF_METADATA_PARENT).glob("*.metadata"),
        key=lambda path: path.as_posix(),
    )
    if not metadata_paths:
        raise ValueError("MISSING_HF_METADATA: no Hugging Face download metadata found")
    rows = []
    for path in metadata_paths:
        try:
            with path.open("r", encoding="utf-8") as handle:
                revision = handle.readline().strip()
        except UnicodeDecodeError as exc:
            raise ValueError(f"INVALID_HF_METADATA: {path} is not UTF-8") from exc
        if not revision:
            raise ValueError(f"EMPTY_HF_METADATA: {path}")
        if revision != MODEL_REVISION:
            raise ValueError(
                f"WRONG_REVISION: {path} expected={MODEL_REVISION} actual={revision}"
            )
        rows.append(
            {
                "path": path.relative_to(model_path).as_posix(),
                "sha256": sha256_file(path),
            }
        )
    canonical = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "metadata_count": len(rows),
        "metadata_sha256": hashlib.sha256(canonical).hexdigest(),
        "revision": MODEL_REVISION,
    }


def build_content_manifest(model_path: str | Path) -> dict[str, Any]:
    root = Path(model_path).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"model path is not a directory: {root}")
    provenance = _hf_metadata_provenance(root)
    files = []
    for path in _model_files(root):
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise ValueError(f"model path contains no regular files: {root}")
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "schema_version": 1,
        "model_variant": MODEL_VARIANT,
        "revision": MODEL_REVISION,
        "transformers_version": TRANSFORMERS_VERSION,
        "hf_metadata_provenance": provenance,
        "tree_sha256": hashlib.sha256(canonical).hexdigest(),
        "file_count": len(files),
        "total_bytes": sum(int(row["bytes"]) for row in files),
        "files": files,
    }


def validate_content_manifest(value: Mapping[str, Any]) -> dict[str, Any]:
    manifest = dict(value)
    if manifest.get("schema_version") != 1:
        raise ValueError("unsupported Step3 artifact manifest schema")
    if manifest.get("model_variant") != MODEL_VARIANT:
        raise ValueError("Step3 model variant mismatch")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError("Step3 revision mismatch")
    if manifest.get("transformers_version") != TRANSFORMERS_VERSION:
        raise ValueError("Step3 transformers version mismatch")
    provenance = manifest.get("hf_metadata_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("Step3 manifest is missing Hugging Face metadata provenance")
    if provenance.get("revision") != MODEL_REVISION:
        raise ValueError("Step3 Hugging Face metadata provenance revision mismatch")
    metadata_count = int(provenance.get("metadata_count", 0))
    metadata_sha256 = str(provenance.get("metadata_sha256") or "")
    if metadata_count <= 0 or not _SHA256_RE.fullmatch(metadata_sha256):
        raise ValueError("Step3 Hugging Face metadata provenance is invalid")
    tree_hash = str(manifest.get("tree_sha256") or "")
    if not _SHA256_RE.fullmatch(tree_hash):
        raise ValueError("Step3 manifest tree_sha256 is invalid")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("Step3 manifest must contain a non-empty file list")
    paths: set[str] = set()
    total_bytes = 0
    for row in files:
        if not isinstance(row, Mapping):
            raise ValueError("Step3 manifest file entries must be objects")
        relative = str(row.get("path") or "")
        path = Path(relative)
        if (
            not relative
            or path.is_absolute()
            or ".." in path.parts
            or relative in paths
        ):
            raise ValueError(f"unsafe or duplicate Step3 manifest path: {relative!r}")
        paths.add(relative)
        size = int(row.get("bytes", -1))
        digest = str(row.get("sha256") or "")
        if size < 0 or not _SHA256_RE.fullmatch(digest):
            raise ValueError(f"invalid Step3 manifest metadata for {relative!r}")
        total_bytes += size
    if int(manifest.get("file_count", -1)) != len(files):
        raise ValueError("Step3 manifest file_count mismatch")
    if int(manifest.get("total_bytes", -1)) != total_bytes:
        raise ValueError("Step3 manifest total_bytes mismatch")
    metadata_rows = [
        {"path": row["path"], "sha256": row["sha256"]}
        for row in files
        if Path(str(row["path"])).parent.as_posix() == _HF_METADATA_PARENT
        and str(row["path"]).endswith(".metadata")
    ]
    if len(metadata_rows) != metadata_count:
        raise ValueError("Step3 Hugging Face metadata provenance count mismatch")
    metadata_canonical = json.dumps(
        metadata_rows, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if hashlib.sha256(metadata_canonical).hexdigest() != metadata_sha256:
        raise ValueError("Step3 Hugging Face metadata provenance hash mismatch")
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != tree_hash:
        raise ValueError("Step3 manifest tree_sha256 does not match its file entries")
    return manifest


def compare_manifests(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
    expected_value = validate_content_manifest(expected)
    actual_value = validate_content_manifest(actual)
    for field in (
        "revision",
        "transformers_version",
        "tree_sha256",
        "file_count",
        "total_bytes",
        "hf_metadata_provenance",
    ):
        if actual_value[field] != expected_value[field]:
            raise RuntimeError(
                f"copied Step3 artifact differs from source manifest: {field} "
                f"expected={expected_value[field]!r} actual={actual_value[field]!r}"
            )


def build_lan_rsync_command(
    *,
    source_user: str,
    source_host: str,
    source_path: str,
    destination: str | Path,
) -> list[str]:
    if not _REMOTE_COMPONENT_RE.fullmatch(source_user):
        raise ValueError("source_user contains unsafe characters")
    try:
        address = ipaddress.ip_address(source_host)
    except ValueError as exc:
        raise ValueError("source_host must be a numeric private LAN address") from exc
    if not address.is_private:
        raise ValueError("source_host must be a private LAN address")
    if not _REMOTE_PATH_RE.fullmatch(source_path) or ".." in Path(source_path).parts:
        raise ValueError("source_path must be a shell-safe absolute remote path")
    target = Path(destination).resolve()
    target.mkdir(parents=True, exist_ok=True)
    remote = f"{source_user}@{source_host}:{source_path.rstrip('/')}/"
    return [
        "rsync",
        "--archive",
        "--copy-links",
        "--partial",
        "--checksum",
        "--protect-args",
        "--human-readable",
        "--info=progress2",
        "-e",
        "ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=10 -o Compression=no",
        remote,
        f"{target}/",
    ]


def copy_over_lan(command: list[str]) -> None:
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"LAN rsync failed with returncode={result.returncode}")


def verify_clean_bf16_load(model_path: str | Path) -> dict[str, Any]:
    import transformers

    if transformers.__version__ != TRANSFORMERS_VERSION:
        raise RuntimeError(
            f"transformers version mismatch: expected={TRANSFORMERS_VERSION} actual={transformers.__version__}"
        )
    from slow_planner.step3_vl_10b import Step3VLSlowPlanner

    planner = Step3VLSlowPlanner.from_pretrained(
        Path(model_path).resolve(),
        expected_transformers_version=TRANSFORMERS_VERSION,
        require_clean_checkpoint_load=True,
        require_all_parameters_bf16=True,
    )
    try:
        health = planner.health()
        if not health.get("checkpoint_load_clean"):
            raise RuntimeError("Step3 loader did not report a clean checkpoint load")
        if int(health.get("parameter_count", 0)) != EXPECTED_PARAMETER_COUNT:
            raise RuntimeError(
                "Step3 parameter count mismatch: "
                f"expected={EXPECTED_PARAMETER_COUNT} actual={health.get('parameter_count')}"
            )
        dtype_counts = dict(health.get("parameter_dtype_counts") or {})
        if set(dtype_counts) != {"torch.bfloat16"}:
            raise RuntimeError(
                f"Step3 parameters are not exclusively BF16: {dtype_counts}"
            )
        return dict(health)
    finally:
        planner.close()


def _load_manifest(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("artifact manifest must be a JSON object")
    return validate_content_manifest(value)


def _write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Copy Step3 over the private LAN and verify the frozen clean BF16 artifact on DGX_B."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    manifest_parser = subparsers.add_parser(
        "manifest", help="Hash an already materialized source artifact."
    )
    manifest_parser.add_argument("--model-path", required=True)
    manifest_parser.add_argument("--output", required=True)

    copy_parser = subparsers.add_parser(
        "copy", help="Copy from DGX_A over LAN and verify all source hashes."
    )
    copy_parser.add_argument("--source-user", default="railgun")
    copy_parser.add_argument("--source-host", default="10.100.100.128")
    copy_parser.add_argument("--source-path", required=True)
    copy_parser.add_argument("--source-manifest", required=True)
    copy_parser.add_argument("--destination", required=True)
    copy_parser.add_argument("--output", required=True)

    verify_parser = subparsers.add_parser(
        "verify", help="Re-hash and perform the clean all-BF16 model load."
    )
    verify_parser.add_argument("--model-path", required=True)
    verify_parser.add_argument("--artifact-manifest", required=True)
    verify_parser.add_argument("--output", required=True)

    args = parser.parse_args()
    if args.command == "manifest":
        value = build_content_manifest(args.model_path)
        _write_json(args.output, value)
        print(
            json.dumps(
                {
                    key: value[key]
                    for key in ("tree_sha256", "file_count", "total_bytes")
                }
            )
        )
        return 0

    expected = _load_manifest(
        args.source_manifest if args.command == "copy" else args.artifact_manifest
    )
    if args.command == "copy":
        command = build_lan_rsync_command(
            source_user=args.source_user,
            source_host=args.source_host,
            source_path=args.source_path,
            destination=args.destination,
        )
        copy_over_lan(command)
        actual = build_content_manifest(args.destination)
        compare_manifests(expected, actual)
        report = {
            "schema_version": 1,
            "status": "COPY_VERIFIED",
            "verified_wall_time_s": time.time(),
            "model_variant": MODEL_VARIANT,
            "revision": MODEL_REVISION,
            "transformers_version": TRANSFORMERS_VERSION,
            "source": f"{args.source_user}@{args.source_host}:{args.source_path}",
            "destination": str(Path(args.destination).resolve()),
            "tree_sha256": actual["tree_sha256"],
            "file_count": actual["file_count"],
            "total_bytes": actual["total_bytes"],
            "clean_load_pending": True,
        }
    else:
        actual = build_content_manifest(args.model_path)
        compare_manifests(expected, actual)
        health = verify_clean_bf16_load(args.model_path)
        report = {
            "schema_version": 1,
            "status": "CLEAN_BF16_LOAD_VERIFIED",
            "verified_wall_time_s": time.time(),
            "model_variant": MODEL_VARIANT,
            "revision": MODEL_REVISION,
            "transformers_version": TRANSFORMERS_VERSION,
            "model_path": str(Path(args.model_path).resolve()),
            "tree_sha256": actual["tree_sha256"],
            "health": health,
        }
    _write_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
