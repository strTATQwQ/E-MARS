import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/materialize_t5_screen_dataset.py"


def write_source(root: Path) -> tuple[Path, str, list[str]]:
    path = root / "val_unseen/val_unseen.json.gz"
    path.parent.mkdir(parents=True)
    episodes = [
        {"trajectory_id": 100 + index, "episode_id": index, "payload": index}
        for index in range(5)
    ]
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        json.dump({"episodes": episodes, "metadata": {"frozen": True}}, stream)
    return (
        path,
        hashlib.sha256(path.read_bytes()).hexdigest(),
        [f"{100 + index}_{index}" for index in range(5)],
    )


@pytest.mark.parametrize("count", (1, 3))
def test_materializes_deterministic_first_n_with_audit(tmp_path: Path, count: int) -> None:
    source_root = tmp_path / "source"
    source, source_sha, keys = write_source(source_root)
    output_root = tmp_path / f"screen{count}"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-root",
            str(source_root),
            "--output-root",
            str(output_root),
            "--count",
            str(count),
            "--expected-source-sha256",
            source_sha,
            "--expected-episode-keys",
            ",".join(keys),
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    audit = json.loads(completed.stdout)
    assert audit["status"] == "PASS"
    assert audit["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert audit["selected_episode_keys"] == keys[:count]
    with gzip.open(
        output_root / "val_unseen/val_unseen.json.gz", "rt", encoding="utf-8"
    ) as stream:
        selected = json.load(stream)
    assert [row["episode_id"] for row in selected["episodes"]] == list(range(count))
    assert selected["metadata"] == {"frozen": True}


def test_rejects_wrong_source_hash_without_creating_output(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    _, _, keys = write_source(source_root)
    output_root = tmp_path / "screen"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-root",
            str(source_root),
            "--output-root",
            str(output_root),
            "--count",
            "1",
            "--expected-source-sha256",
            "0" * 64,
            "--expected-episode-keys",
            ",".join(keys),
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode != 0
    assert not output_root.exists()
