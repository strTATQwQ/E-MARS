from pathlib import Path

from scripts.audit_t5_kit_gpu_log import audit


def _log(tmp_path: Path, active0: str, active1: str) -> Path:
    path = tmp_path / "kit.log"
    path.write_text(
        "\n".join(
            [
                "| GPU | Name | Active | LDA | GPU Memory | Vendor-ID | LUID |",
                f"| 0 | NVIDIA GeForce RTX 5060 Ti | {active0} | | 16557 MB | 10de | 0 |",
                "|   |                               |        | |          | 2d04 | 0d6815f7.. |",
                "|   |                               |        | |          | 81 | |",
                f"| 1 | NVIDIA GeForce RTX 5060 Ti | {active1} | | 16557 MB | 10de | 0 |",
                "|   |                               |        | |          | 2d04 | 78b303b3.. |",
                "|   |                               |        | |          | c1 | |",
                "|================================================================================|",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_gpu0_table_passes_only_for_lane_a(tmp_path: Path) -> None:
    path = _log(tmp_path, "Yes: 0", "")
    assert audit(path, 0, "GPU-0d6815f7-b5e3-5561-3da9-d73bc647c702", "00000000:81:00.0")["status"] == "PASS"
    assert audit(path, 1, "GPU-78b303b3-d460-2429-7e65-f545b85ee64a", "00000000:c1:00.0")["status"] == "FAIL"


def test_gpu1_table_passes_only_for_lane_b(tmp_path: Path) -> None:
    path = _log(tmp_path, "", "Yes: 0")
    result = audit(path, 1, "GPU-78b303b3-d460-2429-7e65-f545b85ee64a", "00000000:c1:00.0")
    assert result["status"] == "PASS"
    assert result["active_gpu_indices_by_table"] == [[1]]


def test_multi_gpu_or_missing_table_fails(tmp_path: Path) -> None:
    assert audit(_log(tmp_path, "Yes: 0", "Yes: 1"), 0, "GPU-0d6815f7-b5e3-5561-3da9-d73bc647c702", "00000000:81:00.0")["status"] == "FAIL"
    missing = tmp_path / "missing-table.log"
    missing.write_text("Isaac started without a GPU table\n", encoding="utf-8")
    assert audit(missing, 0, "GPU-0d6815f7-b5e3-5561-3da9-d73bc647c702", "00000000:81:00.0")["status"] == "FAIL"


def test_active_index_with_wrong_physical_identity_fails(tmp_path: Path) -> None:
    path = _log(tmp_path, "", "Yes: 0")
    result = audit(
        path,
        1,
        "GPU-0d6815f7-b5e3-5561-3da9-d73bc647c702",
        "00000000:81:00.0",
    )
    assert result["status"] == "FAIL"
    assert result["checks"]["active_gpu_uuid_matches"] is False


def test_matching_index_and_uuid_with_wrong_pci_bus_fails(tmp_path: Path) -> None:
    path = _log(tmp_path, "Yes: 0", "")
    result = audit(
        path,
        0,
        "GPU-0d6815f7-b5e3-5561-3da9-d73bc647c702",
        "00000000:c1:00.0",
    )
    assert result["status"] == "FAIL"
    assert result["checks"]["active_gpu_uuid_matches"] is True
    assert result["checks"]["active_gpu_pci_bus_matches"] is False


def test_active_row_without_uuid_and_bus_continuations_fails(tmp_path: Path) -> None:
    path = tmp_path / "identity-missing.log"
    path.write_text(
        "| GPU | Name | Active | LDA | GPU Memory | Vendor-ID | LUID |\n"
        "| 0 | NVIDIA GeForce RTX 5060 Ti | Yes: 0 | | 16557 MB | 10de | 0 |\n"
        "|================================================================================|\n",
        encoding="utf-8",
    )
    result = audit(
        path,
        0,
        "GPU-0d6815f7-b5e3-5561-3da9-d73bc647c702",
        "00000000:81:00.0",
    )
    assert result["status"] == "FAIL"
    assert result["checks"]["exact_expected_gpu_active"] is True
    assert result["checks"]["active_gpu_uuid_matches"] is False
    assert result["checks"]["active_gpu_pci_bus_matches"] is False


def test_identity_on_primary_row_is_supported_without_fixed_columns(
    tmp_path: Path,
) -> None:
    path = tmp_path / "primary-identity.log"
    path.write_text(
        "| GPU | Name | Active | LDA | GPU Memory | Vendor-ID | LUID | UUID | PCI |\n"
        "| 0 | NVIDIA GeForce RTX 5060 Ti | Yes: 0 | | 16557 MB | 10de | 0 | 0d6815f7.. | 00000000:81:00.0 |\n"
        "|================================================================================|\n",
        encoding="utf-8",
    )
    result = audit(
        path,
        0,
        "GPU-0d6815f7-b5e3-5561-3da9-d73bc647c702",
        "00000000:81:00.0",
    )
    assert result["status"] == "PASS"


def test_single_digit_kit_bus_id_matches_zero_padded_nvidia_smi_id(
    tmp_path: Path,
) -> None:
    path = _log(tmp_path, "Yes: 0", "")
    text = path.read_text(encoding="utf-8").replace("| 81 | |", "| 1 | |", 1)
    path.write_text(text, encoding="utf-8")
    result = audit(
        path,
        0,
        "GPU-0d6815f7-b5e3-5561-3da9-d73bc647c702",
        "00000000:01:00.0",
    )
    assert result["status"] == "PASS"
    assert result["tables"][0][0]["pci_bus_component"] == "01"
