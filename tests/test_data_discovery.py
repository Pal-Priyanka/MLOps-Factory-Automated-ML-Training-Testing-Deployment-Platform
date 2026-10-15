"""Unit tests for dataset discovery, exclusion rules, and cryptographic checksums."""

from pathlib import Path
import pytest
from mlops_factory.data.discovery import (
    compute_file_sha256,
    discover_dataset,
    inspect_dataset_provenance,
)


def test_discover_dataset_finds_telco_csv(tmp_path):
    """Validates that recursive discovery correctly locates dataset file in workspace or search root."""
    target_csv = Path("telco.csv")
    if target_csv.is_file():
        found_path = discover_dataset("telco.csv")
    else:
        # For CI environments where raw datasets are intentionally excluded from Git
        mock_file = tmp_path / "telco.csv"
        mock_file.write_text("Customer ID,Gender\n1,Male\n", encoding="utf-8")
        found_path = discover_dataset("telco.csv", search_root=tmp_path)

    assert found_path.exists()
    assert found_path.is_file()
    assert found_path.name == "telco.csv"


def test_discover_dataset_raises_on_missing_file(tmp_path):
    """Ensures FileNotFoundError is raised with descriptive error message if file is absent."""
    with pytest.raises(FileNotFoundError, match="could not be located"):
        discover_dataset("non_existent_dataset_12345.csv", search_root=tmp_path)


def test_compute_file_sha256_is_deterministic(tmp_path):
    """Verifies that cryptographic checksum computation is strictly deterministic."""
    sample_file = tmp_path / "sample.txt"
    sample_file.write_text("mlops-factory-test-content-2026", encoding="utf-8")

    hash1 = compute_file_sha256(sample_file)
    hash2 = compute_file_sha256(sample_file)

    assert hash1 == hash2
    assert len(hash1) == 64
    assert isinstance(hash1, str)


def test_inspect_dataset_provenance(tmp_path):
    """Validates extraction of provenance metadata on a known CSV."""
    csv_file = tmp_path / "test.csv"
    csv_file.write_text("A,B,C\n1,2,3\n4,5,6\n", encoding="utf-8")

    meta = inspect_dataset_provenance(csv_file)
    assert meta.filename == "test.csv"
    assert meta.row_count == 2
    assert meta.column_count == 3
    assert meta.columns == ["A", "B", "C"]
    assert len(meta.sha256_checksum) == 64


def test_empty_dataset_raises_value_error(tmp_path):
    """Validates that a 0-byte file raises an immediate ValueError."""
    empty_file = tmp_path / "empty.csv"
    empty_file.touch()

    with pytest.raises(ValueError, match="empty"):
        inspect_dataset_provenance(empty_file)
