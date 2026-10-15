"""Dataset discovery, fingerprinting, and physical validation."""

import hashlib
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd


@dataclass
class DatasetMetadata:
    """Metadata and provenance fingerprint for a dataset file."""
    filepath: str
    filename: str
    sha256_checksum: str
    size_bytes: int
    row_count: int
    column_count: int
    columns: List[str]

    def to_dict(self) -> Dict:
        return asdict(self)


def compute_file_sha256(filepath: Path) -> str:
    """Computes SHA-256 hash of a file deterministically in chunks."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def discover_dataset(
    target_filename: str = "telco.csv",
    search_root: Optional[Path] = None,
    excluded_dirs: Optional[List[str]] = None,
) -> Path:
    """
    Recursively discovers the dataset file within search_root, strictly excluding
    virtual environments, caches, git internals, and generated artifacts.
    """
    if search_root is None:
        search_root = Path.cwd()

    if excluded_dirs is None:
        excluded_dirs = [
            ".git",
            ".venv",
            "venv",
            "ENV",
            "env",
            "__pycache__",
            ".pytest_cache",
            "artifacts",
            "dist",
            "build",
            ".idea",
            ".vscode",
        ]

    # Direct check first
    direct_path = search_root / target_filename
    if direct_path.is_file():
        return direct_path

    # Recursive search
    for root, dirs, files in os.walk(search_root):
        # Prune excluded directories in-place
        dirs[:] = [d for d in dirs if d not in excluded_dirs and not d.startswith(".")]

        if target_filename in files:
            found_path = Path(root) / target_filename
            return found_path

    raise FileNotFoundError(
        f"Dataset '{target_filename}' could not be located in '{search_root}'. "
        f"Please verify the file exists and is not placed in an excluded directory."
    )


def inspect_dataset_provenance(filepath: Path) -> DatasetMetadata:
    """Computes cryptographic fingerprint and basic tabular properties."""
    if not filepath.exists() or not filepath.is_file():
        raise FileNotFoundError(f"Dataset path does not exist or is not a regular file: {filepath}")

    size_bytes = filepath.stat().st_size
    if size_bytes == 0:
        raise ValueError(f"Dataset file is empty (0 bytes): {filepath}")

    checksum = compute_file_sha256(filepath)

    # Read schema and row count safely without mutating
    df = pd.read_csv(filepath)
    row_count, col_count = df.shape

    try:
        display_path = filepath.relative_to(Path.cwd()).as_posix()
    except ValueError:
        display_path = filepath.name

    return DatasetMetadata(
        filepath=display_path,
        filename=filepath.name,
        sha256_checksum=checksum,
        size_bytes=size_bytes,
        row_count=row_count,
        column_count=col_count,
        columns=df.columns.tolist(),
    )
