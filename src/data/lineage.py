"""Shared helpers for materializing Casting datasets and manifests."""

from __future__ import annotations

import csv
import shutil
from collections import Counter
from pathlib import Path

from src.config import PATHS

__all__ = [
    "copy_file",
    "copy_images_by_folder",
    "project_relative",
    "resolve",
    "summary_rows",
    "write_csv",
]


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PATHS.root / path


def project_relative(path: str | Path, root: Path | None = None) -> str:
    root = PATHS.root if root is None else root
    path = Path(path)
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def write_csv(path: str | Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        raise ValueError(f"Cannot infer CSV fieldnames for empty rows: {path}")
    names = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def copy_file(source: str | Path, destination: str | Path, *, overwrite: bool = False) -> str:
    source_path = resolve(source)
    destination_path = resolve(destination)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.exists() and not overwrite:
        return "skipped_existing"
    shutil.copy2(source_path, destination_path)
    return "copied"


def summary_rows(rows: list[dict[str, object]], key: str, count_name: str = "images") -> list[dict[str, str]]:
    counts = Counter(str(row[key]) for row in rows)
    return [{key: value, count_name: str(count)} for value, count in sorted(counts.items())]


def copy_images_by_folder(
    sources: list[tuple[Path, str, dict[str, object]]],
    output_dir: str | Path,
    *,
    overwrite: bool = False,
    folder_key: str = "folder",
) -> list[dict[str, object]]:
    output_dir = resolve(output_dir)
    rows: list[dict[str, object]] = []
    for source, folder, metadata in sources:
        destination = output_dir / folder / source.name
        action = copy_file(source, destination, overwrite=overwrite)
        rows.append(
            {
                **metadata,
                folder_key: folder,
                "source_path": project_relative(source),
                "classified_path": project_relative(destination),
                "action": action,
            }
        )
    return rows





