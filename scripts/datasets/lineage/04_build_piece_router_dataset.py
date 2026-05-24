"""Build a piece-level classifier dataset from the final pattern-router folders."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.lineage import project_relative, summary_rows, write_csv


PIECE_RULES = {
    "C1_P1_": "C1_piece_B",
    "C1_P2_": "C1_piece_B",
    "C1_P3_": "C1_piece_B",
    "C1_P4_": "C1_piece_A",
    "C2_family_A_": "C2_piece_A_long_black_body",
    "C2_family_B_": "C2_piece_B_rectangular_port",
    "C2_family_C_": "C2_piece_C_machined_circular_zone",
    "C2_family_D_": "C2_piece_C_machined_circular_zone",
    "C3_": "C3_piece",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Materialize a piece-level casting classifier dataset.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("data/classified/Casting_pattern_router/train/good"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/classified/Casting_piece_router/train/good"),
    )
    parser.add_argument(
        "--manifest-csv",
        type=Path,
        default=Path("reports/tables/summary/casting_piece_router_train_manifest.csv"),
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("reports/tables/summary/casting_piece_router_train_summary.csv"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def piece_folder(pattern_folder: str) -> str:
    for prefix, piece in PIECE_RULES.items():
        if pattern_folder.startswith(prefix):
            return piece
    raise ValueError(f"No piece mapping for pattern folder: {pattern_folder}")


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str]] = []

    for pattern_dir in sorted(path for path in args.input_dir.iterdir() if path.is_dir()):
        piece = piece_folder(pattern_dir.name)
        destination_dir = args.output_dir / piece
        destination_dir.mkdir(parents=True, exist_ok=True)
        for source in sorted(pattern_dir.glob("*.jpg")):
            destination = destination_dir / source.name
            if destination.exists() and not args.overwrite:
                action = "skipped_existing"
            else:
                import shutil

                shutil.copy2(source, destination)
                action = "copied"
            rows.append(
                {
                    "source_pattern_folder": pattern_dir.name,
                    "piece_folder": piece,
                    "source_path": project_relative(source),
                    "classified_path": project_relative(destination),
                    "action": action,
                }
            )

    write_csv(args.manifest_csv, rows)
    write_csv(args.output_dir / "classified_manifest.csv", rows)

    summary = summary_rows(rows, "piece_folder")
    write_csv(args.summary_csv, summary)
    write_csv(args.output_dir / "classified_summary.csv", summary)

    print(f"Wrote {len(rows)} rows to {args.output_dir}")
    print(f"Pieces: {len(summary)}")
    for row in summary:
        print(f"{row['piece_folder']}: {row['images']}")


if __name__ == "__main__":
    main()






