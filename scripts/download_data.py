"""Téléchargement et extraction des datasets MVTec AD + HSS-IAD.

Usage :
    uv run python scripts/download_data.py            # télécharge MVTec AD
    uv run python scripts/download_data.py --hss-iad  # télécharge HSS-IAD aussi
    uv run python scripts/download_data.py --check    # vérifie l'intégrité sans télécharger

Datasets :
- MVTec AD : 4.9 GB, https://www.mvtec.com/company/research/datasets/mvtec-ad
  Requiert une inscription manuelle. Lien direct à mettre à jour si MVTec change.
- HSS-IAD : https://github.com/Cynthia2024/HSS-IAD (Casting + KolektorSDD + STEEL etc.)
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import PATHS

# --- Configuration des datasets ---
DATASETS = {
    "mvtec": {
        "url": "https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz",
        "dest_archive": PATHS.processed_dir.parent / "raw" / "mvtec_ad.tar.xz",
        "dest_dir": PATHS.mvtec_dir,
        "size_gb": 4.9,
        "checksum_sha256": "eefca59f2cede9c3fc5b6befbfec275e",  # TODO : à compléter
    },
    "hss-iad": {
        "url": "https://github.com/Cynthia2024/HSS-IAD/releases/download/v1.0/HSS-IAD.zip",
        "dest_archive": PATHS.processed_dir.parent / "raw" / "HSS-IAD.zip",
        "dest_dir": PATHS.hssiad_dir,
        "size_gb": 2.1,
        "checksum_sha256": None,  # à compléter
    },
}


def progress(blocks, blocksize, total):
    pct = min(100, blocks * blocksize * 100 // total) if total > 0 else 0
    sys.stdout.write(f"\r  {pct:3d}%")
    sys.stdout.flush()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  ✓ Archive déjà présente : {dest.name}")
        return
    print(f"  Téléchargement : {url}")
    urlretrieve(url, dest, reporthook=progress)
    print()


def extract(archive: Path, dest_dir: Path) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest_dir)
    elif archive.name.endswith((".tar.xz", ".tar.gz", ".tar")):
        with tarfile.open(archive) as t:
            t.extractall(dest_dir)
    else:
        raise ValueError(f"Format d'archive non supporté : {archive}")


def check_dataset(name: str, ds_config: dict) -> bool:
    """Vérifie qu'un dataset est présent (au moins un sous-dossier)."""
    dest = ds_config["dest_dir"]
    if not dest.exists():
        return False
    return any(dest.iterdir())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mvtec", action="store_true", help="Télécharge MVTec AD")
    parser.add_argument("--hss-iad", action="store_true", help="Télécharge HSS-IAD")
    parser.add_argument(
        "--all", action="store_true", help="Télécharge tout (MVTec + HSS-IAD)"
    )
    parser.add_argument(
        "--check", action="store_true", help="Vérifie l'intégrité, ne télécharge pas"
    )
    args = parser.parse_args()

    if args.check:
        for name, cfg in DATASETS.items():
            status = "✓ présent" if check_dataset(name, cfg) else "✗ absent"
            print(f"  {name:<10} → {cfg['dest_dir']}  [{status}]")
        return 0

    if args.all:
        args.mvtec = True
        setattr(args, "hss_iad", True)
    if not (args.mvtec or args.hss_iad):
        # Default : MVTec seul
        args.mvtec = True

    if args.mvtec:
        print(f"\n=== MVTec AD ({DATASETS['mvtec']['size_gb']} GB) ===")
        print(
            "⚠ MVTec AD requiert une inscription sur https://www.mvtec.com/company/research/datasets/mvtec-ad\n"
            "Si le lien direct est mort, télécharge manuellement et place l'archive dans data/raw/"
        )
        cfg = DATASETS["mvtec"]
        try:
            download(cfg["url"], cfg["dest_archive"])
            print(f"  Extraction vers {cfg['dest_dir']}...")
            extract(cfg["dest_archive"], cfg["dest_dir"])
            print("  ✓ MVTec AD prêt")
        except Exception as e:
            print(f"  ✗ Erreur : {e}")
            print(
                f"\n  Workaround manuel :\n"
                f"    1. Télécharge depuis https://www.mvtec.com/company/research/datasets/mvtec-ad\n"
                f"    2. Place l'archive dans : {cfg['dest_archive']}\n"
                f"    3. Re-run : uv run python scripts/download_data.py --mvtec"
            )

    if getattr(args, "hss_iad", False):
        print(f"\n=== HSS-IAD ({DATASETS['hss-iad']['size_gb']} GB) ===")
        cfg = DATASETS["hss-iad"]
        try:
            download(cfg["url"], cfg["dest_archive"])
            print(f"  Extraction vers {cfg['dest_dir']}...")
            extract(cfg["dest_archive"], cfg["dest_dir"])
            print("  ✓ HSS-IAD prêt")
        except Exception as e:
            print(f"  ✗ Erreur : {e}")
            print(
                "  Repo de référence : https://github.com/Cynthia2024/HSS-IAD"
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
