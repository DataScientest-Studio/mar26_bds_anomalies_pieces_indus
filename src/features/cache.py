"""Cache d'embeddings sur disque au format .npz compressÃ©."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np


def save_embeddings(
    path: Path,
    embeddings: np.ndarray,
    image_paths: list[str],
    backbone: str,
    layer: str,
    input_size: int,
    **extra_metadata,
) -> None:
    """Sauvegarde les embeddings + index alignÃ© + mÃ©tadonnÃ©es.

    L'alignement entre `embeddings[i]` et `image_paths[i]` est garanti par la sauvegarde
    conjointe. Relire via `load_embeddings()`.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    metadata = {
        "backbone": backbone,
        "layer": layer,
        "input_size": input_size,
        "shape": list(embeddings.shape),
        "created_at": datetime.utcnow().isoformat(),
        **extra_metadata,
    }

    np.savez_compressed(
        path,
        embeddings=embeddings,
        image_paths=np.array(image_paths, dtype=object),
        metadata=np.array([str(metadata)], dtype=object),
    )


def load_embeddings(path: Path) -> tuple[np.ndarray, list[str], dict]:
    """Charge des embeddings sauvegardÃ©s via `save_embeddings()`.

    Returns
    -------
    (embeddings, image_paths, metadata)
    """
    data = np.load(path, allow_pickle=True)
    embeddings = data["embeddings"]
    image_paths = data["image_paths"].tolist()
    meta_str = str(data["metadata"][0])
    # Le dict est sÃ©rialisÃ© en str ; on le reconstruit via eval sÃ»r (dict literal)
    metadata = eval(meta_str, {"__builtins__": {}}, {})
    return embeddings, image_paths, metadata





