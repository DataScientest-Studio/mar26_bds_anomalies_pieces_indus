"""Checkpoint and run-metadata helpers for functional-surface training."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
import torch


def save_functional_surface_checkpoint(
    run_dir: Path,
    model,
    optimizer,
    args,
    history: list[dict[str, Any]],
    best_epoch: int | None,
    best_val_loss: float | None,
    completed_epochs: int,
    *,
    save_as_best: bool,
    class_names: dict[int, str],
) -> None:
    """Save last/best checkpoints plus history and params for a ROI run."""

    checkpoint = {
        "model_type": args.model_type,
        "num_classes": args.num_classes,
        "class_names": class_names,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "category": args.category,
        "input_size": args.input_size,
        "context_size": args.context_size,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "completed_epochs": completed_epochs,
        "run_created_at": args.run_created_at,
        "command_line": args.command_line,
        "semantic_mask_column": args.semantic_mask_column,
    }
    torch.save(checkpoint, run_dir / "checkpoint_last.pt")
    checkpoint_every = int(getattr(args, "checkpoint_every_epochs", 0))
    if checkpoint_every > 0 and completed_epochs % checkpoint_every == 0:
        torch.save(checkpoint, run_dir / f"checkpoint_epoch_{completed_epochs:03d}.pt")
    if save_as_best:
        torch.save(checkpoint, run_dir / "checkpoint_best.pt")
        shutil.copy2(run_dir / "checkpoint_best.pt", run_dir / "checkpoint.pt")
    elif not (run_dir / "checkpoint.pt").exists():
        shutil.copy2(run_dir / "checkpoint_last.pt", run_dir / "checkpoint.pt")
    pd.DataFrame(history).to_csv(run_dir / "loss_history.csv", index=False)
    params = {key: value for key, value in vars(args).items() if key not in {"command_line"}}
    params.update(
        {
            "checkpoint_path": str(run_dir / "checkpoint.pt"),
            "checkpoint_best_path": str(run_dir / "checkpoint_best.pt") if (run_dir / "checkpoint_best.pt").exists() else None,
            "checkpoint_last_path": str(run_dir / "checkpoint_last.pt"),
            "best_epoch": best_epoch,
            "best_val_loss": best_val_loss,
            "completed_epochs": completed_epochs,
            "class_names": class_names,
        }
    )
    (run_dir / "params.json").write_text(json.dumps(params, indent=2, default=str), encoding="utf-8")





