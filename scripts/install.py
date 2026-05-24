"""Print the recommended uv environment setup commands."""

from __future__ import annotations


def main() -> None:
    print("Recommended environment setup:")
    print("  uv sync --extra cu128   # CUDA 12.8 workstation")
    print("  uv sync --extra cu124   # CUDA 12.4 workstation")
    print("  uv sync --extra cu121   # CUDA 12.1 workstation")
    print("  uv sync --extra cpu     # CPU-only environment")
    print()
    print("Then run commands with, for example:")
    print("  uv run --extra cu128 python -m pytest tests -q")


if __name__ == "__main__":
    main()
