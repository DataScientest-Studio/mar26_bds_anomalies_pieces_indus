"""Détecte CUDA et lance uv sync avec le bon extra."""
import subprocess
import sys

def detect_cuda():
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                      text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "cpu"
    # Driver → CUDA mapping (règle simplifiée, à ajuster)
    # Driver >= 570 → cu128, >= 545 → cu124, >= 525 → cu121, sinon cpu
    version = float(out.split('.')[0])
    if version >= 570: return "cu128"
    if version >= 545: return "cu124"
    if version >= 525: return "cu121"
    return "cpu"

extra = detect_cuda()
print(f"Detected {extra}")
subprocess.run(["uv", "sync", "--extra", extra], check=True)
