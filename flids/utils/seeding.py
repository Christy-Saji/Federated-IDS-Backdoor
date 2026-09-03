"""Task 1.5 - determinism.

This project has no torch dependency (the models are pure numpy), so the torch
/ CUDA determinism knobs from the plan's sketch do not apply. What must still be
seeded, explicitly and separately:
  * global ``random`` and ``numpy`` state
  * the Dirichlet partition                (seed passed to dirichlet_partition)
  * poison-sample selection                (seed derived per client per round)
  * model initialisation                   (seed passed to build_model)

A single unseeded ``np.random.choice`` in the poisoning function is enough to
make results irreproducible - keep every RNG call seeded from ``seed``.
"""

from __future__ import annotations

import os
import random

import numpy as np


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def env_info(seed: int) -> dict:
    """Everything that goes into results/<run_id>/env.json."""
    import platform
    import sys

    info = {
        "seed": seed,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
    }
    for mod in ("pandas", "sklearn", "scipy"):
        try:
            info[mod] = __import__(mod).__version__
        except Exception:
            info[mod] = None
    info["git_commit"] = _git_commit()
    return info


def _git_commit():
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
            cwd=os.path.dirname(__file__)).decode().strip()
    except Exception:
        return None
