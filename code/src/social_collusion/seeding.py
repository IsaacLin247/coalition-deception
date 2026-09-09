"""Deterministic seeding for Python, NumPy and (if present) PyTorch.

PyTorch documents that bit-exact reproducibility is not guaranteed across releases or hardware
even with identical seeds [R22], so `runmeta.collect()` records versions/hardware for every run
and the determinism guarantee we actually *test* is on the pure NumPy engine, which is exact.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic_torch: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def episode_rng(base_seed: int, episode_index: int) -> np.random.Generator:
    """A fresh, independent generator per episode.

    Uses SeedSequence spawning rather than `base_seed + i` so that neighbouring episode indices
    are not correlated (this matters: evaluation scenario sets are built by index).
    """
    ss = np.random.SeedSequence([int(base_seed), int(episode_index)])
    return np.random.Generator(np.random.PCG64(ss))


def split_rng(rng: np.random.Generator, n: int) -> list[np.random.Generator]:
    """Deterministically derive `n` child generators (used for shadow rollouts / CRN)."""
    seeds = rng.integers(0, 2**63 - 1, size=n, dtype=np.int64)
    return [np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(s)))) for s in seeds]
