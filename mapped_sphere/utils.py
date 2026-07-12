"""Shared utilities for the mapped-sphere PINN framework."""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and PyTorch (including CUDA) RNGs.

    If ``deterministic`` is True, additionally enforce deterministic
    algorithms for CUDA and cuDNN. This makes a per-machine baseline
    bit-stable across re-runs (at some cost to speed), but **does not**
    guarantee cross-machine bit-equivalence — driver/library versions,
    AVX vs SVE, and Metal shader compilation can still produce
    different bit patterns. Enable for baseline-capture sessions and
    audit runs; leave off for parameter sweeps and dev loops.

    The CUBLAS_WORKSPACE_CONFIG env var must be set BEFORE process
    start for ``use_deterministic_algorithms`` to succeed on CUDA;
    the function only sets it if it isn't already, in which case the
    user will need to relaunch the process for it to take effect.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        import os
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # warn_only=True keeps the op-not-deterministic warning instead
        # of raising — important because some ops (e.g. scatter_add_)
        # are not deterministic and we don't want training to die mid-run.
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
