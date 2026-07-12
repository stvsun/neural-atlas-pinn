"""Device and dtype resolution for the mapped-sphere PINN framework.

Usage::

    from mapped_sphere.device import resolve_device, resolve_dtype

    device = resolve_device(args.device)       # 'auto' | 'cuda' | 'mps' | 'cpu'
    dtype  = resolve_dtype(args.dtype, device) # 'auto' | 'float32' | 'float64'
"""

from __future__ import annotations

import torch


def resolve_device(device_arg: str) -> torch.device:
    """Resolve a ``--device`` CLI argument to a ``torch.device``.

    ``'auto'`` prefers CUDA, then MPS, then CPU. Other values are mapped
    directly; if the requested backend is unavailable, falls back to CPU
    with a printed warning. Raises ``ValueError`` on an unknown value.
    """
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            print("Requested --device cuda but CUDA is unavailable; falling back to CPU.")
            return torch.device("cpu")
        return torch.device("cuda")
    if device_arg == "mps":
        if getattr(torch.backends, "mps", None) is None or not torch.backends.mps.is_available():
            print("Requested --device mps but MPS is unavailable; falling back to CPU.")
            return torch.device("cpu")
        return torch.device("mps")
    if device_arg == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device option: {device_arg}")


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    """Resolve a ``--dtype`` CLI argument to a ``torch.dtype``.

    ``'auto'`` picks float32 on GPU backends (CUDA, MPS) and float64 on CPU.
    ``'float64'`` on MPS raises ``RuntimeError`` because MPS lacks adequate
    float64 support. Raises ``ValueError`` on an unknown value.
    """
    if dtype_arg == "auto":
        if device.type in ("cuda", "mps"):
            return torch.float32
        return torch.float64
    if dtype_arg == "float32":
        return torch.float32
    if dtype_arg == "float64":
        if device.type == "mps":
            raise RuntimeError("MPS backend does not support float64 well; use --dtype float32 or auto.")
        return torch.float64
    raise ValueError(f"Unsupported dtype option: {dtype_arg}")
