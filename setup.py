"""Compatibility shim so ``pip install -e .`` also works on older pip/setuptools.

All package metadata lives in ``pyproject.toml``; this file only exists so that
legacy (pre-PEP-660) toolchains can still perform an editable install.
"""

from setuptools import setup

setup()
