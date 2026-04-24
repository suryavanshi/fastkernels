"""Numerical tolerances for reference and custom-kernel comparisons."""

from __future__ import annotations

from typing import Any


def default_tolerance(dtype: Any) -> tuple[float, float]:
    name = str(dtype).lower()
    if "float64" in name:
        return 1e-8, 1e-8
    if "float32" in name:
        return 1e-4, 1e-4
    if "bfloat16" in name:
        return 3e-2, 3e-2
    if "float16" in name:
        return 2e-2, 2e-2
    return 1e-4, 1e-4
