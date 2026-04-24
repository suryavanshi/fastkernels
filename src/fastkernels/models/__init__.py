"""Model shape specifications used by fastkernels."""

from .qwen35 import Qwen35MoESpec, qwen35_35b_a3b_spec
from .qwen36 import Qwen36A3BSpec, qwen36_35b_a3b_spec, synthetic_qwen36_spec

__all__ = [
    "Qwen35MoESpec",
    "Qwen36A3BSpec",
    "qwen35_35b_a3b_spec",
    "qwen36_35b_a3b_spec",
    "synthetic_qwen36_spec",
]
