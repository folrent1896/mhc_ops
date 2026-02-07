"""
MHC Forward Pre Operator - Source Package

This package contains implementations of the MHC Forward Pre operator:
- Golden reference implementation
- Triton GPU kernels
- TileLang DSL implementations
"""

__version__ = "0.1.0"

from .golden import (
    mhc_forward_pre,
    mhc_pre_backward_manual,
)

from .mhc_forward_pre_triton import (
    mhc_forward_pre_triton,
    mhc_forward_pre_triton_optimized,
)

__all__ = [
    "mhc_forward_pre",
    "mhc_pre_backward_manual",
    "mhc_forward_pre_triton",
    "mhc_forward_pre_triton_optimized",
]
