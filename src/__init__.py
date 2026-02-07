"""
MHC Forward Pre Operator - Source Package

This package contains optimized implementations of the MHC Forward Pre operator:
- Triton GPU kernels
- TileLang DSL implementations
"""

__version__ = "0.1.0"

from .mhc_forward_pre_triton import (
    mhc_forward_pre_triton,
    mhc_forward_pre_triton_optimized,
)

__all__ = [
    "mhc_forward_pre_triton",
    "mhc_forward_pre_triton_optimized",
]
