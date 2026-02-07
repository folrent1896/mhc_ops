"""
MHC Forward Pre Operator - Source Package

This package contains implementations of the MHC Forward Pre operator:
- Golden reference implementations (forward and backward)
- Triton GPU kernels (forward and backward)
- TileLang DSL implementations (forward and backward)
"""

__version__ = "0.1.1"

# Forward implementations
from .forward import (
    mhc_forward_pre,
    mhc_forward_pre_triton,
    mhc_forward_pre_triton_optimized,
    MHCForwardPreTileLang,
    mhc_forward_pre_tilelang,
    mhc_forward_pre_tvm,
)

# Backward implementations
from .backward import (
    mhc_backward_manual,
    mhc_backward_triton,
    MHCBackwardTileLang,
    mhc_backward_tilelang,
)

__all__ = [
    # Forward
    "mhc_forward_pre",
    "mhc_forward_pre_triton",
    "mhc_forward_pre_triton_optimized",
    "MHCForwardPreTileLang",
    "mhc_forward_pre_tilelang",
    "mhc_forward_pre_tvm",
    # Backward
    "mhc_backward_manual",
    "mhc_backward_triton",
    "MHCBackwardTileLang",
    "mhc_backward_tilelang",
]
