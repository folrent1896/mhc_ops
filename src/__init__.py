"""
MHC Forward Pre Operator - Source Package

This package contains implementations of the MHC Forward Pre operator:
- Golden reference implementation
- Triton GPU kernels (forward and backward)
- TileLang DSL implementations (forward and backward)
"""

__version__ = "0.1.0"

# Golden reference implementation
from .golden import (
    mhc_forward_pre,
    mhc_pre_backward_manual,
)

# Triton implementations
from .mhc_forward_pre_triton import (
    mhc_forward_pre_triton,
    mhc_forward_pre_triton_optimized,
)

from .mhc_backward_triton import (
    mhc_backward_triton,
)

# TileLang implementations
from .mhc_forward_pre_tilelang import (
    MHCForwardPreTileLang,
    mhc_forward_pre_tilelang,
    mhc_forward_pre_tvm,
)

from .mhc_backward_tilelang import (
    MHCBackwardTileLang,
    mhc_backward_tilelang,
)

__all__ = [
    # Golden
    "mhc_forward_pre",
    "mhc_pre_backward_manual",
    # Triton forward
    "mhc_forward_pre_triton",
    "mhc_forward_pre_triton_optimized",
    # Triton backward
    "mhc_backward_triton",
    # TileLang forward
    "MHCForwardPreTileLang",
    "mhc_forward_pre_tilelang",
    "mhc_forward_pre_tvm",
    # TileLang backward
    "MHCBackwardTileLang",
    "mhc_backward_tilelang",
]
