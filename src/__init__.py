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
)

# Backward implementations
from .backward import (
    mhc_backward_manual,
    mhc_backward_triton,
)

# Optional: TileLang support
try:
    from .forward import (
        MHCForwardPreTileLang,
        mhc_forward_pre_tilelang,
        mhc_forward_pre_tvm,
    )
    from .backward import (
        MHCBackwardTileLang,
        mhc_backward_tilelang,
    )
    _tilelang_available = True
except ImportError:
    _tilelang_available = False

__all__ = [
    # Forward
    "mhc_forward_pre",
    "mhc_forward_pre_triton",
    "mhc_forward_pre_triton_optimized",
    # Backward
    "mhc_backward_manual",
    "mhc_backward_triton",
]

if _tilelang_available:
    __all__.extend([
        # Forward TileLang
        "MHCForwardPreTileLang",
        "mhc_forward_pre_tilelang",
        "mhc_forward_pre_tvm",
        # Backward TileLang
        "MHCBackwardTileLang",
        "mhc_backward_tilelang",
    ])
