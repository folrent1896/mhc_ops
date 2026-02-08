"""
MHC Forward Pre - Forward Implementations

Golden reference and optimized implementations for the forward pass.
"""

from .golden import mhc_forward_pre
from .mhc_forward_pre_triton import (
    mhc_forward_pre_triton,
    mhc_forward_pre_triton_optimized,
)

# Optional: TileLang support
try:
    from .mhc_forward_pre_tilelang import (
        MHCForwardPreTileLang,
        mhc_forward_pre_tilelang,
        mhc_forward_pre_tvm,
    )
    _tilelang_available = True
except ImportError:
    _tilelang_available = False

__all__ = [
    "mhc_forward_pre",
    "mhc_forward_pre_triton",
    "mhc_forward_pre_triton_optimized",
]

if _tilelang_available:
    __all__.extend([
        "MHCForwardPreTileLang",
        "mhc_forward_pre_tilelang",
        "mhc_forward_pre_tvm",
    ])
