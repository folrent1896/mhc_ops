"""
MHC Forward Pre - Backward Implementations

Golden reference and optimized implementations for the backward pass.
"""

from .golden import mhc_backward_manual
from .mhc_backward_triton import mhc_backward_triton

# Optional: TileLang support
try:
    from .mhc_backward_tilelang import (
        MHCBackwardTileLang,
        mhc_backward_tilelang,
    )
    _tilelang_available = True
except ImportError:
    _tilelang_available = False

__all__ = [
    "mhc_backward_manual",
    "mhc_backward_triton",
]

if _tilelang_available:
    __all__.extend([
        "MHCBackwardTileLang",
        "mhc_backward_tilelang",
    ])
