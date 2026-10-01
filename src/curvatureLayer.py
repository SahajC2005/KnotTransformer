"""Backward compatibility wrapper for the original curvatureLayer module.

This module re-exports the new nD implementations from curvature_layer_nd.py
so legacy scripts such as knot_transformer.py can import the old interface.
"""

from curvature_layer_nd import (
    solve,
    global_curvature,
    arc,
    chordal_nd as chordal,
    centripetal_nd as centripetal,
    dup,
)

__all__ = [
    "solve",
    "global_curvature",
    "arc",
    "chordal",
    "centripetal",
    "dup",
]
