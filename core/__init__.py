"""
Core modules for curve drawing and image processing.

This package provides:
- curve_drawer: Interactive curve drawing in Qt
- curve_segment: Curve segmentation and matching
- image_display: Image display utilities for Qt labels
"""

__version__ = "1.0.0"

from .curve_drawer import CurveDrawer
from .curve_segment import CurveSegmenter, SegmentMatcher
from .image_display import (
    show_grayscale_on_label,
    show_two_curves_on_label,
    show_curves_overlay
)

__all__ = [
    'CurveDrawer',
    'CurveSegmenter',
    'SegmentMatcher',
    'show_grayscale_on_label',
    'show_two_curves_on_label',
    'show_curves_overlay',
]