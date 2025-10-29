"""
Main integration module for curve drawing and visualization.
Combines curve drawing, image display, and segment matching.
"""

import numpy as np
import cv2
import os
from typing import Tuple, Optional

from core.image_display import (
    show_grayscale_on_label,
    show_two_curves_on_label,
    show_curves_overlay
)
from core.curve_drawer import CurveDrawer
from core.curve_segment import SegmentMatcher


def draw_closed_curves_qt(
    view1, view2=None, view3=None, view4=None, view5=None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Draw two closed curves in Qt environment and display results.
    
    Parameters
    ----------
    view1 : QGraphicsView
        View for interactive curve drawing
    view2 : QLabel, optional
        View for displaying both curves
    view3 : QLabel, optional
        View for standard image with curves
    view4 : QLabel, optional
        View for object image with curves
    view5 : QLabel, optional
        View for object image with curves (duplicate)
        
    Returns
    -------
    curve_std : np.ndarray or None
        Standard curve coordinates (N, 2)
    curve_obj : np.ndarray or None
        Object curve coordinates (N, 2)
    im_std_scaled : np.ndarray or None
        Scaled standard image
    im_obj_scaled : np.ndarray or None
        Scaled object image
    """
    # Get base size from view1
    base_w = view1.viewport().width()
    base_h = view1.viewport().height()
    base_size = (base_w, base_h)
    
    print(f"📏 Base size: {base_w}x{base_h}")

    # Initialize image containers
    im_std_scaled, im_obj_scaled = None, None

    # Load and display standard image
    if os.path.exists("std.png"):
        im_std = cv2.imread("std.png", cv2.IMREAD_GRAYSCALE)
        if im_std is not None and view3 is not None:
            im_std_scaled = show_grayscale_on_label(
                view3, im_std, base_size, add_corner_markers=True
            )
            print("✅ std.png → view3")

    # Load and display object image
    if os.path.exists("obj.png"):
        im_obj = cv2.imread("obj.png", cv2.IMREAD_GRAYSCALE)
        if im_obj is not None:
            if view4 is not None:
                im_obj_scaled = show_grayscale_on_label(
                    view4, im_obj, base_size, add_corner_markers=True
                )
                print("✅ obj.png → view4")
            if view5 is not None:
                show_grayscale_on_label(
                    view5, im_obj, base_size, add_corner_markers=True
                )
                print("✅ obj.png → view5")

    # Draw curves
    drawer = CurveDrawer(view1, num_curves=2)
    closed_curves = drawer.start_drawing()
    
    if len(closed_curves) < 2:
        print("⚠️ Not enough curves drawn")
        return None, None, im_std_scaled, im_obj_scaled

    curve_std, curve_obj = closed_curves[0], closed_curves[1]

    # Display curves on view2
    if view2 is not None:
        show_two_curves_on_label(view2, curve_std, curve_obj, size=base_size)
        print("🎨 Curves → view2")
    
    # Overlay curves on images
    if view3 is not None and im_std_scaled is not None:
        show_curves_overlay(view3, im_std_scaled, curve_std, curve_obj, base_size)
        print("🎨 Overlay → view3")
    
    if view4 is not None and im_obj_scaled is not None:
        show_curves_overlay(view4, im_obj_scaled, curve_std, curve_obj, base_size)
        print("🎨 Overlay → view4")
    
    if view5 is not None and im_obj_scaled is not None:
        show_curves_overlay(view5, im_obj_scaled, curve_std, curve_obj, base_size)
        print("🎨 Overlay → view5")
    
    # Enable segment matching
    if view2 is not None:
        matcher = SegmentMatcher(
            view2, curve_std, curve_obj,
            num_segments=10,
            segmentation_method="points"
        )
        matcher.enable_selection()
        print("🎯 Segment matching enabled")

    return curve_std, curve_obj, im_std_scaled, im_obj_scaled