"""
Image display utilities for Qt labels.
Handles grayscale images, curve overlays, and visualization.
"""

import numpy as np
import cv2
from typing import List, Tuple, Optional
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QBrush
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel


def show_grayscale_on_label(
    label: QLabel,
    image: np.ndarray,
    target_size: Tuple[int, int],
    add_corner_markers: bool = False
) -> Optional[np.ndarray]:
    """
    Display grayscale image on QLabel with optional corner markers.
    
    Parameters
    ----------
    label : QLabel
        Target label widget
    image : np.ndarray
        Grayscale image (H, W)
    target_size : tuple
        (width, height) to resize to
    add_corner_markers : bool
        Whether to add red corner markers
        
    Returns
    -------
    np.ndarray or None
        Resized image array, or None if failed
    """
    if image is None or label is None:
        print("⚠️ No image or label provided")
        return None
    
    target_w, target_h = target_size
    
    # Resize image to target size
    img_resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    h, w = img_resized.shape
    
    print(f"📏 Original: {image.shape[1]}x{image.shape[0]} → Resized: {w}x{h}")
    
    # Create QImage
    img_resized = np.ascontiguousarray(img_resized)
    qimg = QImage(img_resized.data, w, h, img_resized.strides[0], QImage.Format.Format_Grayscale8)
    
    # Get label size
    vw = label.width()
    vh = label.height()
    if vw == 0 or vh == 0:
        vw, vh = 320, 180
    
    # Scale to fit label
    pixmap = QPixmap.fromImage(qimg).scaled(
        vw, vh,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation
    )
    
    # Add corner markers if requested
    if add_corner_markers:
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        pen = QPen(Qt.GlobalColor.red, 2)
        brush = QBrush(Qt.GlobalColor.red)
        painter.setPen(pen)
        painter.setBrush(brush)
        
        radius = 5
        pw = pixmap.width()
        ph = pixmap.height()
        
        corners = [
            (radius, radius),
            (pw - radius, radius),
            (radius, ph - radius),
            (pw - radius, ph - radius)
        ]
        
        for x, y in corners:
            painter.drawEllipse(int(x - radius), int(y - radius), radius * 2, radius * 2)
        
        painter.end()
        print(f"🔴 Corner markers added (pixmap: {pw}x{ph})")
    
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setContentsMargins(0, 0, 0, 0)
    label.setStyleSheet("background-color: black; padding: 0px; margin: 0px;")
    label.setPixmap(pixmap)
    label.setScaledContents(False)
    
    print(f"🖼 {label.objectName() or 'label'} ← Resized: {w}x{h} → Pixmap: {pixmap.width()}x{pixmap.height()}")
    
    return img_resized


def show_two_curves_on_label(
    label: QLabel,
    curve_std: np.ndarray,
    curve_obj: np.ndarray,
    size: Tuple[int, int] = (800, 600)
):
    """
    Display two curves on a white canvas.
    
    Parameters
    ----------
    label : QLabel
        Target label
    curve_std : np.ndarray
        Standard curve coordinates (N, 2) in Qt coordinate system
    curve_obj : np.ndarray
        Object curve coordinates (N, 2) in Qt coordinate system
    size : tuple
        Canvas size (width, height)
    """
    h, w = size[1], size[0]
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

    curve_std = np.clip(curve_std.astype(int), 0, [w - 1, h - 1])
    curve_obj = np.clip(curve_obj.astype(int), 0, [w - 1, h - 1])

    curve_std[:, 1] = size[1] - curve_std[:, 1]
    curve_obj[:, 1] = size[1] - curve_obj[:, 1]

    cv2.polylines(canvas, [curve_std.reshape(-1, 1, 2)], isClosed=True, color=(0, 0, 255), thickness=2)  # Red
    cv2.polylines(canvas, [curve_obj.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=2)  # Green

    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    qimg = QImage(img_rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)

    label.setPixmap(QPixmap.fromImage(qimg))


def show_curves_overlay(
    label: QLabel,
    base_image: np.ndarray,
    curve_std: np.ndarray,
    curve_obj: np.ndarray,
    image_size: Tuple[int, int]
):
    """
    Overlay two curves on a base image.
    
    Parameters
    ----------
    label : QLabel
        Target label
    base_image : np.ndarray
        Base grayscale image (background)
    curve_std : np.ndarray
        Standard curve coordinates (N, 2) in Qt coordinate system
    curve_obj : np.ndarray
        Object curve coordinates (N, 2) in Qt coordinate system
    image_size : tuple
        (width, height)
    """
    h, w = image_size[1], image_size[0]
    
    # Convert grayscale to BGR
    if len(base_image.shape) == 2:
        canvas = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = base_image.copy()
    
    # Clip coordinates
    curve_std_pixel = np.clip(curve_std.astype(int), 0, [w - 1, h - 1])
    curve_obj_pixel = np.clip(curve_obj.astype(int), 0, [w - 1, h - 1])
    
    # Draw curves
    cv2.polylines(canvas, [curve_std_pixel.reshape(-1, 1, 2)], isClosed=True, color=(0, 0, 255), thickness=2)  # Red
    cv2.polylines(canvas, [curve_obj_pixel.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=2)  # Green
    
    # Convert to QPixmap
    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    qimg = QImage(img_rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    
    # Scale to label size
    vw = label.width()
    vh = label.height()
    if vw == 0 or vh == 0:
        vw, vh = 320, 180
    
    pixmap = QPixmap.fromImage(qimg).scaled(
        vw, vh,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation
    )
    
    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    label.setPixmap(pixmap)
    
    print(f"🎨 {label.objectName()} ← Curve overlay complete ({w}x{h})")