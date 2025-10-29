"""
Curve drawing utilities for QGraphicsView.
Handles interactive closed curve drawing with mouse events.
"""

import numpy as np
from typing import List, Optional, Tuple
from PyQt6.QtCore import Qt, QObject, QEvent, QPointF, QRectF
from PyQt6.QtGui import QPainter, QPen, QColor, QPainterPath
from PyQt6.QtWidgets import QGraphicsScene, QApplication, QGraphicsPathItem, QGraphicsView


class CurveDrawer:
    """
    Interactive closed curve drawer for QGraphicsView.
    
    Parameters
    ----------
    view : QGraphicsView
        Target view for drawing
    num_curves : int
        Number of curves to draw (default: 2)
    pen_color : str
        Color of the drawing pen (default: "red")
    pen_width : int
        Width of the drawing pen (default: 2)
    close_threshold : int
        Distance threshold for auto-closing curve (default: 8)
    min_points : int
        Minimum points before allowing close (default: 10)
    """
    
    def __init__(
        self,
        view: QGraphicsView,
        num_curves: int = 2,
        pen_color: str = "red",
        pen_width: int = 2,
        close_threshold: int = 8,
        min_points: int = 10
    ):
        self.view = view
        self.num_curves = num_curves
        self.pen = QPen(QColor(pen_color), pen_width)
        self.close_threshold = close_threshold
        self.min_points = min_points
        
        # State
        self.closed_curves: List[np.ndarray] = []
        self.drawing = False
        self.current_path_item: Optional[QGraphicsPathItem] = None
        self.points: List[QPointF] = []
        self.start_point: Optional[QPointF] = None
        self.esc_pressed = False
        
        # Scene setup
        self.scene = QGraphicsScene()
        self.view.setScene(self.scene)
        
        visible_rect = self.view.viewport().rect()
        self.scene.setSceneRect(QRectF(visible_rect))
        
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Event filter
        self.filter_obj = self._create_event_filter()
    
    def _create_event_filter(self) -> QObject:
        """Create event filter for mouse and keyboard events."""
        parent = self
        
        class Filter(QObject):
            def eventFilter(self, obj, event):
                if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
                    parent.esc_pressed = True
                    print("🚪 ESC pressed — waiting to exit")
                    return True
                
                if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                    if len(parent.closed_curves) >= parent.num_curves:
                        return True
                    parent._on_mouse_press(event)
                    
                elif event.type() == QEvent.Type.MouseMove and parent.drawing:
                    parent._on_mouse_move(event)
                    
                elif event.type() == QEvent.Type.MouseButtonRelease:
                    parent.drawing = False
                    
                return False
        
        return Filter()
    
    def _on_mouse_press(self, event):
        """Handle mouse press event."""
        self.drawing = True
        pos = event.position()
        self.start_point = QPointF(pos.x(), pos.y())
        self.points = [self.start_point]
        
        path = QPainterPath()
        path.moveTo(self.start_point)
        self.current_path_item = QGraphicsPathItem(path)
        self.current_path_item.setPen(self.pen)
        self.scene.addItem(self.current_path_item)
    
    def _on_mouse_move(self, event):
        """Handle mouse move event."""
        pos = event.position()
        self.points.append(QPointF(pos.x(), pos.y()))
        
        path = self.current_path_item.path()
        path.lineTo(pos)
        self.current_path_item.setPath(path)
        
        # Auto-close if near start point
        if len(self.points) > self.min_points:
            dist = (pos - self.start_point).manhattanLength()
            if dist < self.close_threshold:
                self._close_curve()
    
    def _close_curve(self):
        """Close current curve and add to list."""
        self.drawing = False
        if len(self.points) < 3:
            return
        
        # Close the path
        self.points.append(self.points[0])
        path = self.current_path_item.path()
        path.lineTo(self.points[0])
        self.current_path_item.setPath(path)
        
        # Convert to numpy array
        pts_np = np.array([[p.x(), p.y()] for p in self.points])
        self.closed_curves.append(pts_np)
        
        print(f"✅ Curve #{len(self.closed_curves)} completed ({len(pts_np)} pts)")
        
        # Check if done
        if len(self.closed_curves) >= self.num_curves:
            self._finalize()
    
    def _finalize(self):
        """Finalize drawing process."""
        print(f"=== {self.num_curves} closed curves completed ===")
        self.view.viewport().removeEventFilter(self.filter_obj)
    
    def start_drawing(self) -> List[np.ndarray]:
        """
        Start interactive drawing and block until complete.
        
        Returns
        -------
        List[np.ndarray]
            List of closed curves, each as (N, 2) array
        """
        self.view.viewport().installEventFilter(self.filter_obj)
        
        app = QApplication.instance()
        while len(self.closed_curves) < self.num_curves and not self.esc_pressed:
            app.processEvents()
        
        return self.closed_curves
    
    def get_curves(self) -> List[np.ndarray]:
        """
        Get currently completed curves.
        
        Returns
        -------
        List[np.ndarray]
            List of closed curves
        """
        return self.closed_curves
    
    def clear(self):
        """Clear all curves and reset state."""
        self.closed_curves = []
        self.scene.clear()
        self.drawing = False
        self.current_path_item = None
        self.points = []
        self.start_point = None
        self.esc_pressed = False
        print("🗑️ Curves cleared")