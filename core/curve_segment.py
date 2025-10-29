"""
Curve segmentation and matching utilities.
Handles curve segmentation by distance and interactive segment matching.
"""

import numpy as np
from typing import List, Tuple, Optional
from PyQt6.QtCore import Qt, QPointF
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush, QPixmap, QPainterPath
from PyQt6.QtWidgets import QLabel
import matplotlib.cm as cm


class CurveSegmenter:
    """
    Segment curves into equal-distance parts.
    """
    
    @staticmethod
    def segment_by_distance(curve: np.ndarray, num_segments: int) -> List[np.ndarray]:
        """
        Segment curve into equal-distance parts.
        
        Parameters
        ----------
        curve : np.ndarray
            Curve coordinates (N, 2)
        num_segments : int
            Number of segments to create
            
        Returns
        -------
        List[np.ndarray]
            List of curve segments
        """
        dist = np.sqrt(np.sum(np.diff(curve, axis=0)**2, axis=1))
        total = np.sum(dist)
        step = total / num_segments
        
        segments = []
        acc = 0
        seg = [curve[0]]
        
        for i in range(1, len(curve)):
            d = dist[i-1]
            acc += d
            seg.append(curve[i])
            if acc >= step:
                segments.append(np.array(seg))
                seg = [curve[i]]
                acc = 0
        
        if len(seg) > 1:
            segments.append(np.array(seg))
        
        return segments[:num_segments]
    
    @staticmethod
    def segment_by_points(curve: np.ndarray, num_segments: int) -> List[np.ndarray]:
        """
        Segment curve into equal-point parts.
        
        Parameters
        ----------
        curve : np.ndarray
            Curve coordinates (N, 2)
        num_segments : int
            Number of segments to create
            
        Returns
        -------
        List[np.ndarray]
            List of curve segments
        """
        return list(np.array_split(curve, num_segments))
    
    @staticmethod
    def find_nearest_segment(segments: List[np.ndarray], x: float, y: float) -> int:
        """
        Find the nearest segment index to a point.
        
        Parameters
        ----------
        segments : List[np.ndarray]
            List of curve segments
        x, y : float
            Point coordinates
            
        Returns
        -------
        int
            Index of nearest segment
        """
        min_dist = float("inf")
        nearest_idx = -1
        
        for i, seg in enumerate(segments):
            center = np.mean(seg, axis=0)
            dist = np.hypot(center[0] - x, center[1] - y)
            if dist < min_dist:
                min_dist = dist
                nearest_idx = i
        
        return nearest_idx


class SegmentMatcher:
    """
    Interactive segment matching between two curves.
    Allows user to click segments and creates correspondence pairs.
    """
    
    def __init__(
        self,
        label: QLabel,
        curve_std: np.ndarray,
        curve_obj: np.ndarray,
        im_std_scaled: np.ndarray,
        im_obj_scaled: np.ndarray,
        num_segments: int = 10,
        segmentation_method: str = "points",
        im_std: Optional[np.ndarray] = None,
        im_obj: Optional[np.ndarray] = None,
        view3: Optional[QLabel] = None,
        view5: Optional[QLabel] = None
    ):
        """
        Parameters
        ----------
        label : QLabel
            Target label for display
        curve_std : np.ndarray
            Standard curve (N, 2)
        curve_obj : np.ndarray
            Object curve (N, 2)
        num_segments : int
            Number of segments per curve
        segmentation_method : str
            "distance" or "points"
        im_std : np.ndarray, optional
            Standard image (scaled) for view3
        im_obj : np.ndarray, optional
            Object image (scaled) for view5
        view3 : QLabel, optional
            View for standard image
        view5 : QLabel, optional
            View for object image
        """
        self.label = label
        self.num_segments = num_segments
        self.im_std = im_std
        self.im_obj = im_obj
        self.view3 = view3
        self.view5 = view5
        self.patch_size = 11  # Size of pixel patch to copy
        self.im_std_scaled = im_std_scaled
        self.im_obj_scaled = im_obj_scaled

        
        # Segment curves
        if segmentation_method == "distance":
            self.seg_std = CurveSegmenter.segment_by_distance(curve_std, num_segments)
            self.seg_obj = CurveSegmenter.segment_by_distance(curve_obj, num_segments)
        else:
            self.seg_std = CurveSegmenter.segment_by_points(curve_std, num_segments)
            self.seg_obj = CurveSegmenter.segment_by_points(curve_obj, num_segments)
        
        # State
        self.selected_pairs: List[Tuple[int, int]] = []
        self.click_buffer: List[Tuple[str, int]] = []
        
        # Colors
        self.cmap_std = cm.rainbow(np.linspace(0, 1, len(self.seg_std)))
        self.cmap_obj = cm.viridis(np.linspace(0, 1, len(self.seg_obj)))
    
    def draw_segments(self, painter: QPainter):
        """Draw all segments on painter."""
        # Draw standard segments (thick)
        for i, seg in enumerate(self.seg_std):
            path = QPainterPath()
            path.moveTo(seg[0][0], seg[0][1])
            for p in seg[1:]:
                path.lineTo(p[0], p[1])
            color = QColor(*(int(x * 255) for x in self.cmap_std[i][:3]))
            painter.setPen(QPen(color, 3))
            painter.drawPath(path)
        
        # Draw object segments (thin)
        for i, seg in enumerate(self.seg_obj):
            path = QPainterPath()
            path.moveTo(seg[0][0], seg[0][1])
            for p in seg[1:]:
                path.lineTo(p[0], p[1])
            color = QColor(*(int(x * 255) for x in self.cmap_obj[i][:3]))
            painter.setPen(QPen(color, 1.5))
            painter.drawPath(path)
    
    def calc_pointpairs_of_arrows(self, pts: list):
        """Draw arrows between matched segments."""
        
        for i, (std_idx, obj_idx) in enumerate(self.selected_pairs):
            c_std = np.mean(self.seg_std[std_idx], axis=0)
            c_obj = np.mean(self.seg_obj[obj_idx], axis=0)
            
            p1 = [float(c_std[0]), float(c_std[1])]
            p2 = [float(c_obj[0]), float(c_obj[1])]

            pts.append([p1, p2])

        return pts


    def draw_arrows(self, painter: QPainter, pts: np.ndarray):
        """Draw arrows between matched segments."""
        arrow_pen = QPen(Qt.GlobalColor.black, 2)
        painter.setPen(arrow_pen)
        
        for std_idx, obj_idx in self.selected_pairs:
            c_std = np.mean(self.seg_std[std_idx], axis=0)
            c_obj = np.mean(self.seg_obj[obj_idx], axis=0)
            
            p1 = QPointF(float(c_std[0]), float(c_std[1]))
            p2 = QPointF(float(c_obj[0]), float(c_obj[1]))
            painter.drawLine(p1, p2)
            
            # Arrow head
            dx, dy = c_obj - c_std
            angle = np.arctan2(dy, dx)
            arrow_size = 8
            
            start = QPointF(
                float(c_obj[0] - arrow_size * np.cos(angle - np.pi / 6)),
                float(c_obj[1] - arrow_size * np.sin(angle - np.pi / 6)),
            )
            end = QPointF(
                float(c_obj[0] - arrow_size * np.cos(angle + np.pi / 6)),
                float(c_obj[1] - arrow_size * np.sin(angle + np.pi / 6)),
            )
            painter.drawLine(p2, start)
            painter.drawLine(p2, end)


    def move_pixels(self, pts: list, patch_size=11, blend_ratio=1.0):
        """
        view5: Qlabel - 결과 이미지를 표시할 뷰
        im_std_resized: view3에 표시된 리사이즈된 std 이미지 (gray, HxW)
        im_obj_resized: view4/5에 표시된 리사이즈된 obj 이미지 (gray, HxW)
        pts: [(std_center, obj_center), ...]  — view 좌표 기준
        patch_size: odd number, default 11
        blend_ratio: float, default 0.2 (20% std + 80% obj)
        """
        import numpy as np

        from core.image_display import show_grayscale_on_label  # ✅ 이미 네 코드에 존재하는 함수 재활용

        im_std = self.im_std_scaled.copy()
        im_obj = self.im_obj_scaled.copy()
        im_disp = self.im_obj_scaled.copy()
        H, W = im_disp.shape
        r = patch_size // 2
        alpha = blend_ratio
        beta = 1.0 - alpha

        for std_pt, obj_pt in pts:
            x1, y1 = int(std_pt[0]), int(std_pt[1])
            x2, y2 = int(obj_pt[0]), int(obj_pt[1])

            # 범위 제한 (clamp)
            xs1, xe1 = max(0, x1 - r), min(W, x1 + r + 1)
            ys1, ye1 = max(0, y1 - r), min(H, y1 + r + 1)

            xs2, xe2 = max(0, x2 - r), min(W, x2 + r + 1)
            ys2, ye2 = max(0, y2 - r), min(H, y2 + r + 1)

            patch_std = im_std[ys1:ye1, xs1:xe1]
            patch_obj = im_obj[ys2:ye2, xs2:xe2]

            # 모서리 등에서 패치 크기 안 맞는 경우
            h = min(patch_std.shape[0], patch_obj.shape[0])
            w = min(patch_std.shape[1], patch_obj.shape[1])
            patch_std = patch_std[:h, :w]
            patch_obj = patch_obj[:h, :w]

            # 블렌딩
            blended = (alpha * patch_std + beta * patch_obj).astype(np.uint8)
            im_disp[ys2:ys2+h, xs2:xs2+w] = blended

        # 🔥 view5에 바로 표시
        # Get base size from view1
        base_w = self.view5.width()
        base_h = self.view5.height()
        base_size = (base_w, base_h)

        im_result = show_grayscale_on_label(self.view5, im_disp, base_size, add_corner_markers=True)

        print(f"✅ view5 갱신 완료 ({len(pts)}개의 픽셀 이동, patch={patch_size}x{patch_size})")

        # 리턴도 가능 (원하면 결과 배열을 추가 반환)
        return im_disp


    def redraw(self):
        """Redraw the entire view."""
        pixmap = QPixmap(self.label.width(), self.label.height())
        pixmap.fill(Qt.GlobalColor.white)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        self.draw_segments(painter)
        pts = []
        pts = self.calc_pointpairs_of_arrows(pts)
        self.draw_arrows(painter, pts)
        painter.end()
        self.label.setPixmap(pixmap)
        print(pts)
        _ = self.move_pixels(pts)
    
    def on_click(self, event):
        """Handle mouse click event."""
        if event.button() != Qt.MouseButton.LeftButton:
            return
        
        pos = event.position()
        x, y = pos.x(), pos.y()
        
        # Even click: select standard segment
        if len(self.click_buffer) % 2 == 0:
            idx = CurveSegmenter.find_nearest_segment(self.seg_std, x, y)
            self.click_buffer.append(("std", idx))
            print(f"🔴 Standard segment selected: {idx}")
        else:
            # Odd click: select object segment
            idx = CurveSegmenter.find_nearest_segment(self.seg_obj, x, y)
            self.click_buffer.append(("obj", idx))
            print(f"🔵 Object segment selected: {idx}")
            
            # Two clicks complete one pair
            if len(self.click_buffer) == 2:
                std_idx = self.click_buffer[0][1]
                obj_idx = self.click_buffer[1][1]
                self.selected_pairs.append((std_idx, obj_idx))
                print(f"✅ Pair registered: std={std_idx} → obj={obj_idx}")
                self.click_buffer = []
    
    def on_key(self, event):
        """Handle key press event."""
        if event.key() == Qt.Key.Key_Escape:
            print("🚪 ESC pressed — rendering final arrows...")
            self.redraw()
            # Remove event handlers
            self.label.mousePressEvent = lambda e: None
            self.label.keyPressEvent = lambda e: None
    
    def enable_selection(self):
        """Enable interactive segment selection."""
        # Set up event handlers
        self.label.mousePressEvent = self.on_click
        self.label.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.label.keyPressEvent = self.on_key
        
        # Initial draw
        self.redraw()
        
        print(f"🎯 Segment selection enabled ({len(self.seg_std)} segments)")
    
    def get_pairs(self) -> List[Tuple[int, int]]:
        """
        Get selected segment pairs.
        
        Returns
        -------
        List[Tuple[int, int]]
            List of (std_idx, obj_idx) pairs
        """
        return self.selected_pairs


def visualize_segments_with_colors(
    label: QLabel,
    curve_std: np.ndarray,
    curve_obj: np.ndarray,
    num_segments: int = 10
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Visualize segments with color coding.
    
    Parameters
    ----------
    label : QLabel
        Target label
    curve_std : np.ndarray
        Standard curve
    curve_obj : np.ndarray
        Object curve
    num_segments : int
        Number of segments
        
    Returns
    -------
    seg_std, seg_obj : List[np.ndarray]
        Segmented curves
    """
    w, h = label.width(), label.height()
    pixmap = QPixmap(w, h)
    pixmap.fill(Qt.GlobalColor.white)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    
    seg_std = CurveSegmenter.segment_by_distance(curve_std, num_segments)
    seg_obj = CurveSegmenter.segment_by_distance(curve_obj, num_segments)
    
    cmap_std = cm.rainbow(np.linspace(0, 1, len(seg_std)))
    cmap_obj = cm.viridis(np.linspace(0, 1, len(seg_obj)))
    
    # Draw standard curve (thick)
    for i, seg in enumerate(seg_std):
        color = QColor.fromRgbF(*cmap_std[i])
        painter.setPen(QPen(color, 10))
        for j in range(len(seg) - 1):
            painter.drawLine(
                int(seg[j, 0]), int(seg[j, 1]),
                int(seg[j+1, 0]), int(seg[j+1, 1])
            )
    
    # Draw object curve (thin)
    for i, seg in enumerate(seg_obj):
        color = QColor.fromRgbF(*cmap_obj[i])
        painter.setPen(QPen(color, 2))
        for j in range(len(seg) - 1):
            painter.drawLine(
                int(seg[j, 0]), int(seg[j, 1]),
                int(seg[j+1, 0]), int(seg[j+1, 1])
            )
    
    painter.end()
    label.setPixmap(pixmap)
    
    print(f"🎯 Segmented into {len(seg_std)} segments and displayed.")
    return seg_std, seg_obj