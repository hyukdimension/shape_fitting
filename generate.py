import numpy as np
import cv2
from PyQt6.QtCore import Qt, QObject, QEvent, QPointF
from PyQt6.QtGui import QPainter, QPen, QColor, QPainterPath, QImage, QPixmap
from PyQt6.QtWidgets import QGraphicsScene
from PyQt6.QtWidgets import QGraphicsScene, QApplication, QGraphicsPathItem
import matplotlib.cm as cm
from PyQt6.QtCore import Qt, QObject, QEvent, QPointF, QRectF   # ← 반드시 추가해야 함




def show_two_curves_on_label(label, curve_std, curve_obj, size=(800, 600)):
    h, w = size[1], size[0]
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

    curve_std = np.clip(curve_std.astype(int), 0, [w - 1, h - 1])
    curve_obj = np.clip(curve_obj.astype(int), 0, [w - 1, h - 1])

    curve_std[:, 1] = size[1] - curve_std[:, 1]
    curve_obj[:, 1] = size[1] - curve_obj[:, 1]

    cv2.polylines(canvas, [curve_std.reshape(-1, 1, 2)], isClosed=True, color=(0, 0, 255), thickness=2)  # 빨강
    cv2.polylines(canvas, [curve_obj.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=2)  # 초록

    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    qimg = QImage(img_rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)

    label.setPixmap(QPixmap.fromImage(qimg))



def show_curves_overlay(label, base_image, curve_std, curve_obj, image_size):
    """
    기본 이미지 위에 두 곡선을 오버레이하여 표시
    
    Parameters:
    -----------
    label : QLabel
        표시할 QLabel
    base_image : np.ndarray
        기본 grayscale 이미지 (배경)
    curve_std : np.ndarray
        표준 곡선 좌표 (Qt 좌표계: x, y)
    curve_obj : np.ndarray
        객체 곡선 좌표 (Qt 좌표계: x, y)
    image_size : tuple
        (width, height)
    """
    from PyQt6.QtGui import QImage, QPixmap
    from PyQt6.QtCore import Qt
    import cv2
    import numpy as np
    
    h, w = image_size[1], image_size[0]
    
    # grayscale을 BGR로 변환
    if len(base_image.shape) == 2:
        canvas = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = base_image.copy()
    
    # 🔥 Qt 좌표계 → 픽셀 좌표계 변환
    # Qt: (x, y) where y increases downward
    # Pixel: (row, col) where row increases downward
    # 변환: row = y, col = x
    
    curve_std_pixel = curve_std.copy().astype(int)
    curve_obj_pixel = curve_obj.copy().astype(int)
    
    # x, y → col, row로 재배치
    curve_std_pixel = np.clip(curve_std_pixel, 0, [w - 1, h - 1])
    curve_obj_pixel = np.clip(curve_obj_pixel, 0, [w - 1, h - 1])
    
    # OpenCV는 (x, y) 형식을 받으므로 그대로 사용
    # 하지만 Qt 좌표계는 이미 y가 위에서 아래로 증가하므로 변환 불필요
    
    # 곡선 그리기
    cv2.polylines(canvas, [curve_std_pixel.reshape(-1, 1, 2)], isClosed=True, color=(0, 0, 255), thickness=2)  # 빨강
    cv2.polylines(canvas, [curve_obj_pixel.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=2)  # 초록
    
    # QPixmap으로 변환
    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    qimg = QImage(img_rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    
    # QLabel 크기에 맞춰 스케일링
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
    
    print(f"🎨 {label.objectName()} ← 곡선 오버레이 완료 ({w}x{h})")

    
def show_curves_overlay_OLD(label, base_image, curve_std, curve_obj, image_size):
    """
    기본 이미지 위에 두 곡선을 오버레이하여 표시
    
    Parameters:
    -----------
    label : QLabel
        표시할 QLabel
    base_image : np.ndarray
        기본 grayscale 이미지 (배경)
    curve_std : np.ndarray
        표준 곡선 좌표
    curve_obj : np.ndarray
        객체 곡선 좌표
    image_size : tuple
        (width, height)
    """
    from PyQt6.QtGui import QImage, QPixmap
    from PyQt6.QtCore import Qt
    import cv2
    import numpy as np
    
    h, w = image_size[1], image_size[0]
    
    # grayscale을 BGR로 변환
    if len(base_image.shape) == 2:
        canvas = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = base_image.copy()
    
    # 곡선 좌표 클리핑
    curve_std = np.clip(curve_std.astype(int), 0, [w - 1, h - 1])
    curve_obj = np.clip(curve_obj.astype(int), 0, [w - 1, h - 1])
    
    # Y좌표 반전 (Qt 좌표계 → 이미지 좌표계)
    curve_std_draw = curve_std.copy()
    curve_obj_draw = curve_obj.copy()
    curve_std_draw[:, 1] = h - 1 - curve_std[:, 1]
    curve_obj_draw[:, 1] = h - 1 - curve_obj[:, 1]
    
    # 곡선 그리기
    cv2.polylines(canvas, [curve_std_draw.reshape(-1, 1, 2)], isClosed=True, color=(0, 0, 255), thickness=2)  # 빨강
    cv2.polylines(canvas, [curve_obj_draw.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=2)  # 초록
    
    # QPixmap으로 변환
    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    qimg = QImage(img_rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    
    # QLabel 크기에 맞춰 스케일링
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
    
    print(f"🎨 {label.objectName()} ← 곡선 오버레이 완료 ({w}x{h})")


# ✅ 추가: QLabel에 두 시퀀스를 표시하는 함수
def show_two_curves_on_label_OLD(label, curve_std, curve_obj, size=(800, 600)):
    h, w = size[1], size[0]
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

    curve_std = np.clip(curve_std.astype(int), 0, [w - 1, h - 1])
    curve_obj = np.clip(curve_obj.astype(int), 0, [w - 1, h - 1])

    curve_std[:, 1] = size[1] - curve_std[:, 1]
    curve_obj[:, 1] = size[1] - curve_obj[:, 1]

    cv2.polylines(canvas, [curve_std.reshape(-1, 1, 2)], isClosed=True, color=(0, 0, 255), thickness=2)  # 빨강
    cv2.polylines(canvas, [curve_obj.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=2)  # 초록

    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    qimg = QImage(img_rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)

    label.setPixmap(QPixmap.fromImage(qimg))


# ✅ 수정된 draw_closed_curves_qt
def draw_closed_curves_qt(view1, view2=None, view3=None, view4=None, view5=None):
    """
    Qt 환경에서 직접 두 개의 폐곡선 그리기.
    view1: QGraphicsView (드로잉용)
    view2~view5: QLabel (표시용)
    return: (curve_std, curve_obj, im_std_scaled, im_obj_scaled)
    """
    from PyQt6.QtWidgets import QGraphicsScene, QApplication, QGraphicsPathItem
    from PyQt6.QtGui import QPainter, QPen, QColor, QPainterPath, QPixmap, QImage
    from PyQt6.QtCore import Qt, QObject, QEvent, QPointF, QRectF
    import numpy as np
    import cv2
    import os

    # === 🔥 모든 view를 동일한 기준 크기로 통일 ===
    base_w = view1.viewport().width()
    base_h = view1.viewport().height()
    
    print(f"🔍 기준 크기: {base_w}x{base_h}")

    scene = QGraphicsScene()
    view1.setScene(scene)

    visible_rect = view1.viewport().rect()
    scene.setSceneRect(QRectF(visible_rect))

    view1.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    view1.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    view1.setRenderHint(QPainter.RenderHint.Antialiasing)

    pen = QPen(QColor("red"), 2)
    closed_curves = []
    drawing = False
    current_path_item = None
    points = []
    start_point = None
    threshold = 8
    min_points = 10
    esc_pressed = False

    # === 이미지 로드 및 스케일링 ===
    im_std_scaled, im_obj_scaled = None, None

    def show_gray_on_label(view, img):
        """
        QLabel(viewX)에 그레이 이미지를 표시.
        - 기준 크기(base_w x base_h)에 맞춰 스케일링
        - 스케일링된 이미지를 반환
        """
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QBrush
        import numpy as np
        
        if img is None or view is None:
            print("⚠️ 표시할 이미지 또는 view가 없습니다.")
            return None
        
        # 🔥 원본 이미지를 base_w x base_h 크기로 리사이즈
        img_resized = cv2.resize(img, (base_w, base_h), interpolation=cv2.INTER_LINEAR)
        h, w = img_resized.shape
        
        print(f"📐 원본: {img.shape[1]}x{img.shape[0]} → 리사이즈: {w}x{h}")
        
        # QImage 생성
        img_resized = np.ascontiguousarray(img_resized)
        qimg = QImage(img_resized.data, w, h, img_resized.strides[0], QImage.Format.Format_Grayscale8)
        
        # QLabel 크기
        vw = view.width()
        vh = view.height()
        if vw == 0 or vh == 0:
            vw, vh = 320, 180
        
        # QLabel 크기에 맞춰 비율 유지하며 스케일링 (표시용)
        pixmap = QPixmap.fromImage(qimg).scaled(
            vw, vh,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        
        # 🔥 view3, 4, 5에만 코너 동그라미 추가
        view_name = view.objectName() if view.objectName() else ""
        if view_name in ["preview_3", "preview_4", "preview_5"]:
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
            print(f"🔴 {view_name}에 코너 마커 추가 (pixmap: {pw}x{ph})")
        
        view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        view.setContentsMargins(0, 0, 0, 0)
        view.setStyleSheet("background-color: black; padding: 0px; margin: 0px;")
        
        view.setPixmap(pixmap)
        view.setScaledContents(False)
        
        print(f"🖼 {view.objectName() or 'view'} ← 리사이즈: {w}x{h} → pixmap: {pixmap.width()}x{pixmap.height()}")
        
        # 🔥 스케일링된 넘파이 배열 반환
        return img_resized

    # std.png → view3
    if os.path.exists("std.png"):
        im_std = cv2.imread("std.png", cv2.IMREAD_GRAYSCALE)
        if im_std is not None:
            im_std_scaled = show_gray_on_label(view3, im_std)
            print("✅ std.png → view3 표시 완료")

    # obj.png → view4, view5
    if os.path.exists("obj.png"):
        im_obj = cv2.imread("obj.png", cv2.IMREAD_GRAYSCALE)
        if im_obj is not None:
            im_obj_scaled = show_gray_on_label(view4, im_obj)
            show_gray_on_label(view5, im_obj)
            print("✅ obj.png → view4, view5 표시 완료")


    print(f"view3: {view3.width()}x{view3.height()}")
    print(f"view4: {view4.width()}x{view4.height()}")  
    print(f"view5: {view5.width()}x{view5.height()}")



    # === 곡선 그리기 ===
    # 곡선 그리기 로직...
    def close_curve():
        nonlocal drawing, current_path_item, points
        drawing = False
        if len(points) < 3:
            return
        points.append(points[0])
        path = current_path_item.path()
        path.lineTo(points[0])
        current_path_item.setPath(path)
        pts_np = np.array([[p.x(), p.y()] for p in points])
        closed_curves.append(pts_np)
        print(f"✅ Curve #{len(closed_curves)} completed ({len(pts_np)} pts)")
        if len(closed_curves) == 2:
            finalize_curves()



    def finalize_curves():
        print("=== 두 개의 폐곡선 완성 ===")
        view1.viewport().removeEventFilter(filter_obj)
        
        # view2에 표시
        if view2 is not None:
            show_two_curves_on_label(view2, closed_curves[0], closed_curves[1], size=(base_w, base_h))
            print("🎨 두 폐곡선이 view2에 표시됨")
        
        # 🔥 view3, 4, 5에도 곡선 오버레이 표시
        if view3 is not None and im_std_scaled is not None:
            show_curves_overlay(view3, im_std_scaled, closed_curves[0], closed_curves[1], (base_w, base_h))
            print("🎨 곡선이 view3에 오버레이됨")
        
        if view4 is not None and im_obj_scaled is not None:
            show_curves_overlay(view4, im_obj_scaled, closed_curves[0], closed_curves[1], (base_w, base_h))
            print("🎨 곡선이 view4에 오버레이됨")
        
        if view5 is not None and im_obj_scaled is not None:
            show_curves_overlay(view5, im_obj_scaled, closed_curves[0], closed_curves[1], (base_w, base_h))
            print("🎨 곡선이 view5에 오버레이됨")
        
        enable_segment_selection(view2, closed_curves[0], closed_curves[1])



    def finalize_curves_OLD():
        print("=== 두 개의 폐곡선 완성 ===")
        view1.viewport().removeEventFilter(filter_obj)
        if view2 is not None:
            # 🔥 view2도 동일한 크기로 표시
            show_two_curves_on_label(view2, closed_curves[0], closed_curves[1], size=(base_w, base_h))
            print("🎨 두 폐곡선이 view2에 표시됨")
            enable_segment_selection(view2, closed_curves[0], closed_curves[1])

    class Filter(QObject):
        def eventFilter(self, obj, event):
            nonlocal drawing, current_path_item, points, start_point, esc_pressed
            if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
                esc_pressed = True
                print("🚪 ESC pressed – 종료 대기 중")
                return True
            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                if len(closed_curves) >= 2:
                    return True
                drawing = True
                pos = event.position()
                start_point = QPointF(pos.x(), pos.y())
                points = [start_point]
                path = QPainterPath()
                path.moveTo(start_point)
                current_path_item = QGraphicsPathItem(path)
                current_path_item.setPen(pen)
                scene.addItem(current_path_item)
            elif event.type() == QEvent.Type.MouseMove and drawing:
                pos = event.position()
                points.append(QPointF(pos.x(), pos.y()))
                path = current_path_item.path()
                path.lineTo(pos)
                current_path_item.setPath(path)
                if len(points) > min_points:
                    dist = (pos - start_point).manhattanLength()
                    if dist < threshold:
                        close_curve()
            elif event.type() == QEvent.Type.MouseButtonRelease:
                drawing = False
            return False

    filter_obj = Filter()
    view1.viewport().installEventFilter(filter_obj)

    app = QApplication.instance()
    while len(closed_curves) < 2 and not esc_pressed:
        app.processEvents()

    return closed_curves[0], closed_curves[1], im_std_scaled, im_obj_scaled



def segment_and_show_on_label(view2, curve_std, curve_obj, num_segments=10):
    """
    두 곡선을 일정한 픽셀 수 기준으로 분할 후 view2(QLabel)에 표시
    """
    from PyQt6.QtGui import QPixmap, QPainter, QColor, QPen
    from PyQt6.QtCore import Qt
    import matplotlib.cm as cm

    w2, h2 = view2.width(), view2.height()
    pixmap = QPixmap(w2, h2)
    pixmap.fill(Qt.GlobalColor.white)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    def segment_curve(curve, n_seg):
        """곡선을 거리 기반 균등 분할"""
        dist = np.sqrt(np.sum(np.diff(curve, axis=0)**2, axis=1))
        total = np.sum(dist)
        step = total / n_seg
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
        return segments[:n_seg]

    # 세그먼트화
    seg_std = segment_curve(curve_std, num_segments)
    seg_obj = segment_curve(curve_obj, num_segments)

    # 색상 팔레트
    cmap_std = cm.rainbow(np.linspace(0, 1, len(seg_std)))
    cmap_obj = cm.viridis(np.linspace(0, 1, len(seg_obj)))

    # 표준 곡선 (굵게)
    for i, seg in enumerate(seg_std):
        color = QColor.fromRgbF(*cmap_std[i])
        painter.setPen(QPen(color, 10))
        for j in range(len(seg)-1):
            painter.drawLine(int(seg[j, 0]), int(seg[j, 1]),
                             int(seg[j+1, 0]), int(seg[j+1, 1]))

    # 입력 곡선 (얇게)
    for i, seg in enumerate(seg_obj):
        color = QColor.fromRgbF(*cmap_obj[i])
        painter.setPen(QPen(color, 2))
        for j in range(len(seg)-1):
            painter.drawLine(int(seg[j, 0]), int(seg[j, 1]),
                             int(seg[j+1, 0]), int(seg[j+1, 1]))

    painter.end()
    view2.setPixmap(pixmap)

    print(f"🎯 {len(seg_std)}개 세그먼트로 분할 완료 및 표시됨.")
    return seg_std, seg_obj


# ==========================
# ✅ ESC 이후 실행 파트들
# ==========================


def find_nearest_segment(segments, x, y):
    filter_obj = None   # ✅ 먼저 선언!
    """클릭 좌표(x, y)에 가장 가까운 세그먼트 인덱스를 찾는다."""
    min_dist = float("inf")
    nearest_idx = -1
    for i, seg in enumerate(segments):
        c = np.mean(seg, axis=0)
        dist = np.hypot(c[0] - x, c[1] - y)
        if dist < min_dist:
            min_dist = dist
            nearest_idx = i
    return nearest_idx


def enable_segment_selection(view2, curve_std, curve_obj):
    filter_obj = None
    """
    두 곡선을 세그먼트로 나누고, 포인팅(2n번 클릭) + ESC 후 화살표 표시
    """
    import numpy as np
    from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QBrush
    from PyQt6.QtCore import Qt

    seg_std = np.array_split(curve_std, 10)
    seg_obj = np.array_split(curve_obj, 10)
    selected_pairs = []
    click_count = 0
    esc_done = False

    w2, h2 = view2.width(), view2.height()
    pixmap = QPixmap(w2, h2)
    pixmap.fill(Qt.GlobalColor.white)

    def redraw():
        pixmap = QPixmap(view2.width(), view2.height())
        pixmap.fill(Qt.GlobalColor.white)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # --- 표준 (굵게)
        cmap_std = cm.rainbow(np.linspace(0, 1, len(seg_std)))
        for i, seg in enumerate(seg_std):
            path = QPainterPath()
            path.moveTo(seg[0][0], seg[0][1])
            for p in seg[1:]:
                path.lineTo(p[0], p[1])
            c = QColor(*(int(x * 255) for x in cmap_std[i][:3]))
            painter.setPen(QPen(c, 3))
            painter.drawPath(path)

        # --- 오브젝트 (얇게)
        cmap_obj = cm.viridis(np.linspace(0, 1, len(seg_obj)))
        for i, seg in enumerate(seg_obj):
            path = QPainterPath()
            path.moveTo(seg[0][0], seg[0][1])
            for p in seg[1:]:
                path.lineTo(p[0], p[1])
            c = QColor(*(int(x * 255) for x in cmap_obj[i][:3]))
            painter.setPen(QPen(c, 1.5))
            painter.drawPath(path)

        # --- 화살표 표시
        arrow_pen = QPen(Qt.GlobalColor.black, 2)
        painter.setPen(arrow_pen)

        for std_idx, obj_idx in selected_pairs:
            c_std = np.mean(seg_std[std_idx], axis=0)
            c_obj = np.mean(seg_obj[obj_idx], axis=0)

            # ✅ QPointF로 변환
            p1 = QPointF(float(c_std[0]), float(c_std[1]))
            p2 = QPointF(float(c_obj[0]), float(c_obj[1]))
            painter.drawLine(p1, p2)

            # 간단한 화살표 머리
            dx, dy = c_obj - c_std
            angle = np.arctan2(dy, dx)
            arrow_size = 8
            left = QPointF(
                float(c_obj[0] - arrow_size * np.cos(angle - np.pi / 6)),
                float(c_obj[1] - arrow_size * np.sin(angle - np.pi / 6)),
            )
            right = QPointF(
                float(c_obj[0] - arrow_size * np.cos(angle + np.pi / 6)),
                float(c_obj[1] - arrow_size * np.sin(angle + np.pi / 6)),
            )
            painter.drawLine(p2, left)
            painter.drawLine(p2, right)

        painter.end()
        view2.setPixmap(pixmap)

    selected_pairs = []  # (std_idx, inp_idx)
    click_buffer = []    # 클릭 2개 모으는 버퍼

    def on_click(event):
        nonlocal click_buffer
        if event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position()
        x, y = pos.x(), pos.y()

        # 짝수 클릭: standard 선택
        if len(click_buffer) % 2 == 0:
            idx = find_nearest_segment(seg_std, x, y)
            click_buffer.append(("std", idx))
            print(f"🔴 Standard segment selected: {idx}")
        else:
            idx = find_nearest_segment(seg_obj, x, y)
            click_buffer.append(("inp", idx))
            print(f"🔵 Object segment selected: {idx}")

            # ✅ 두 번 클릭 시 한 쌍 완성
            if len(click_buffer) == 2:
                std_idx = click_buffer[0][1]
                inp_idx = click_buffer[1][1]
                selected_pairs.append((std_idx, inp_idx))
                print(f"✅ Pair registered: std={std_idx} → inp={inp_idx}")
                click_buffer = []


    def on_key(event):
        nonlocal filter_obj  # ✅ 이제 정상 동작!
        if event.key() == Qt.Key.Key_Escape:
            print("🚪 ESC pressed — rendering final arrows...")
            redraw()  # ✅ ESC 눌렀을 때만 화살표 렌더링
            view2.removeEventFilter(filter_obj)

    view2.mousePressEvent = on_click
    view2.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    view2.keyPressEvent = on_key
    redraw()



def finalize_segment_pairs(view2, seg_std, seg_obj, selected_pairs):
    """ESC 후 실행: 선택된 세그먼트 강조 + 화살표 표시"""
    from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QBrush
    from PyQt6.QtCore import Qt
    import numpy as np
    import matplotlib.cm as cm

    w2, h2 = view2.width(), view2.height()
    pixmap = QPixmap(w2, h2)
    pixmap.fill(Qt.GlobalColor.white)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    cmap_std = cm.rainbow(np.linspace(0, 1, len(seg_std)))
    cmap_obj = cm.viridis(np.linspace(0, 1, len(seg_obj)))
    centers_std = [np.mean(seg, axis=0) for seg in seg_std]
    centers_obj = [np.mean(seg, axis=0) for seg in seg_obj]

    # 기본 세그먼트 표시
    for i, seg in enumerate(seg_std):
        color = QColor.fromRgbF(*cmap_std[i])
        painter.setPen(QPen(color, 2))
        for j in range(len(seg)-1):
            painter.drawLine(int(seg[j,0]), int(seg[j,1]), int(seg[j+1,0]), int(seg[j+1,1]))

    for i, seg in enumerate(seg_obj):
        color = QColor.fromRgbF(*cmap_obj[i])
        painter.setPen(QPen(color, 1))
        for j in range(len(seg)-1):
            painter.drawLine(int(seg[j,0]), int(seg[j,1]), int(seg[j+1,0]), int(seg[j+1,1]))

    # 선택된 세그먼트 강조 + 화살표
    pen_sel = QPen(QColor("yellow"), 4)
    painter.setPen(pen_sel)
    painter.setBrush(QBrush(QColor("yellow")))

    for std_idx, inp_idx in selected_pairs:
        std_c = centers_std[std_idx % len(centers_std)]
        inp_c = centers_obj[inp_idx % len(centers_obj)]
        painter.drawEllipse(int(std_c[0])-5, int(std_c[1])-5, 10, 10)
        painter.drawEllipse(int(inp_c[0])-5, int(inp_c[1])-5, 10, 10)
        # 화살표
        painter.drawLine(int(std_c[0]), int(std_c[1]),
                         int(inp_c[0]), int(inp_c[1]))

    painter.end()
    view2.setPixmap(pixmap)
    print("🏁 ESC 이후: 세그먼트 강조 및 화살표 연결 완료")