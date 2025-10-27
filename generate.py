import numpy as np
import cv2
from PyQt6.QtCore import Qt, QObject, QEvent, QPointF
from PyQt6.QtGui import QPainter, QPen, QColor, QPainterPath, QImage, QPixmap
from PyQt6.QtWidgets import QGraphicsScene
from PyQt6.QtWidgets import QGraphicsScene, QApplication, QGraphicsPathItem
import matplotlib.cm as cm

# ✅ 추가: QLabel에 두 시퀀스를 표시하는 함수
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


# ✅ 수정된 draw_closed_curves_qt
def draw_closed_curves_qt(view1, view2=None):
    """
    Qt 환경에서 직접 두 개의 폐곡선을 그리기.
    view1: QGraphicsView (드로잉용)
    view2: QLabel (표시용, 선택사항)
    return: (curve_std, curve_obj) as np.ndarray
    """
    from PyQt6.QtWidgets import QGraphicsScene, QApplication
    from PyQt6.QtGui import QPainter, QPen, QColor, QPainterPath, QPixmap, QBrush
    from PyQt6.QtCore import Qt, QObject, QEvent, QPointF
    import numpy as np
    import matplotlib.cm as cm

    scene = QGraphicsScene()
    view1.setScene(scene)
    scene.setSceneRect(0, 0, view1.viewport().width(), view1.viewport().height())
    view1.fitInView(scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
    view1.setRenderHint(QPainter.RenderHint.Antialiasing)

    pen = QPen(QColor("red"), 2)
    closed_curves = []
    drawing = False
    current_path_item = None
    points = []
    start_point = None
    threshold = 8
    min_points = 10

    # ✅ ESC 처리용
    esc_pressed = False

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
        if view2 is not None:
            show_two_curves_on_label(view2, closed_curves[0], closed_curves[1])
            print("🎨 두 폐곡선 시퀀스가 view2에 표시되었습니다.")
            enable_segment_selection(view2, closed_curves[0], closed_curves[1])

    class Filter(QObject):
        def eventFilter(self, obj, event):
            nonlocal drawing, current_path_item, points, start_point, esc_pressed

            if event.type() == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
                esc_pressed = True
                print("🚪 ESC pressed — 종료 대기 중")
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

    return closed_curves[0], closed_curves[1]



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