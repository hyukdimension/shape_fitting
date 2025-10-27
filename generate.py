import numpy as np
import cv2
from PyQt6.QtCore import Qt, QObject, QEvent, QPointF
from PyQt6.QtGui import QPainter, QPen, QColor, QPainterPath, QImage, QPixmap
from PyQt6.QtWidgets import QGraphicsScene
from PyQt6.QtWidgets import QGraphicsScene, QGraphicsPathItem

# ✅ 추가: QLabel에 두 시퀀스를 표시하는 함수
def show_two_curves_on_label(label, curve_std, curve_inp, size=(800, 600)):
    h, w = size[1], size[0]
    canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

    curve_std = np.clip(curve_std.astype(int), 0, [w - 1, h - 1])
    curve_inp = np.clip(curve_inp.astype(int), 0, [w - 1, h - 1])

    curve_std[:, 1] = size[1] - curve_std[:, 1]
    curve_inp[:, 1] = size[1] - curve_inp[:, 1]

    cv2.polylines(canvas, [curve_std.reshape(-1, 1, 2)], isClosed=True, color=(0, 0, 255), thickness=2)  # 빨강
    cv2.polylines(canvas, [curve_inp.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=2)  # 초록

    img_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    qimg = QImage(img_rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)

    label.setPixmap(QPixmap.fromImage(qimg))


# ✅ 수정된 draw_closed_curves_qt
def draw_closed_curves_qt(view1, view2=None):
    """
    Qt 환경에서 직접 두 개의 폐곡선을 그리기.
    view1: QGraphicsView (드로잉용)
    view2: QLabel (표시용, 선택사항)
    return: (curve_std, curve_inp) as np.ndarray
    """
    scene = QGraphicsScene()
    view1.setScene(scene)

    # Scene 크기 일치
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
            finalize()

    def finalize():
        print("=== 두 개의 폐곡선 완성 ===")
        view1.viewport().removeEventFilter(filter_obj)

        if view2 is not None:
            print("🎨 두 곡선을 view2에 표시 중...")

            # ✅ 스케일 정규화 후 view2 크기에 맞게 변환
            w1, h1 = view1.viewport().width(), view1.viewport().height()
            w2, h2 = view2.width(), view2.height()

            def normalize_to_view2(curve):
                # view1 비율 기반 정규화 → view2 크기 재스케일
                x = (curve[:, 0] / w1) * w2
                y = (curve[:, 1] / h1) * h2
                return np.column_stack([x, y])

            c1 = normalize_to_view2(closed_curves[0])
            c2 = normalize_to_view2(closed_curves[1])

            # ✅ QLabel 표시용 QPixmap 생성
            from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QPen
            from PyQt6.QtCore import Qt

            pixmap = QPixmap(w2, h2)
            pixmap.fill(Qt.GlobalColor.white)
            painter = QPainter(pixmap)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)

            # 곡선 1 (빨강)
            pen1 = QPen(QColor("red"), 2)
            painter.setPen(pen1)
            for i in range(len(c1) - 1):
                painter.drawLine(int(c1[i, 0]), int(c1[i, 1]),
                                 int(c1[i+1, 0]), int(c1[i+1, 1]))

            # 곡선 2 (파랑)
            pen2 = QPen(QColor("blue"), 2)
            painter.setPen(pen2)
            for i in range(len(c2) - 1):
                painter.drawLine(int(c2[i, 0]), int(c2[i, 1]),
                                 int(c2[i+1, 0]), int(c2[i+1, 1]))

            painter.end()

            # QLabel에 표시
            view2.setPixmap(pixmap)
            print("🎯 view2에 스케일 정규화된 두 곡선이 표시되었습니다.")

    class Filter(QObject):
        def eventFilter(self, obj, event):
            nonlocal drawing, current_path_item, points, start_point

            if event.type() == QEvent.Type.MouseButtonPress and event.button() == Qt.MouseButton.LeftButton:
                if len(closed_curves) >= 2:
                    return True
                drawing = True
                pos = event.position()
                start_point = QPointF(pos.x(), pos.y())
                points = [start_point]

                path = QPainterPath()
                path.moveTo(start_point)  # ✅ (0,0) 자동 포함 방지
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

    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    while len(closed_curves) < 2:
        app.processEvents()

    return closed_curves[0], closed_curves[1]
