import numpy as np
import cv2
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6 import QtWidgets, QtGui
from scipy.ndimage import map_coordinates
from coords import pixel_to_math, math_to_pixel
from generate import draw_closed_curves_qt

# ============================================================
# Qt 내에서 전체 main.py 로직이 돌아가는 통합 함수
# ============================================================

def show_generated_views(view1, view2, view3, view4, view5):
    """
    view1: QGraphicsView (드로잉용)
    view2: QLabel (표시용)
    """
    print("✅ show_generated_views 호출됨")
    curve_std, curve_inp, im_std, im_obj = draw_closed_curves_qt(view1, view2, view3, view4, view5)
    print("✅ 두 곡선 입력 완료")