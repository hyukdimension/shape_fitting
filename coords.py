# coords.py
# 픽셀 좌표계 (row, col) ↔ 수학 좌표계 (x, y) 변환 모듈
import numpy as np

def pixel_to_math(pixels, img_height):
    """
    픽셀 좌표(row, col) → 수학 좌표(x, y) 변환

    Parameters
    ----------
    pixels : array-like of shape (N, 2)
        픽셀 좌표 (col, row) 순서로 들어올 것을 권장.
        (OpenCV 기준: (x=col, y=row))
    img_height : int
        영상 높이 (픽셀 단위)

    Returns
    -------
    ndarray of shape (N, 2)
        수학 좌표 (x, y)
    """
    pixels = np.asarray(pixels)
    if pixels.ndim == 1:
        pixels = pixels.reshape(1, -1)
    x_math = pixels[:, 0]
    y_math = img_height - 1 - pixels[:, 1]
    return np.column_stack((x_math, y_math))


def math_to_pixel(points, img_height):
    """
    수학 좌표(x, y) → 픽셀 좌표(row, col) 변환

    Parameters
    ----------
    points : array-like of shape (N, 2)
        수학 좌표 (x, y)
    img_height : int
        영상 높이 (픽셀 단위)

    Returns
    -------
    ndarray of shape (N, 2)
        픽셀 좌표 (row, col)
    """
    points = np.asarray(points)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    col = points[:, 0]
    row = img_height - 1 - points[:, 1]
    return np.column_stack((row, col))


def normalize_to_image(points, img_shape):
    """
    수학 좌표를 영상 크기 범위 내로 정규화 (클리핑 포함)
    """
    H, W = img_shape[:2]
    points = np.asarray(points)
    points[:, 0] = np.clip(points[:, 0], 0, W - 1)
    points[:, 1] = np.clip(points[:, 1], 0, H - 1)
    return points
