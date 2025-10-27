# image_io.py
import cv2
import numpy as np
from pathlib import Path

def safe_imread(path: str, flags=cv2.IMREAD_COLOR):
    """유니코드 경로 안전 이미지 로드"""
    p = Path(path)
    data = np.fromfile(str(p), dtype=np.uint8)
    if data.size == 0:
        raise FileNotFoundError(f"읽기 실패: {p}")
    img = cv2.imdecode(data, flags)
    if img is None:
        raise IOError(f"디코딩 실패: {p}")
    return img

def safe_imwrite(path: str, img, ext: str = ".png"):
    """유니코드 경로 안전 이미지 저장"""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    suf = p.suffix.lower() if p.suffix else ext
    if not suf.startswith("."):
        suf = "." + suf
    ok, buf = cv2.imencode(suf, img)
    if not ok:
        raise IOError(f"인코딩 실패: {path}")
    buf.tofile(str(p))
    return str(p)

def load_png_as_gray(path: str) -> np.ndarray:
    """PNG를 GRAY uint8 배열로 로드"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img
