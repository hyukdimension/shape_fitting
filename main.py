# main.py (통합 버전)
import sys, re, cv2, numpy as np, time
from pathlib import Path
from typing import Optional, List, Tuple, Literal
from PyQt6 import QtWidgets, uic
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QKeySequence, QShortcut
from PyQt6.QtWidgets import QProgressBar

# 스틸 이미지 처리부 import
from image_io import load_png_as_gray

from show_generated_views import show_generated_views


def to_pixmap(bgr: np.ndarray) -> QPixmap:
    if bgr is None or bgr.size == 0:
        return QPixmap()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    qimg = QImage(rgb.data, w, h, w*ch, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)


# ============ Main 클래스 ============
class Main(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = uic.loadUi("main_window.ui", self)

        # ========== 윈도우 크기 및 위치 설정 ==========
        self.showMaximized()

        self.mode = None
        
        # ========== FFmpeg Rate 파라미터 (하드코딩) ==========
        self.SAMPLE_RATE = 5.0 # 30.0
        self.DISPLAY_RATE = 5.0 # 30.0
        self.BUDGET_RATE = 5.0 # 30.0
        
        # UI에 표시 (*.ui 파일의 QLabel이 있다고 가정)
        if hasattr(self, 'labelSampleRate'):
            self.labelSampleRate.setText(f"Sample Rate: {self.SAMPLE_RATE} Hz")
        if hasattr(self, 'labelDisplayRate'):
            self.labelDisplayRate.setText(f"Display Rate: {self.DISPLAY_RATE} Hz")
        if hasattr(self, 'labelBudgetRate'):
            self.labelBudgetRate.setText(f"Budget Rate: {self.BUDGET_RATE} Hz")
        
        # ========== 처리 파라미터 (공유) ==========
        self.processing_params = {
            'delc_range': (-3, 3), # Todo: 이 값의 폭이 c_range보다 커야하는거 아닌지? 확인하기
            'K': 5,
            'c_range': (-2, 2),
            'select_line': 'top', # 바이너리 밴드에서 center col을 구한다. 그 밑의 히스토그램 말고.
            'threshold': 50  # ← threshold 추가 # Todo : 슬라이드로 값 변경 테스트할때만 쓰였던거다. 로직에선 안 쓰이니, 응용만 하고 지운다.
        }
        # =============================================
        
        # 위젯
        self.btnOpen   = self.findChild(QtWidgets.QPushButton, "btnOpen")
        self.btnRun    = self.findChild(QtWidgets.QPushButton, "btnRun")
        self.btnStop    = self.findChild(QtWidgets.QPushButton, "btnStop")
        self.btnPause  = self.findChild(QtWidgets.QPushButton, "btnPause")
        self.btnExit   = self.findChild(QtWidgets.QPushButton, "btnExit")
        self.labelPath = self.findChild(QtWidgets.QLabel,    "labelPath")
        self.slider1   = self.findChild(QtWidgets.QSlider,   "slider1")
        #self.view1     = self.findChild(QtWidgets.QLabel,    "preview_1")
        self.view1     = self.findChild(QtWidgets.QGraphicsView, "preview_1")
        self.view2     = self.findChild(QtWidgets.QLabel,    "preview_2")
        self.view3     = self.findChild(QtWidgets.QLabel,    "preview_3")
        self.view4     = self.findChild(QtWidgets.QLabel,    "preview_4")
        self.view5     = self.findChild(QtWidgets.QLabel,    "preview_5")
        self.view6     = self.findChild(QtWidgets.QLabel,    "preview_6")
        self.view7     = self.findChild(QtWidgets.QLabel,    "preview_7")
        self.view8     = self.findChild(QtWidgets.QLabel,    "preview_8")
        self.view9     = self.findChild(QtWidgets.QLabel,    "preview_9")


        # ✅ 수정된 코드
        for v in (self.view1, self.view2, self.view3, self.view4, self.view5, self.view6, self.view7, self.view8, self.view9):
            # v.setAlignment(Qt.AlignmentFlag.AlignCenter)  # ← 주석 처리 또는 삭제
            v.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)  # ✅ 좌상단 정렬
            v.setMinimumSize(320, 180)
            v.setStyleSheet("QLabel { background:#20252b; color:#d0d4d9; }")
            
            fixed_w, fixed_h = 320, 180
            v.setFixedSize(fixed_w, fixed_h)


        # 비디오 모드 상태
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_idx: int = -1
        self.frame_n: Optional[np.ndarray] = None
        self.frame_np1: Optional[np.ndarray] = None
        
        # VideoThread (추가)
        self.video_thread: Optional[VideoThread] = None

        # 스틸 모드 상태
        self.input_dir: Optional[Path] = None
        self.out_dir: Optional[Path] = None
        self.still_seq: Optional[List[Tuple[Path, int]]] = None
        self.still_pair_idx: int = 0

        # 연결
        self.btnRun.clicked.connect(self.step_forward)
        self.btnExit.clicked.connect(self.close)

        QShortcut(QKeySequence("F5"), self, activated=self.step_forward)

        self.statusBar().showMessage(
            "파일 선택: 비디오 또는 스틸 이미지 1개. 실행: 비디오 전진/스틸 한 스텝 처리.",
            5000
        )
        
        # 슬라이더 연결 수정
        if self.slider1 is not None:
            # slider1 범위 설정 (0~255)
            self.slider1.setMinimum(0)
            self.slider1.setMaximum(255)
            self.slider1.setValue(50)  # 초기값
            self.slider1.valueChanged.connect(self.on_threshold_changed)

        # __init__에서
        # 아래쪽 views 반복문도 수정
        views = [
            (self.view2, "현재 프레임"),
            (self.view3, "중간 단계"),
            (self.view4, "중간 단계"),
            (self.view5, "중간 단계"),
            (self.view6, "중간 단계"),
            (self.view7, "중간 단계"),
            (self.view8, "중간 단계"),
            (self.view9, "처리 결과"),
        ]

        for view, text in views:
            view.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)  # ✅ 좌상단
            view.setStyleSheet("""
                QLabel {
                    background-color: #1e1e1e;
                    color: #808080;
                    border: 2px dashed #404040;
                    font-size: 16px;
                    font-weight: bold;
                }
            """)

        # 아이콘 파일 사용 (icons/placeholder.png)
        pixmap = QPixmap("icons/play.png")
        scaled_pixmap = pixmap.scaled(
            self.view1.size(),  # QLabel의 현재 크기
            Qt.AspectRatioMode.KeepAspectRatio  # 비율 유지
        )
        # Todo self.view1.setPixmap(scaled_pixmap)
        pixmap = QPixmap("icons/play.png")
        scaled_pixmap = pixmap.scaled(
            self.view2.size(),  # QLabel의 현재 크기
            Qt.AspectRatioMode.KeepAspectRatio,  # 비율 유지
            Qt.TransformationMode.SmoothTransformation  # 부드럽게 리사이즈
        )
        self.view2.setPixmap(scaled_pixmap)


    def on_progress_update(self, stage: str, image: np.ndarray):
        """중간 결과 업데이트"""
        height, width = image.shape[:2]

        # 채널 수 출력 (디버깅)
        if len(image.shape) == 2:
            print(f"[DEBUG] {stage}: Grayscale image ({height}x{width}) - converting to colormap")
            channels = 1

            # Grayscale → 컬러맵 (0=빨강, 최대값=초록)
            # 정규화
            img_normalized = image.astype(np.float32)
            img_min = img_normalized.min()
            img_max = img_normalized.max()

            if img_max > img_min:
                img_normalized = (img_normalized - img_min) / (img_max - img_min)
            else:
                img_normalized = np.zeros_like(img_normalized)

            # RGB 생성: 0 → (255, 0, 0) 빨강, 1 → (0, 255, 0) 초록
            red = ((1.0 - img_normalized) * 255).astype(np.uint8)
            green = (img_normalized * 255).astype(np.uint8)
            blue = np.zeros_like(red)

            # BGR로 합치기
            image = cv2.merge([blue, green, red])  # OpenCV는 BGR 순서

        else:
            channels = image.shape[2]
            print(f"[DEBUG] {stage}: Color image ({height}x{width}x{channels})")

        # 이미지는 이제 항상 컬러
        height, width = image.shape[:2]
        bytes_per_line = 3 * width
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        q_image = QImage(rgb_image.data, width, height, bytes_per_line,
                         QImage.Format.Format_RGB888)

        pixmap = QPixmap.fromImage(q_image)

        # 단계별로 적절한 라벨에 표시
        if stage == "direction":
            self.view4.setPixmap(pixmap.scaled(self.view4.size(), Qt.AspectRatioMode.KeepAspectRatio))
        elif stage == "histogram":
            dummy = 1


    def on_threshold_changed(self, value):
        """슬라이더 값 변경 → threshold 업데이트"""
        self.processing_params['threshold'] = value
        print(f"[PARAM] Threshold updated: {value}")
        
        # 상태바에 표시 (선택)
        self.statusBar().showMessage(f"Threshold: {value}", 1000)



    # ---------- 공용: 라벨 지우기 ----------
    def _clear_views(self):
        for v in (self.view1, self.view2, self.view3, self.view4, self.view5, self.view6, self.view7, self.view8, self.view9):
            v.setPixmap(QPixmap())
            v.setText("")

    # ---------- 스틸 시퀀스(5장) 수집/검증 ----------
    def _collect_still_sequence(self, dir_path: Path, expected_count: int = 5) -> Optional[List[Tuple[Path, int]]]:
        """기존 코드 그대로"""
        pat = re.compile(r"^frame_(\d{10})\.png$")
        cand: List[Tuple[Path, int]] = []
        for p in sorted(dir_path.glob("frame_*.png")):
            m = pat.match(p.name)
            if m:
                cand.append((p, int(m.group(1))))

        if len(cand) != expected_count:
            QtWidgets.QMessageBox.critical(
                self, "에러",
                f"스틸 이미지 개수 불일치: {len(cand)}개 발견 (필요: {expected_count}개)\n경로: {dir_path}"
            )
            QtWidgets.QApplication.instance().exit(1)
            return None

        cand.sort(key=lambda x: x[1])
        nums = [n for _, n in cand]
        for i in range(len(nums) - 1):
            if nums[i+1] != nums[i] + 1:
                QtWidgets.QMessageBox.critical(
                    self, "에러",
                    f"연속 번호 아님: {nums[i]} 다음이 {nums[i+1]} (연속 필요)\n경로: {dir_path}"
                )
                QtWidgets.QApplication.instance().exit(1)
                return None

        return cand

    def _detect_input_mode(self, folder: Path) -> Literal["stills", "video", "invalid"]:
        """기존 코드 그대로"""
        folder = Path(folder)
        if not folder.exists() or not folder.is_dir():
            return "invalid"

        png_count = mp4_count = other_count = dir_count = 0

        for p in folder.iterdir():
            if p.is_dir():
                dir_count += 1
                continue
            ext = p.suffix.lower()
            if ext == ".png":
                png_count += 1
            elif ext == ".mp4":
                mp4_count += 1
            else:
                other_count += 1

        if dir_count == 0 and other_count == 0 and mp4_count == 0 and png_count >= 1:
            return "stills"
        elif dir_count == 0 and other_count == 0 and png_count == 0 and mp4_count == 1:
            return "video"
        else:
            return "invalid"

    def still_step_once(self):
        """기존 코드 그대로"""
        if not getattr(self, "input_dir", None) or not getattr(self, "out_dir", None):
            QtWidgets.QMessageBox.warning(self, "안내", "먼저 스틸 이미지를 파일 선택으로 지정하세요.")
            return
        if not getattr(self, "still_seq", None) or len(self.still_seq) < 2:
            QtWidgets.QMessageBox.warning(self, "안내", "스틸 이미지가 2장 이상 필요합니다.")
            return

        total_pairs = len(self.still_seq) - 1
        if not hasattr(self, "still_pair_idx") or self.still_pair_idx is None:
            self.still_pair_idx = 0

        i = self.still_pair_idx
        if i >= total_pairs:
            QtWidgets.QMessageBox.information(self, "완료", "모든 인접 페어 처리가 끝났습니다.")
            return

        img1, n1 = self.still_seq[i]
        img2, n2 = self.still_seq[i + 1]

        im1 = load_png_as_gray(str(img1))
        im2 = load_png_as_gray(str(img2))
        if im1 is None or im2 is None:
            QtWidgets.QMessageBox.critical(self, "에러", f"이미지 로드 실패:\n{img1}\n{img2}")
            return

    # ========== 기존 메소드들 (그대로 유지) ==========
    def step_forward(self):
        """Run 버튼 눌렀을 때 동작"""

        show_generated_views(self.view1, self.view2, self.view3, self.view4, self.view5)


        QtWidgets.QMessageBox.warning(self, "오류", f"알 수 없는 모드: {self.mode}")

def main():
    app = QtWidgets.QApplication(sys.argv)
    # 🔹 스타일 지정
    app.setStyle("Fusion")  # 또는 "Windows", "macOS", "WindowsVista" 등
    w = Main()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()