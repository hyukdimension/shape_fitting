from generate import draw_closed_curves
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from pathlib import Path
from scipy.ndimage import map_coordinates

# -----------------------------
# 설정
# -----------------------------
segment_length = 20
sigma_src = 80
IMG_PATH = Path(__file__).resolve().parent / "background.jpg"

# -----------------------------
# 폐곡선 직접 입력받기
# -----------------------------
curve_std, curve_inp = draw_closed_curves()
print(f"✅ 입력된 곡선 길이: {len(curve_std)}, {len(curve_inp)}")

# 두 곡선을 나란히 배치
offset = np.max(curve_std[:, 0]) + 20
curve_inp_shifted = curve_inp.copy()
curve_inp_shifted[:, 0] += offset
std_seq = curve_std
inp_seq = curve_inp_shifted

# -----------------------------
# 세그먼트 분할
# -----------------------------
def make_segments(seq, seg_len):
    segs = []
    n = len(seq)
    for i in range(0, n - 1, seg_len):
        end = min(i + seg_len + 1, n)
        segs.append(seq[i:end])
    return segs

std_segments = make_segments(std_seq, segment_length)
inp_segments = make_segments(inp_seq, segment_length)

# -----------------------------
# 배경 이미지 준비
# -----------------------------
if IMG_PATH.exists():
    img = cv2.imread(str(IMG_PATH))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    print(f"✅ Loaded background image: {IMG_PATH}")
else:
    print("⚠️ background.jpg가 없어 기본 그라데이션 생성")
    h, w = 300, 800
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        img[i, :, 0] = np.linspace(0, 255, w).astype(np.uint8)
        img[i, :, 1] = 255 - np.linspace(0, 255, w).astype(np.uint8)
h, w, _ = img.shape

# -----------------------------
# 초기 렌더링
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(img, extent=[0, np.max(inp_seq[:, 0]) + 20, 0, np.max(inp_seq[:, 1]) + 20])
colors_std = cm.rainbow(np.linspace(0, 1, len(std_segments)))
colors_inp = cm.viridis(np.linspace(0, 1, len(inp_segments)))

for i, seg in enumerate(std_segments):
    ax.plot(seg[:, 0], seg[:, 1], color=colors_std[i], linewidth=2)
for i, seg in enumerate(inp_segments):
    ax.plot(seg[:, 0], seg[:, 1], color=colors_inp[i], linewidth=2)

ax.set_title("Click standard → input | ESC to apply asymmetric warp")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)
plt.tight_layout()

# -----------------------------
# 선택 로직
# -----------------------------
selected_pairs = []
click_count = 0
current_selection = []
interaction_active = True

def select_segment(seg_list, x_click, y_click):
    centers = [np.mean(seg, axis=0) for seg in seg_list]
    dists = [np.hypot(cx - x_click, cy - y_click) for cx, cy in centers]
    return int(np.argmin(dists))

def on_click(event):
    global click_count, current_selection
    if not interaction_active or event.inaxes != ax:
        return
    x_click, y_click = event.xdata, event.ydata
    curve_type = 'standard' if click_count % 2 == 0 else 'input'
    seg_list = std_segments if curve_type == 'standard' else inp_segments
    idx = select_segment(seg_list, x_click, y_click)
    seg = seg_list[idx]
    ax.plot(seg[:, 0], seg[:, 1], color='yellow', linewidth=4, zorder=5)
    fig.canvas.draw_idle()
    current_selection.append((curve_type, idx))
    click_count += 1
    if click_count % 2 == 0:
        selected_pairs.append(tuple(current_selection))
        print(f"✅ Pair {len(selected_pairs)} selected → {selected_pairs[-1]}")
        current_selection = []

def on_key(event):
    global interaction_active
    if event.key == 'escape':
        interaction_active = False
        fig.canvas.mpl_disconnect(cid_click)
        fig.canvas.mpl_disconnect(cid_key)
        plt.close(fig)
        print("\n🚪 ESC pressed — applying asymmetric warp...\n")
        apply_asymmetric_warp(selected_pairs)

# -----------------------------
# 비대칭 와핑 함수 (문지르기 효과 + 화살표 표시)
# -----------------------------
def apply_asymmetric_warp(pairs):
    global std_segments
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    dx = np.zeros_like(xx, dtype=np.float32)
    dy = np.zeros_like(yy, dtype=np.float32)
    arrows = []

    for pair in pairs:
        std_idx = pair[0][1]
        inp_idx = pair[1][1]
        c_std = np.mean(std_segments[std_idx], axis=0)
        c_inp = np.mean(inp_segments[inp_idx], axis=0)
        delta = c_inp - c_std
        print(f"Pair: std {std_idx} → inp {inp_idx}, Δ = {delta.round(2)}")

        # 문지르기 효과 (부드러운 영향)
        d2 = (xx - (c_std[0] / np.max(inp_seq[:, 0]) * w))**2 + \
             (yy - (h - c_std[1] / np.max(inp_seq[:, 1]) * h))**2
        influence = np.exp(-d2 / (2 * (sigma_src**2)))
        dx += delta[0] * (w / np.max(inp_seq[:, 0])) * influence * 0.3
        dy += -delta[1] * (h / np.max(inp_seq[:, 1])) * influence * 0.3

        # 이동된 세그먼트 좌표 갱신
        std_segments[std_idx] = std_segments[std_idx] + delta
        arrows.append((c_std, delta))

    # 와핑 적용
    xmap = np.clip(xx + dx, 0, w - 1)
    ymap = np.clip(yy + dy, 0, h - 1)
    warped = np.zeros_like(img)
    for c in range(3):
        warped[..., c] = map_coordinates(img[..., c], [ymap, xmap], order=1)

    # 전체 렌더링
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.imshow(warped, extent=[0, np.max(inp_seq[:, 0]) + 20, 0, np.max(inp_seq[:, 1]) + 20])
    colors_std = cm.rainbow(np.linspace(0, 1, len(std_segments)))
    colors_inp = cm.viridis(np.linspace(0, 1, len(inp_segments)))

    for i, seg in enumerate(std_segments):
        ax2.plot(seg[:, 0], seg[:, 1], color=colors_std[i], linewidth=2)
    for i, seg in enumerate(inp_segments):
        ax2.plot(seg[:, 0], seg[:, 1], color=colors_inp[i], linewidth=2)

    # 화살표
    for c_src, delta in arrows:
        ax2.arrow(c_src[0], c_src[1], delta[0], delta[1],
                  head_width=3, head_length=6,
                  fc='lime', ec='black', lw=0.8, alpha=0.9, zorder=5)
        ax2.scatter(c_src[0], c_src[1], s=25, color='lime', edgecolors='black', zorder=6)

    ax2.set_title("Warped Background + Segments + Move Arrows")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

# -----------------------------
# 이벤트 연결
# -----------------------------
cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
cid_key = fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()
