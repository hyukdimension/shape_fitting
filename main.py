import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from scipy.ndimage import map_coordinates

# -----------------------------
# 1. 기본 설정
# -----------------------------
IMG_PATH = "background.jpg"  # 배경 이미지 경로
segment_length = 20          # 세그먼트 길이
sigma = 60                   # 와핑 반경 (커질수록 부드럽게)
strength = 1.0               # 이동 강도 (0~1)

# -----------------------------
# 2. 곡선 생성 + 세그먼트 분할
# -----------------------------
x = np.linspace(0, 360, 200)
rad = np.deg2rad(x)
y_std = 50 + 20 * np.sin(rad)   # standard
y_inp = 50 + 20 * np.cos(rad)   # input

std_seq = np.column_stack([x, y_std])
inp_seq = np.column_stack([x, y_inp])

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
# 3. 배경 이미지 준비
# -----------------------------
img = cv2.imread(IMG_PATH)
if img is None:
    # 테스트용 그라데이션 이미지 생성
    h, w = 200, 400
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        img[i, :, 0] = np.linspace(0, 255, w).astype(np.uint8)
        img[i, :, 1] = 255 - np.linspace(0, 255, w).astype(np.uint8)
else:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
h, w, _ = img.shape

# -----------------------------
# 4. 렌더링 초기화
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(img, extent=[0, 360, 0, 100])  # 이미지 배경
colors_std = cm.rainbow(np.linspace(0, 1, len(std_segments)))
colors_inp = cm.viridis(np.linspace(0, 1, len(inp_segments)))

for i, seg in enumerate(std_segments):
    ax.plot(seg[:, 0], seg[:, 1], color=colors_std[i], linewidth=2)
for i, seg in enumerate(inp_segments):
    ax.plot(seg[:, 0], seg[:, 1], color=colors_inp[i], linewidth=2)

ax.set_title("Click: standard → input | ESC: apply warp")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)
plt.tight_layout()

# -----------------------------
# 5. 세그먼트 선택 알고리즘 (무게중심 + x범위)
# -----------------------------
def select_segment(seg_list, x_click, y_click):
    centers = [np.mean(seg, axis=0) for seg in seg_list]
    candidates = []
    for i, seg in enumerate(seg_list):
        x_min, x_max = np.min(seg[:, 0]), np.max(seg[:, 0])
        if x_min <= x_click <= x_max:
            candidates.append(i)
    if not candidates:
        distances = [np.hypot(cx - x_click, cy - y_click) for cx, cy in centers]
        return int(np.argmin(distances))
    distances = [np.hypot(centers[i][0] - x_click, centers[i][1] - y_click) for i in candidates]
    return candidates[int(np.argmin(distances))]

# -----------------------------
# 6. 인터랙티브 핸들러
# -----------------------------
selected_pairs = []
click_count = 0
current_selection = []
interaction_active = True

def on_click(event):
    global click_count, current_selection, interaction_active

    if not interaction_active or event.inaxes != ax:
        return

    x_click, y_click = event.xdata, event.ydata
    curve_type = 'standard' if click_count % 2 == 0 else 'input'
    seg_list = std_segments if curve_type == 'standard' else inp_segments

    idx = select_segment(seg_list, x_click, y_click)
    seg = seg_list[idx]
    ax.plot(seg[:, 0], seg[:, 1], color='yellow', linewidth=4, zorder=5)
    fig.canvas.draw()

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
        print("\n🚪 ESC pressed — applying transformation...\n")
        apply_transform(selected_pairs)

# -----------------------------
# 7. 변형 및 와핑
# -----------------------------
def apply_transform(pairs):
    moved_segments = [seg.copy() for seg in std_segments]
    dx = np.zeros((h, w), dtype=np.float32)
    dy = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')

    for pair in pairs:
        std_idx = pair[0][1]
        inp_idx = pair[1][1]
        c_std = np.mean(std_segments[std_idx], axis=0)
        c_inp = np.mean(inp_segments[inp_idx], axis=0)
        delta = (c_inp - c_std) * strength
        moved_segments[std_idx] = std_segments[std_idx] + delta

        # 와핑용 displacement field 누적
        cx, cy = c_std
        dist2 = (xx - (cx / 360 * w))**2 + (yy - (h - cy / 100 * h))**2
        weight = np.exp(-dist2 / (2 * sigma**2))
        dx += -delta[0] * (w / 360) * weight   # ← x방향 부호 반전
        dy +=  delta[1] * (h / 100) * weight   # ← y방향 부호 반전

        print(f"Moved standard seg {std_idx} → input seg {inp_idx}, Δ = {delta.round(2)}")

    # 와핑 수행
    xmap = np.clip(xx + dx, 0, w - 1)
    ymap = np.clip(yy + dy, 0, h - 1)
    warped = np.zeros_like(img)
    for c in range(3):
        warped[..., c] = map_coordinates(img[..., c], [ymap, xmap], order=1)

    # 결과 표시
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.imshow(warped, extent=[0, 360, 0, 100])
    for seg in moved_segments:
        ax2.plot(seg[:, 0], seg[:, 1], color='orange', linewidth=2)
    for seg in inp_segments:
        ax2.plot(seg[:, 0], seg[:, 1], color='blue', linewidth=1, alpha=0.5)
    ax2.set_title("Warped Image + Moved Segments")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

# -----------------------------
# 8. 이벤트 등록 및 실행
# -----------------------------
cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
cid_key = fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()
