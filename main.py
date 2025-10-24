import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# -----------------------------
# 1. 두 곡선 생성 및 세그먼트 분할
# -----------------------------
x = np.linspace(0, 360, 200)
rad = np.deg2rad(x)
y_std = 50 + 20 * np.sin(rad)   # standard
y_inp = 50 + 20 * np.cos(rad)   # input

std_seq = np.column_stack([x, y_std])
inp_seq = np.column_stack([x, y_inp])

def make_segments(seq, seg_len):
    """시퀀스를 일정 픽셀 개수 단위로 나누어 세그먼트 리스트 반환"""
    segments = []
    n = len(seq)
    for i in range(0, n - 1, seg_len):
        end = min(i + seg_len + 1, n)
        segments.append(seq[i:end])
    return segments

segment_length = 20
std_segments = make_segments(std_seq, segment_length)
inp_segments = make_segments(inp_seq, segment_length)

# -----------------------------
# 2. 초기 렌더링
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 5))
colors_std = cm.rainbow(np.linspace(0, 1, len(std_segments)))
colors_inp = cm.viridis(np.linspace(0, 1, len(inp_segments)))

for i, seg in enumerate(std_segments):
    ax.plot(seg[:, 0], seg[:, 1], color=colors_std[i], linewidth=2)
for i, seg in enumerate(inp_segments):
    ax.plot(seg[:, 0], seg[:, 1], color=colors_inp[i], linewidth=2)

ax.set_title("Standard (sin) vs Input (cos) — Click to Select Pairs, ESC to Apply Shift")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)
plt.tight_layout()

# -----------------------------
# 3. 인터랙션 로직
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
# 4. 변형 함수 정의
# -----------------------------
def select_segment(seg_list, x_click, y_click):
    """무게중심 기반 + x범위 우선 세그먼트 선택"""
    centers = [np.mean(seg, axis=0) for seg in seg_list]

    # 후보: 클릭한 x좌표가 세그먼트의 x범위 안에 있는 것
    candidates = []
    for i, seg in enumerate(seg_list):
        x_min, x_max = np.min(seg[:, 0]), np.max(seg[:, 0])
        if x_min <= x_click <= x_max:
            candidates.append(i)

    if not candidates:
        # 모든 세그먼트의 중심 중에서 가장 가까운 것
        distances = [np.hypot(cx - x_click, cy - y_click) for cx, cy in centers]
        return int(np.argmin(distances))

    # 후보 중에서 무게중심과의 거리로 가장 가까운 것 선택
    distances = [np.hypot(centers[i][0] - x_click, centers[i][1] - y_click) for i in candidates]
    idx = candidates[int(np.argmin(distances))]
    return idx


def apply_transform(pairs):
    """선택된 세그먼트 쌍들에 대해 standard 세그먼트를 input 세그먼트 쪽으로 이동"""
    moved_segments = [seg.copy() for seg in std_segments]

    for pair in pairs:
        std_idx = pair[0][1]
        inp_idx = pair[1][1]

        # 두 세그먼트의 중심 계산
        c_std = np.mean(std_segments[std_idx], axis=0)
        c_inp = np.mean(inp_segments[inp_idx], axis=0)

        # 차이 벡터
        delta = c_inp - c_std

        # 이동 적용
        moved_segments[std_idx] = std_segments[std_idx] + delta
        print(f"Moved standard seg {std_idx} → input seg {inp_idx}, Δ = {delta.round(2)}")

    # 새 렌더링
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    # 이동된 standard
    for i, seg in enumerate(moved_segments):
        ax2.plot(seg[:, 0], seg[:, 1], color='orange', linewidth=2)
    # input 기준
    for i, seg in enumerate(inp_segments):
        ax2.plot(seg[:, 0], seg[:, 1], color='blue', linewidth=1, alpha=0.5)
    ax2.set_title("Transformed Standard (orange) aligned toward Input (blue)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.grid(True)
    plt.tight_layout()
    plt.show()

# -----------------------------
# 5. 이벤트 연결
# -----------------------------
cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
cid_key = fig.canvas.mpl_connect('key_press_event', on_key)

plt.show()
