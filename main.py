import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# -----------------------------
# 1. 사인/코사인 곡선 생성 및 세그먼트 분할
# -----------------------------
x = np.linspace(0, 360, 200)
rad = np.deg2rad(x)
y_sin = 50 + 20 * np.sin(rad)
y_cos = 50 + 20 * np.cos(rad)

sin_seq = np.column_stack([x, y_sin])
cos_seq = np.column_stack([x, y_cos])

def make_segments(seq, seg_len):
    """시퀀스를 일정 픽셀 개수 단위로 나누어 세그먼트 리스트 반환"""
    segments = []
    n = len(seq)
    for i in range(0, n - 1, seg_len):
        end = min(i + seg_len + 1, n)
        segments.append(seq[i:end])
    return segments

segment_length = 20
sin_segments = make_segments(sin_seq, segment_length)
cos_segments = make_segments(cos_seq, segment_length)

# -----------------------------
# 2. 렌더링 초기 설정
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 5))
colors_sin = cm.rainbow(np.linspace(0, 1, len(sin_segments)))
colors_cos = cm.viridis(np.linspace(0, 1, len(cos_segments)))

for i, seg in enumerate(sin_segments):
    ax.plot(seg[:, 0], seg[:, 1], color=colors_sin[i], linewidth=2)
for i, seg in enumerate(cos_segments):
    ax.plot(seg[:, 0], seg[:, 1], color=colors_cos[i], linewidth=2)

ax.set_title("Interactive Segment Selection (ESC to finish)")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.grid(True)
plt.tight_layout()

# -----------------------------
# 3. 인터랙티브 로직
# -----------------------------
selected_pairs = []
click_count = 0
current_selection = []
interaction_active = True  # ESC로 False로 전환됨

def on_click(event):
    """마우스 클릭으로 세그먼트 선택"""
    global click_count, current_selection, interaction_active

    if not interaction_active or event.inaxes != ax:
        return

    # 클릭 좌표
    x_click, y_click = event.xdata, event.ydata
    curve_type = 'sin' if click_count % 2 == 0 else 'cos'
    seg_list = sin_segments if curve_type == 'sin' else cos_segments

    # 클릭 위치에 가장 가까운 세그먼트 인식
    centers = [np.mean(seg, axis=0) for seg in seg_list]
    distances = [np.hypot(cx - x_click, cy - y_click) for cx, cy in centers]
    idx = np.argmin(distances)

    # 하이라이트 표시
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
    """ESC 키 입력 감지 → 인터랙션 종료"""
    global interaction_active
    if event.key == 'escape':
        interaction_active = False
        fig.canvas.mpl_disconnect(cid_click)
        fig.canvas.mpl_disconnect(cid_key)
        plt.close(fig)
        print("\n🚪 ESC pressed — selection ended.\n")
        print("=== Final Selected Segment Pairs ===")
        if selected_pairs:
            for i, pair in enumerate(selected_pairs, 1):
                print(f"{i}: {pair}")
        else:
            print("No pairs selected.")

# -----------------------------
# 4. 이벤트 등록
# -----------------------------
cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
cid_key = fig.canvas.mpl_connect('key_press_event', on_key)

plt.show()
