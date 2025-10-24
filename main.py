import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# -----------------------------
# 기본 곡선 생성
# -----------------------------
x = np.linspace(0, 360, 200)      # degree
rad = np.deg2rad(x)
y_sin = 50 + 20 * np.sin(rad)
y_cos = 50 + 20 * np.cos(rad)

sin_seq = np.column_stack([x, y_sin])
cos_seq = np.column_stack([x, y_cos])

# -----------------------------
# 세그먼트 생성 함수
# -----------------------------
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
# 렌더링 (모두 실선)
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 5))
colors_sin = cm.rainbow(np.linspace(0, 1, len(sin_segments)))
colors_cos = cm.viridis(np.linspace(0, 1, len(cos_segments)))

# 사인 세그먼트 렌더링
for i, seg in enumerate(sin_segments):
    ax.plot(seg[:, 0], seg[:, 1], color=colors_sin[i], linewidth=2)

# 코사인 세그먼트 렌더링 (실선)
for i, seg in enumerate(cos_segments):
    ax.plot(seg[:, 0], seg[:, 1], color=colors_cos[i], linewidth=2)

ax.set_title("Sine & Cosine Segmented (All Solid Lines)")
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.grid(True)
ax.legend(["sin segments", "cos segments"], loc="upper right")
plt.show()

# -----------------------------
# 픽셀 세그먼트 데이터
# -----------------------------
sin_segments_pixels = [np.round(seg).astype(int).tolist() for seg in sin_segments]
cos_segments_pixels = [np.round(seg).astype(int).tolist() for seg in cos_segments]

print(f"Total sine segments: {len(sin_segments_pixels)}")
print(f"Total cosine segments: {len(cos_segments_pixels)}")
print("\nFirst sine segment (first 5 points):")
print(sin_segments_pixels[0][:5])
