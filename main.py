import numpy as np
import matplotlib.pyplot as plt

# 🔹 1. 첫 번째 콘투어 (큰 원형)
theta1 = np.linspace(0, 2 * np.pi, 100)
r1 = 40
cx1, cy1 = 50, 50  # 중심을 1사분면에 위치시킴
contour1 = np.array([[cx1 + r1 * np.cos(t), cy1 + r1 * np.sin(t)] for t in theta1])

# 🔹 2. 두 번째 콘투어 (타원형)
theta2 = np.linspace(0, 2 * np.pi, 100)
a, b = 30, 15  # 장축, 단축
cx2, cy2 = 130, 80
contour2 = np.array([[cx2 + a * np.cos(t), cy2 + b * np.sin(t)] for t in theta2])

# 🔹 3. 렌더링
fig, ax = plt.subplots(figsize=(7, 6))

# 콘투어 그리기
ax.plot(contour1[:, 0], contour1[:, 1], 'r-', label='Contour 1 (circle)')
ax.plot(contour2[:, 0], contour2[:, 1], 'b-', label='Contour 2 (ellipse)')

# 🔹 4. 좌표계 표시
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)
ax.text(2, 2, '(0,0)', fontsize=10, ha='left', va='bottom')

# 🔹 5. 축 및 비율 설정
ax.set_xlim(0, 180)
ax.set_ylim(0, 120)
ax.set_aspect('equal')
ax.grid(True)
ax.legend()
ax.set_title("Two Closed Contours in 1st Quadrant")

plt.show()

# 🔹 6. 픽셀 리스트(정수형 변환 예시)
contour1_pixels = np.round(contour1).astype(int).tolist()
contour2_pixels = np.round(contour2).astype(int).tolist()

print("Contour1 pixel list (first 5):", contour1_pixels[:5])
print("Contour2 pixel list (first 5):", contour2_pixels[:5])
