# generate.py
import numpy as np
import matplotlib.pyplot as plt
from coords import normalize_to_image

def draw_closed_curves(img_shape=(300, 800)):
    """
    마우스로 두 개의 폐곡선을 그리는 함수.
    수학좌표계(좌하단 원점, y 위로 증가) 기준으로 작동.
    """
    drawing = False
    points = []
    curves = []
    threshold = 5       # 시작점과의 거리(px)
    min_points = 10     # 거리 체크 시작 최소 포인트 수
    current_line = None
    start_point = None

    h, w = img_shape[:2]

    fig, ax = plt.subplots()
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title("좌클릭으로 곡선 그리기 → 시작점 근처(5px)이면 자동 폐곡선")

    # 드로잉 시작
    def on_press(event):
        nonlocal drawing, points, current_line, start_point
        if event.button != 1 or len(curves) >= 2:
            return
        drawing = True
        points = [(event.xdata, event.ydata)]
        start_point = np.array([event.xdata, event.ydata])
        current_line, = ax.plot(event.xdata, event.ydata, 'r-', lw=2)
        fig.canvas.draw()

    # 마우스 이동 중
    def on_motion(event):
        nonlocal points, drawing
        if not drawing or event.xdata is None or event.ydata is None:
            return
        points.append((event.xdata, event.ydata))
        xs, ys = zip(*points)
        current_line.set_data(xs, ys)
        fig.canvas.draw_idle()

        if len(points) > min_points:
            dist = np.hypot(event.xdata - start_point[0], event.ydata - start_point[1])
            if dist < threshold:
                close_curve()

    # 마우스 버튼 해제
    def on_release(event):
        nonlocal drawing
        drawing = False

    # 폐곡선 처리
    def close_curve():
        nonlocal points, curves, current_line, drawing
        drawing = False
        points.append(points[0])
        closed_points = interpolate_curve(points)
        closed_points = normalize_to_image(closed_points, img_shape)
        curves.append(np.array(closed_points))
        ax.plot(*zip(*closed_points), 'b-', lw=2)
        current_line = None
        fig.canvas.draw()

        print(f"✅ Curve #{len(curves)} completed ({len(closed_points)} points).")
        if len(curves) == 2:
            finalize()

    # 중간 픽셀 보간
    def interpolate_curve(points, step=1.0):
        pts = np.array(points)
        dist = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
        cumdist = np.concatenate([[0], np.cumsum(dist)])
        total_length = cumdist[-1]
        n_points = int(total_length / step)
        t_new = np.linspace(0, total_length, n_points)
        x_new = np.interp(t_new, cumdist, pts[:, 0])
        y_new = np.interp(t_new, cumdist, pts[:, 1])
        return list(zip(x_new, y_new))

    # 두 곡선이 완성되면 종료
    def finalize():
        print("\n=== 두 개의 폐곡선이 완성되었습니다 ===")
        plt.close(fig)

    # 이벤트 등록
    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)
    plt.show()

    if len(curves) != 2:
        raise RuntimeError("두 개의 곡선을 모두 그리지 않았습니다.")
    return curves[0], curves[1]
