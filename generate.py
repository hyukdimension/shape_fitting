import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

def draw_closed_curves():
    drawing = False
    points = []
    curves = []
    threshold = 5
    min_points = 10
    current_line = None
    start_point = None

    fig, ax = plt.subplots()
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 300)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_title("좌클릭으로 곡선 그리기 → 시작점 근처(5px)이면 자동 폐곡선")

    def on_press(event):
        nonlocal drawing, points, current_line, start_point
        if event.button != 1 or len(curves) >= 2:
            return
        drawing = True
        points = [(event.xdata, event.ydata)]
        start_point = np.array([event.xdata, event.ydata])
        current_line, = ax.plot(event.xdata, event.ydata, 'r-', lw=2)
        fig.canvas.draw()

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

    def on_release(event):
        nonlocal drawing
        drawing = False

    def close_curve():
        nonlocal points, curves, current_line, drawing
        drawing = False
        points.append(points[0])
        closed_points = interpolate_curve(points)
        curves.append(np.array(closed_points))
        ax.plot(*zip(*closed_points), 'b-', lw=2)
        current_line = None
        fig.canvas.draw()
        print(f"✅ Curve #{len(curves)} completed ({len(closed_points)} points).")
        if len(curves) == 2:
            finalize()

    def interpolate_curve(points, step=1.0):
        pts = np.array(points)
        dist = np.sqrt(np.sum(np.diff(pts, axis=0)**2, axis=1))
        cumdist = np.concatenate([[0], np.cumsum(dist)])
        total_length = cumdist[-1]
        n_points = max(2, int(total_length / step))
        t_new = np.linspace(0, total_length, n_points)
        x_new = np.interp(t_new, cumdist, pts[:, 0])
        y_new = np.interp(t_new, cumdist, pts[:, 1])
        return list(zip(x_new, y_new))

    def finalize():
        print("\n=== 두 개의 픽셀 시퀀스가 완성되었습니다 ===")
        for i, curve in enumerate(curves, 1):
            print(f"Curve {i}: {len(curve)} points")
        ax.set_title("✅ 두 폐곡선 모두 생성 완료")
        plt.draw()
        plt.pause(1.0)
        plt.close(fig)   # 👈 이제 이걸 남겨도 괜찮음 (이후 루프는 유지)
        plt.pause(0.5)   # 👈 main이 자연스럽게 이어질 여유시간

    fig.canvas.mpl_connect('button_press_event', on_press)
    fig.canvas.mpl_connect('motion_notify_event', on_motion)
    fig.canvas.mpl_connect('button_release_event', on_release)

    plt.show(block=True)
    return curves[0], curves[1]
