import time
import ctypes
import cv2
import bettercam


# ============================================================
# 配置
# ============================================================

CAPTURE_SIZE = 640

# 第一组：20ms × 5
INTERVAL_1 = 0.020
COUNT_1 = 5

# 两组之间等待
WAIT_BETWEEN = 0.050

# 第二组：20ms × 5
INTERVAL_2 = 0.020
COUNT_2 = 5

# 启动后等待 5 秒
START_DELAY = 5.0


# ============================================================
# 获取屏幕中心区域
# ============================================================

user32 = ctypes.windll.user32

screen_w = user32.GetSystemMetrics(0)
screen_h = user32.GetSystemMetrics(1)

cs = min(CAPTURE_SIZE, screen_w, screen_h)

left = (screen_w - cs) // 2
top = (screen_h - cs) // 2

region = (
    left,
    top,
    left + cs,
    top + cs
)


# ============================================================
# BetterCam
# ============================================================

cam = bettercam.create(
    output_color="BGR",
    region=region
)


# ============================================================
# 截图
# ============================================================

frames = []
timestamps = []


try:

    print("=" * 70)
    print("截图回放测试")
    print("=" * 70)

    print()
    print(f"启动后等待 {START_DELAY:.0f} 秒...")

    # --------------------------------------------------------
    # 启动延迟
    # --------------------------------------------------------

    time.sleep(START_DELAY)

    print("开始截图！")
    print()


    capture_start = time.perf_counter()


    # ========================================================
    # 第一组
    # ========================================================

    for i in range(COUNT_1):

        # 绝对时间调度
        target_time = (
            capture_start +
            i * INTERVAL_1
        )

        # 等到目标时间
        while True:

            now = time.perf_counter()
            remain = target_time - now

            if remain <= 0:
                break

            if remain > 0.002:
                time.sleep(remain - 0.001)


        # 截图
        frame = cam.grab()

        actual_time = time.perf_counter()

        if frame is not None:

            frames.append(frame.copy())

            timestamps.append(
                actual_time - capture_start
            )

            print(
                f"第 {len(frames):2d} 张"
                f"  时间 = "
                f"{(actual_time - capture_start) * 1000:8.3f} ms"
            )


    # ========================================================
    # 两组之间等待 50ms
    # ========================================================

    wait_target = (
        capture_start +
        (COUNT_1 - 1) * INTERVAL_1 +
        WAIT_BETWEEN
    )

    while True:

        now = time.perf_counter()
        remain = wait_target - now

        if remain <= 0:
            break

        if remain > 0.002:
            time.sleep(remain - 0.001)


    # ========================================================
    # 第二组
    # ========================================================

    second_start = time.perf_counter()

    for i in range(COUNT_2):

        target_time = (
            second_start +
            i * INTERVAL_2
        )

        while True:

            now = time.perf_counter()
            remain = target_time - now

            if remain <= 0:
                break

            if remain > 0.002:
                time.sleep(remain - 0.001)


        frame = cam.grab()

        actual_time = time.perf_counter()

        if frame is not None:

            frames.append(frame.copy())

            timestamps.append(
                actual_time - capture_start
            )

            print(
                f"第 {len(frames):2d} 张"
                f"  时间 = "
                f"{(actual_time - capture_start) * 1000:8.3f} ms"
            )


    # ========================================================
    # 截图完成
    # ========================================================

    print()
    print("=" * 70)
    print(f"截图完成，共 {len(frames)} 张")
    print("=" * 70)

    print()
    print("查看方式：")
    print("  A / ←  上一张")
    print("  D / →  下一张")
    print("  ESC    退出")
    print()


    if not frames:
        print("没有成功截取到图片。")
        raise SystemExit


    # ========================================================
    # 回放
    # ========================================================

    index = 0

    window_name = "Screenshot Viewer"

    cv2.namedWindow(
        window_name,
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        window_name,
        800,
        800
    )


    while True:

        frame = frames[index]

        # ----------------------------------------------------
        # 显示信息
        # ----------------------------------------------------

        display = frame.copy()

        text1 = (
            f"{index + 1} / {len(frames)}"
        )

        text2 = (
            f"T = {timestamps[index] * 1000:.3f} ms"
        )

        cv2.putText(
            display,
            text1,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            display,
            text2,
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )


        cv2.imshow(
            window_name,
            display
        )


        # ----------------------------------------------------
        # 键盘
        # ----------------------------------------------------

        key = cv2.waitKey(0) & 0xFF


        # ESC
        if key == 27:
            break


        # A：上一张
        elif key in (ord('a'), ord('A')):

            index -= 1

            if index < 0:
                index = len(frames) - 1


        # D：下一张
        elif key in (ord('d'), ord('D')):

            index += 1

            if index >= len(frames):
                index = 0


        # 方向键
        elif key == 81:       # ←
            index -= 1

            if index < 0:
                index = len(frames) - 1


        elif key == 83:       # →
            index += 1

            if index >= len(frames):
                index = 0


finally:

    cam.release()
    cv2.destroyAllWindows()