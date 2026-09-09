import bettercam
from ultralytics import YOLO
import cv2
import keyboard
import ctypes
import time
import os
import csv
import math
from datetime import datetime


# =========================================================
# YOLO
# =========================================================

model = YOLO(
    r"D:\PycharmProjects\computer_controller\runs_1200\yolo26x_exp1\weights\best.pt"
)


# =========================================================
# BetterCam
# =========================================================

camera = bettercam.create(
    output_color="BGR"
)


# =========================================================
# Windows
# =========================================================

user32 = ctypes.windll.user32


# =========================================================
# MOUSEINPUT
# =========================================================

class MOUSEINPUT(ctypes.Structure):

    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong))
    ]


class INPUT(ctypes.Structure):

    _fields_ = [
        ("type", ctypes.c_ulong),
        ("mi", MOUSEINPUT)
    ]


# =========================================================
# 鼠标位置
# =========================================================

class POINT(ctypes.Structure):

    _fields_ = [
        ("x", ctypes.c_long),
        ("y", ctypes.c_long)
    ]


def get_mouse_pos():

    point = POINT()

    user32.GetCursorPos(
        ctypes.byref(point)
    )

    return point.x, point.y


# =========================================================
# SendInput
# =========================================================

def send_move(dx, dy):

    extra = ctypes.c_ulong(0)

    mi = MOUSEINPUT(
        int(dx),
        int(dy),
        0,
        0x0001,
        0,
        ctypes.pointer(extra)
    )

    inp = INPUT(
        0,
        mi
    )

    user32.SendInput(
        1,
        ctypes.byref(inp),
        ctypes.sizeof(inp)
    )


# =========================================================
# 平滑移动
# =========================================================

def smooth_move(
    dx,
    dy,
    steps=100,
    delay=0.0001
):

    step_x = dx / steps
    step_y = dy / steps

    acc_x = 0.0
    acc_y = 0.0

    for _ in range(steps):

        acc_x += step_x
        acc_y += step_y

        move_x = int(acc_x)
        move_y = int(acc_y)

        acc_x -= move_x
        acc_y -= move_y

        if move_x != 0 or move_y != 0:

            send_move(
                move_x,
                move_y
            )

        time.sleep(delay)


# =========================================================
# YOLO检测
# =========================================================

def detect_target(frame):

    yolo_start = time.perf_counter()

    result = model(
        frame,
        verbose=False
    )[0]

    yolo_time = (
        time.perf_counter()
        - yolo_start
    ) * 1000

    # -----------------------------------------------------
    # 没有目标
    # -----------------------------------------------------

    if len(result.boxes) == 0:

        return None, yolo_time

    # -----------------------------------------------------
    # 最高置信度目标
    # -----------------------------------------------------

    idx = result.boxes.conf.argmax()

    target = result.boxes[idx]

    # -----------------------------------------------------
    # 目标框
    # -----------------------------------------------------

    x1, y1, x2, y2 = target.xyxy[0]

    x1 = float(x1)
    y1 = float(y1)
    x2 = float(x2)
    y2 = float(y2)

    # -----------------------------------------------------
    # 中心
    # -----------------------------------------------------

    target_cx = (
        x1 + x2
    ) / 2.0

    target_cy = (
        y1 + y2
    ) / 2.0

    # -----------------------------------------------------
    # 尺寸
    # -----------------------------------------------------

    target_w = x2 - x1
    target_h = y2 - y1

    # -----------------------------------------------------
    # 置信度
    # -----------------------------------------------------

    conf = float(
        target.conf
    )

    return {
        "cx": target_cx,
        "cy": target_cy,
        "w": target_w,
        "h": target_h,
        "conf": conf
    }, yolo_time


# =========================================================
# 实验结果
# =========================================================

RESULT_DIR = (
    r"D:\PycharmProjects\computer_controller"
    r"\experiment_results"
)

RESULT_FILE = os.path.join(
    RESULT_DIR,
    "experiment_results.csv"
)


# =========================================================
# 初始化 CSV
#
# 每一次鼠标输入保存一行
# =========================================================

def init_result_file():

    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )

    if not os.path.exists(RESULT_FILE):

        headers = [

            # -------------------------
            # 基本信息
            # -------------------------

            "实验时间",
            "实验编号",
            "输入次数",

            # -------------------------
            # 本次鼠标输入
            # -------------------------

            "单次输入X",
            "单次输入Y",

            # -------------------------
            # 累计鼠标输入
            # -------------------------

            "总输入X",
            "总输入Y",

            # -------------------------
            # 实验开始目标位置
            # -------------------------

            "初始目标X",
            "初始目标Y",

            # -------------------------
            # 移动前目标
            # -------------------------

            "移动前目标X",
            "移动前目标Y",

            # -------------------------
            # 移动后目标
            # -------------------------

            "移动后目标X",
            "移动后目标Y",

            # -------------------------
            # 本次目标变化
            # -------------------------

            "本次目标变化X",
            "本次目标变化Y",

            # -------------------------
            # 目标总变化
            # -------------------------

            "目标总变化X",
            "目标总变化Y",

            # -------------------------
            # 初始误差
            # -------------------------

            "初始误差X",
            "初始误差Y",
            "初始误差距离",

            # -------------------------
            # 当前误差
            # -------------------------

            "当前误差X",
            "当前误差Y",
            "当前误差距离",

            # -------------------------
            # 单次比例
            # -------------------------

            "单次Kx",
            "单次Ky",

            # -------------------------
            # 累计比例
            # -------------------------

            "累计Kx",
            "累计Ky",

            # -------------------------
            # 目标尺寸
            # -------------------------

            "目标宽度",
            "目标高度",

            # -------------------------
            # 置信度
            # -------------------------

            "目标置信度",

            # -------------------------
            # 时间
            # -------------------------

            "YOLO耗时_ms",
            "鼠标移动耗时_ms",

            # -------------------------
            # 鼠标实际位置
            # -------------------------

            "鼠标X",
            "鼠标Y"

        ]

        with open(
            RESULT_FILE,
            "w",
            newline="",
            encoding="utf-8-sig"
        ) as f:

            writer = csv.writer(f)

            writer.writerow(headers)


# =========================================================
# 保存一次采样
#
# 每移动一次保存一次
# =========================================================

def save_sample(

    experiment_count,
    step_count,

    input_dx,
    input_dy,

    total_input_x,
    total_input_y,

    start_x,
    start_y,

    previous_x,
    previous_y,

    current_x,
    current_y,

    delta_x,
    delta_y,

    total_delta_x,
    total_delta_y,

    initial_error_x,
    initial_error_y,
    initial_error_distance,

    error_x,
    error_y,
    error_distance,

    k_x,
    k_y,

    total_k_x,
    total_k_y,

    target_w,
    target_h,

    target_conf,

    yolo_time,
    move_time,

    mouse_x,
    mouse_y

):

    experiment_time = datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S.%f"
    )[:-3]

    row = [

        # -------------------------
        # 基本信息
        # -------------------------

        experiment_time,
        experiment_count,
        step_count,

        # -------------------------
        # 本次输入
        # -------------------------

        input_dx,
        input_dy,

        # -------------------------
        # 累计输入
        # -------------------------

        total_input_x,
        total_input_y,

        # -------------------------
        # 初始目标
        # -------------------------

        f"{start_x:.4f}",
        f"{start_y:.4f}",

        # -------------------------
        # 移动前目标
        # -------------------------

        f"{previous_x:.4f}",
        f"{previous_y:.4f}",

        # -------------------------
        # 移动后目标
        # -------------------------

        f"{current_x:.4f}",
        f"{current_y:.4f}",

        # -------------------------
        # 本次变化
        # -------------------------

        f"{delta_x:.4f}",
        f"{delta_y:.4f}",

        # -------------------------
        # 总变化
        # -------------------------

        f"{total_delta_x:.4f}",
        f"{total_delta_y:.4f}",

        # -------------------------
        # 初始误差
        # -------------------------

        f"{initial_error_x:.4f}",
        f"{initial_error_y:.4f}",
        f"{initial_error_distance:.4f}",

        # -------------------------
        # 当前误差
        # -------------------------

        f"{error_x:.4f}",
        f"{error_y:.4f}",
        f"{error_distance:.4f}",

        # -------------------------
        # 单次 K
        # -------------------------

        f"{k_x:.6f}",
        f"{k_y:.6f}",

        # -------------------------
        # 累计 K
        # -------------------------

        f"{total_k_x:.6f}",
        f"{total_k_y:.6f}",

        # -------------------------
        # 目标尺寸
        # -------------------------

        f"{target_w:.4f}",
        f"{target_h:.4f}",

        # -------------------------
        # 置信度
        # -------------------------

        f"{target_conf:.6f}",

        # -------------------------
        # 时间
        # -------------------------

        f"{yolo_time:.4f}",
        f"{move_time:.4f}",

        # -------------------------
        # 鼠标位置
        # -------------------------

        mouse_x,
        mouse_y

    ]

    # -----------------------------------------------------
    # 立即写入磁盘
    # -----------------------------------------------------

    with open(
        RESULT_FILE,
        "a",
        newline="",
        encoding="utf-8-sig"
    ) as f:

        writer = csv.writer(f)

        writer.writerow(row)

        # 确保这一条立即写入
        f.flush()


# =========================================================
# MAIN
# =========================================================

if __name__ == '__main__':

    # =====================================================
    # 初始化 CSV
    # =====================================================

    init_result_file()

    print()
    print("=" * 100)
    print("实验数据保存位置：")
    print(RESULT_FILE)
    print("=" * 100)

    # =====================================================
    # 实验编号
    # =====================================================

    experiment_count = 0

    # =====================================================
    # 中心位置
    # =====================================================

    CENTER_X = 1280
    CENTER_Y = 800

    # =====================================================
    # 中心容差
    # =====================================================

    CENTER_TOLERANCE = 20

    # =====================================================
    # 每次鼠标输入
    # =====================================================

    INPUT_DX = 50
    INPUT_DY = 0

    # =====================================================
    # 平滑参数
    # =====================================================

    SMOOTH_STEPS = 100
    SMOOTH_DELAY = 0.0001

    # =====================================================
    # 主循环
    # =====================================================

    while True:

        # =================================================
        # 检测初始目标
        # =================================================

        frame = camera.grab()

        if frame is None:
            continue

        target, yolo_time = detect_target(frame)

        if target is None:
            continue

        # =================================================
        # 初始目标位置
        # =================================================

        current_x = target["cx"]
        current_y = target["cy"]

        # =================================================
        # 判断是否已经在中心
        # =================================================

        error_x = current_x - CENTER_X
        error_y = current_y - CENTER_Y

        error_distance = math.sqrt(
            error_x ** 2 +
            error_y ** 2
        )

        if error_distance <= CENTER_TOLERANCE:

            print()
            print("=" * 100)
            print("当前目标已经在中心")
            print("=" * 100)

            print(
                f"目标位置 : "
                f"({current_x:.2f}, {current_y:.2f})"
            )

            print(
                f"误差     : "
                f"({error_x:+.2f}, {error_y:+.2f})"
            )

            print(
                f"误差距离 : "
                f"{error_distance:.2f}"
            )

            print("=" * 100)

            time.sleep(1)

            continue

        # =================================================
        # 新实验
        # =================================================

        experiment_count += 1

        step_count = 0

        total_input_x = 0
        total_input_y = 0

        # =================================================
        # 实验开始目标
        # =================================================

        start_x = current_x
        start_y = current_y

        # =================================================
        # 上一次目标位置
        # =================================================

        previous_x = current_x
        previous_y = current_y

        # =================================================
        # 初始误差
        # =================================================

        initial_error_x = (
            start_x -
            CENTER_X
        )

        initial_error_y = (
            start_y -
            CENTER_Y
        )

        initial_error_distance = math.sqrt(
            initial_error_x ** 2 +
            initial_error_y ** 2
        )

        # =================================================
        # 鼠标位置
        # =================================================

        mouse_x, mouse_y = get_mouse_pos()

        # =================================================
        # 实验开始输出
        # =================================================

        print()
        print()
        print("=" * 100)
        print(
            f"实验 #{experiment_count} 开始"
        )
        print("=" * 100)

        print(
            f"目标初始位置 : "
            f"({start_x:.4f}, {start_y:.4f})"
        )

        print(
            f"鼠标位置     : "
            f"({mouse_x}, {mouse_y})"
        )

        print(
            f"初始误差     : "
            f"({initial_error_x:+.4f}, "
            f"{initial_error_y:+.4f})"
        )

        print(
            f"初始误差距离 : "
            f"{initial_error_distance:.4f}"
        )

        print("=" * 100)

        # =================================================
        # 当前实验循环
        # =================================================

        while True:

            # =================================================
            # 退出
            # =================================================

            if keyboard.is_pressed("q"):

                print()
                print("Q pressed")
                camera.release()
                cv2.destroyAllWindows()
                raise SystemExit

            key = cv2.waitKey(1)

            if key == ord("p"):

                print()
                print("P pressed")
                camera.release()
                cv2.destroyAllWindows()
                raise SystemExit

            # =================================================
            # 鼠标移动
            # =================================================

            move_start = time.perf_counter()

            smooth_move(
                dx=INPUT_DX,
                dy=INPUT_DY,
                steps=SMOOTH_STEPS,
                delay=SMOOTH_DELAY
            )

            move_time = (
                time.perf_counter()
                - move_start
            ) * 1000

            # =================================================
            # 累计输入
            # =================================================

            step_count += 1

            total_input_x += INPUT_DX
            total_input_y += INPUT_DY

            # =================================================
            # 移动之后重新检测目标
            #
            # 只要检测不到，就一直重新检测
            # =================================================

            target = None

            while target is None:

                # ---------------------------------------------
                # 退出检测
                # ---------------------------------------------

                if keyboard.is_pressed("q"):

                    print()
                    print("Q pressed")
                    camera.release()
                    cv2.destroyAllWindows()
                    raise SystemExit

                key = cv2.waitKey(1)

                if key == ord("p"):

                    print()
                    print("P pressed")
                    camera.release()
                    cv2.destroyAllWindows()
                    raise SystemExit

                # ---------------------------------------------
                # 抓取画面
                # ---------------------------------------------

                frame = camera.grab()

                if frame is None:
                    continue

                # ---------------------------------------------
                # YOLO
                # ---------------------------------------------

                target, yolo_time = detect_target(frame)

            # =================================================
            # 移动之后的目标
            # =================================================

            current_x = target["cx"]
            current_y = target["cy"]

            # =================================================
            # 本次目标变化
            #
            # 当前 - 移动前
            # =================================================

            delta_x = (
                current_x -
                previous_x
            )

            delta_y = (
                current_y -
                previous_y
            )

            # =================================================
            # 单次 K
            # =================================================

            if INPUT_DX != 0:

                k_x = (
                    delta_x /
                    INPUT_DX
                )

            else:

                k_x = 0.0

            if INPUT_DY != 0:

                k_y = (
                    delta_y /
                    INPUT_DY
                )

            else:

                k_y = 0.0

            # =================================================
            # 目标总变化
            # =================================================

            total_delta_x = (
                current_x -
                start_x
            )

            total_delta_y = (
                current_y -
                start_y
            )

            # =================================================
            # 当前误差
            # =================================================

            error_x = (
                current_x -
                CENTER_X
            )

            error_y = (
                current_y -
                CENTER_Y
            )

            error_distance = math.sqrt(
                error_x ** 2 +
                error_y ** 2
            )

            # =================================================
            # 累计 K
            # =================================================

            if total_input_x != 0:

                total_k_x = (
                    total_delta_x /
                    total_input_x
                )

            else:

                total_k_x = 0.0

            if total_input_y != 0:

                total_k_y = (
                    total_delta_y /
                    total_input_y
                )

            else:

                total_k_y = 0.0

            # =================================================
            # 鼠标当前实际位置
            # =================================================

            mouse_x, mouse_y = get_mouse_pos()

            # =================================================
            # 输出
            # =================================================

            print()
            print(
                f"[实验 #{experiment_count}] "
                f"第 {step_count} 次移动"
            )

            print("-" * 100)

            print(
                f"本次输入     : "
                f"({INPUT_DX:+d}, {INPUT_DY:+d})"
            )

            print(
                f"累计输入     : "
                f"({total_input_x:+d}, "
                f"{total_input_y:+d})"
            )

            print()

            print(
                f"移动前目标   : "
                f"({previous_x:.4f}, "
                f"{previous_y:.4f})"
            )

            print(
                f"移动后目标   : "
                f"({current_x:.4f}, "
                f"{current_y:.4f})"
            )

            print(
                f"目标变化     : "
                f"({delta_x:+.4f}, "
                f"{delta_y:+.4f})"
            )

            print()

            print(
                f"单次 K       : "
                f"({k_x:+.6f}, "
                f"{k_y:+.6f})"
            )

            print(
                f"累计 K       : "
                f"({total_k_x:+.6f}, "
                f"{total_k_y:+.6f})"
            )

            print()

            print(
                f"目标总变化   : "
                f"({total_delta_x:+.4f}, "
                f"{total_delta_y:+.4f})"
            )

            print()

            print(
                f"当前误差     : "
                f"({error_x:+.4f}, "
                f"{error_y:+.4f})"
            )

            print(
                f"当前误差距离 : "
                f"{error_distance:.4f}"
            )

            print()

            print(
                f"目标尺寸     : "
                f"{target['w']:.2f} x "
                f"{target['h']:.2f}"
            )

            print(
                f"置信度       : "
                f"{target['conf']:.4f}"
            )

            print()

            print(
                f"YOLO耗时     : "
                f"{yolo_time:.4f} ms"
            )

            print(
                f"移动耗时     : "
                f"{move_time:.4f} ms"
            )

            print(
                f"鼠标位置     : "
                f"({mouse_x}, {mouse_y})"
            )

            print("-" * 100)

            # =================================================
            # 关键：
            # 每一次移动完成后立即保存
            # =================================================

            save_sample(

                experiment_count=experiment_count,
                step_count=step_count,

                input_dx=INPUT_DX,
                input_dy=INPUT_DY,

                total_input_x=total_input_x,
                total_input_y=total_input_y,

                start_x=start_x,
                start_y=start_y,

                previous_x=previous_x,
                previous_y=previous_y,

                current_x=current_x,
                current_y=current_y,

                delta_x=delta_x,
                delta_y=delta_y,

                total_delta_x=total_delta_x,
                total_delta_y=total_delta_y,

                initial_error_x=initial_error_x,
                initial_error_y=initial_error_y,
                initial_error_distance=initial_error_distance,

                error_x=error_x,
                error_y=error_y,
                error_distance=error_distance,

                k_x=k_x,
                k_y=k_y,

                total_k_x=total_k_x,
                total_k_y=total_k_y,

                target_w=target["w"],
                target_h=target["h"],

                target_conf=target["conf"],

                yolo_time=yolo_time,
                move_time=move_time,

                mouse_x=mouse_x,
                mouse_y=mouse_y
            )

            print(
                "✓ 本次移动数据已保存"
            )

            # =================================================
            # 判断是否到达中心
            # =================================================

            if error_distance <= CENTER_TOLERANCE:

                print()
                print("#" * 100)
                print(
                    f"实验 #{experiment_count} 完成"
                )
                print("#" * 100)

                print(
                    f"总输入次数   : "
                    f"{step_count}"
                )

                print(
                    f"总输入量     : "
                    f"({total_input_x}, "
                    f"{total_input_y})"
                )

                print(
                    f"目标总变化   : "
                    f"({total_delta_x:+.4f}, "
                    f"{total_delta_y:+.4f})"
                )

                print(
                    f"最终目标位置 : "
                    f"({current_x:.4f}, "
                    f"{current_y:.4f})"
                )

                print(
                    f"最终误差距离 : "
                    f"{error_distance:.4f}"
                )

                print(
                    f"最终累计 K   : "
                    f"({total_k_x:+.6f}, "
                    f"{total_k_y:+.6f})"
                )

                print()
                print(
                    "本实验所有移动数据均已保存。"
                )

                print("#" * 100)

                break

            # =================================================
            # 更新上一位置
            #
            # 下一次移动会从这个位置开始计算 Δ
            # =================================================

            previous_x = current_x
            previous_y = current_y

        # =====================================================
        # 下一实验
        # =====================================================

        print()
        print(
            "等待 10 秒后开始下一次实验..."
        )

        time.sleep(10)


# =========================================================
# 释放资源
# =========================================================

camera.release()

cv2.destroyAllWindows()

print()
print("程序结束。")