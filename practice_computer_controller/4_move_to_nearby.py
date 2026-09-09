import bettercam
from ultralytics import YOLO
import ctypes
import time
import math


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

VK_ESCAPE = 0x1B


def esc_pressed():
    return bool(
        user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000
    )


# =========================================================
# SendInput
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


def send_move(dx, dy):

    extra = ctypes.c_ulong(0)

    mouse = MOUSEINPUT(
        int(dx),
        int(dy),
        0,
        0x0001,
        0,
        ctypes.pointer(extra)
    )

    inp = INPUT(
        0,
        mouse
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

        # -------------------------------------------------
        # ESC 保护
        # -------------------------------------------------

        if esc_pressed():
            return False

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

    return True


# =========================================================
# 鼠标移动计算
# =========================================================

CENTER_X = 1280
TOLERANCE = 20

INPUT_DX = 50

A = 53.4
E_MAX = 952
C = 932


def mouse_moves(target_x):

    e = abs(
        target_x - CENTER_X
    )

    if e <= TOLERANCE:
        return 0

    n = A * math.log(
        E_MAX / (e + C)
    )

    return math.ceil(n)


# =========================================================
# YOLO 检测
# =========================================================

def detect_target(frame):

    result = model(
        frame,
        verbose=False
    )[0]

    if len(result.boxes) == 0:
        return None

    idx = result.boxes.conf.argmax()

    box = result.boxes[idx]

    x1, y1, x2, y2 = box.xyxy[0]

    x1 = float(x1)
    y1 = float(y1)
    x2 = float(x2)
    y2 = float(y2)

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    return cx, cy


# =========================================================
# MAIN
# =========================================================

# ---------------------------------------------------------
# 移动状态锁
#
# False = 可以检测
# True  = 正在移动，禁止检测
# ---------------------------------------------------------

moving = False


try:

    while True:

        # =================================================
        # ESC 保护
        # =================================================

        if esc_pressed():

            print("ESC pressed，程序停止")

            break


        # =================================================
        # 移动锁
        #
        # 理论上 smooth_move() 是阻塞的，
        # 这里作为额外保护。
        # =================================================

        if moving:

            continue


        # =================================================
        # 截屏
        # =================================================

        frame = camera.grab()

        if frame is None:

            continue


        # =================================================
        # YOLO 检测
        # =================================================

        target = detect_target(frame)

        if target is None:

            continue


        # =================================================
        # 目标中心
        # =================================================

        target_x, target_y = target


        # =================================================
        # X误差
        # =================================================

        error_x = (
            target_x -
            CENTER_X
        )


        # =================================================
        # 计算移动次数
        # =================================================

        steps = mouse_moves(
            target_x
        )

        if steps == 0:

            continue


        # =================================================
        # 判断左右方向
        # =================================================

        if error_x > 0:

            dx = -steps * INPUT_DX

        else:

            dx = steps * INPUT_DX


        # =================================================
        # 开始移动
        #
        # 加锁
        # =================================================

        moving = True


        # =================================================
        # 执行完整移动
        #
        # smooth_move 没结束之前，
        # 不会进行下一次 YOLO
        # =================================================

        success = smooth_move(
            dx=dx,
            dy=0
        )


        # =================================================
        # 移动完成
        #
        # 解锁
        # =================================================

        moving = False


        # =================================================
        # ESC
        # =================================================

        if not success:

            print(
                "ESC pressed，程序停止"
            )

            break


finally:

    camera.release()

    print(
        "程序结束"
    )