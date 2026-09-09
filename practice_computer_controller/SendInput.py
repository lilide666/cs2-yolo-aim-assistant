import ctypes
import time


user32 = ctypes.windll.user32


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

    mi = MOUSEINPUT(
        dx,
        dy,
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


# =========================
# 平滑移动
# =========================

def smooth_move(dx, dy, steps=100, delay=0.005):

    step_x = dx / steps
    step_y = dy / steps

    acc_x = 0
    acc_y = 0


    for i in range(steps):

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



time.sleep(2)


# 平滑向右1000

smooth_move(
    5000,
    0,
    steps=1,
    delay=0.00001
)


time.sleep(1)


# 平滑向左1000

smooth_move(
    -5000,
    0,
    steps=100,
    delay=0.000001
)

time.sleep(1)

send_move(
    5000,
    0
)

time.sleep(1)

send_move(
    -5000,
    0
)