import bettercam
from ultralytics import YOLO
import cv2
import keyboard

model = YOLO(r"D:\PycharmProjects\computer_controller\runs_1200\yolo26x_exp1\weights\best.pt")

camera = bettercam.create(
    output_color="BGR"
)

locked = False

x_locked = None
y_locked = None

x_velocity = None
y_velocity = None

first_time = True


# ----------- sendinput ------------

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

# 获取鼠标位置
class POINT(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_long),
        ("y", ctypes.c_long)
    ]

def get_mouse_pos():
    point = POINT()
    ctypes.windll.user32.GetCursorPos(
        ctypes.byref(point)
    )
    return point.x, point.y

# 移动鼠标
def send_move(dx, dy):

    dx = int(dx)
    dy = int(dy)

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


# ---------- PID ---------
class PID:
    def __init__(self, kp=0.3, ki=0.0, kd=0.1):
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.last_error = 0
        self.integral = 0


    def update(self, error):

        # 积分
        self.integral += error
        if abs(error) < 100:
            self.integral *= 0.3

        # 微分
        derivative = error - self.last_error

        # PID输出
        output = (
            self.kp * error +
            self.ki * self.integral +
            self.kd * derivative
        )

        self.last_error = error

        return output

    def reset(self):
        self.last_error = 0
        self.integral = 0


# ---------- main ----------
pid_x = PID(
    kp=1,
    ki=0.01,
    kd=0.05
)

pid_y = PID(
    kp=1,
    ki=0.01,
    kd=0.05
)


if __name__ == '__main__':
    while True:

        frame = camera.grab()

        if frame is None:
            continue

        result = model(
            frame,
            verbose=False
        )[0]

        if len(result.boxes) == 0:
            locked = False
            pid_x.reset()
            pid_y.reset()
            continue

        show = result.plot()

        # cv2.imshow(
        #     "screen",
        #     show
        # )

        # print("\n orig_shape:",result.orig_shape)
        # print("\n boxes.xyxy:",result.boxes.xyxy)
        # print("\n boxes.conf:",result.boxes.conf)
        # print("\n boxes.cls:",result.boxes.cls)
        # print("\n speed:",result.speed)
        # print("\n names:",result.names)

        # 锁定目标
        if first_time:
            # 选取conf最高的目标
            idx = result.boxes.conf.argmax()
            target = result.boxes[idx]
            # print("target.xyxy:", target.xyxy)
            # print("target.conf:", target.conf)
            # print("target.cls:", target.cls)

            # 计算中心点
            x1, y1, x2, y2 = target.xyxy[0]

            target_cx = ((x1 + x2) / 2).item()
            target_cy = ((y1 + y2) / 2).item()

            target_cx_locked = target_cx
            target_cy_locked = target_cy

            velocity_x = 0
            velocity_y = 0

            locked = True
            first_time = False

            pid_x.reset()
            pid_y.reset()


        if not locked:
            # 选取最近的目标
            idx = result.boxes.conf.argmax()
            target = result.boxes[idx]
            # print("target.xyxy:", target.xyxy)
            # print("target.conf:", target.conf)
            # print("target.cls:", target.cls)

            # 计算中心点
            x1, y1, x2, y2 = target.xyxy[0]

            target_cx = ((x1 + x2) / 2).item()
            target_cy = ((y1 + y2) / 2).item()

            target_cx_locked = target_cx
            target_cy_locked = target_cy

            velocity_x = 0
            velocity_y = 0

            locked = True

            pid_x.reset()
            pid_y.reset()

        else:
            target_x_last = target_cx_locked
            target_y_last = target_cy_locked

            target_x_predict = target_cx_locked + velocity_x
            target_y_predict = target_cy_locked + velocity_y

            distance = [0] * len(result.boxes)

            for i, box in enumerate(result.boxes):
                x1, y1, x2, y2 = box.xyxy[0]
                target_cx = ((x1 + x2) / 2).item()
                target_cy = ((y1 + y2) / 2).item()

                distance[i] = (target_x_predict-target_cx)**2 + (target_y_predict-target_cy)**2

            min_distance = min(distance)
            min_distance_idx = distance.index(min_distance)

            if min_distance < 50**2:
                target = result.boxes[min_distance_idx]

                x1, y1, x2, y2 = target.xyxy[0]

                target_cx_locked = ((x1 + x2) / 2).item()
                target_cy_locked = ((y1 + y2) / 2).item()

                velocity_x = target_cx_locked - target_x_last
                velocity_y = target_cy_locked - target_y_last

            else:
                target_cx_locked = target_cx_locked
                target_cy_locked = target_cy_locked

                velocity_x = 0
                velocity_y = 0

                locked = False

                pid_x.reset()
                pid_y.reset()


        # 获取鼠标位置
        mouse_x, mouse_y = get_mouse_pos()
        # print("mouse_x, mouse_y:", mouse_x, mouse_y)

        # 误差
        error_x = target_cx_locked - mouse_x
        error_y = target_cy_locked - mouse_y

        # PID输出
        move_x = pid_x.update(error_x)
        move_y = pid_y.update(error_y)

        # 平滑移动到目标位置
        smooth_move(
            dx=move_x,
            dy=move_y,
            steps=100,
            delay=0.0001
        )

        # send_move(
        #     move_x,
        #     move_y
        # )

        # 获取鼠标位置
        mouse_x, mouse_y = get_mouse_pos()
        # print("mouse_x_after, mouse_y_after:", mouse_x, mouse_y)


        # time.sleep(5)

        if cv2.waitKey(1)==ord("p"):
            print("P pressed")
            break

        if keyboard.is_pressed("q"):
            print("Q pressed")
            break



    cv2.destroyAllWindows()