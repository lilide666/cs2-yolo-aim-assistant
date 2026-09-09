"""
最小延迟视觉伺服 - 截取中心640, YOLO识别, 一步到位移动, 按下/松开鼠标

流程(每帧)：
  1. 截取屏幕中心 640x640
  2. YOLO 检测，取【任一】目标（不选最高可信度，省时间）
  3. 一步到位: dx=+2.36*(x-320), dy=+2.36*(y-320)  立即移动
  4. 移动后立即按下/松开鼠标
按 ESC 长按退出
"""
import ctypes
import time

import bettercam
from ultralytics import YOLO

# ============================================================
# 配置
# ============================================================
MODEL_PATH = r"D:\PycharmProjects\computer_controller\runs\18_model_aug\weights\last.engine"
CAPTURE_SIZE = 640
CONF = 0.35
TARGET_CLASS = 1          # head_t

MOVE_K = 2.36             # 一步到位映射系数（标定）
CLICK_HOLD = 0.02         # 鼠标按下到松开的时间(秒)
ESCAPE_VK = 0x1B

# ============================================================
# Windows 底层输入
# ============================================================
user32 = ctypes.windll.user32
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]

class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("mi", MOUSEINPUT)]


def send_move(dx, dy):
    extra = ctypes.c_ulong(0)
    mi = MOUSEINPUT(int(dx), int(dy), 0, MOUSEEVENTF_MOVE, 0, ctypes.pointer(extra))
    inp = INPUT(0, mi)
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def mouse_down():
    extra = ctypes.c_ulong(0)
    mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, ctypes.pointer(extra))
    inp = INPUT(0, mi)
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def mouse_up():
    extra = ctypes.c_ulong(0)
    mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, ctypes.pointer(extra))
    inp = INPUT(0, mi)
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def esc_down():
    return bool(user32.GetAsyncKeyState(ESCAPE_VK) & 0x8000)


# ============================================================
# 主流程
# ============================================================
def main():
    model = YOLO(MODEL_PATH)

    sm_w = user32.GetSystemMetrics(0)
    sm_h = user32.GetSystemMetrics(1)
    cs = min(CAPTURE_SIZE, sm_w, sm_h)
    left = (sm_w - cs) // 2
    top = (sm_h - cs) // 2
    camera = bettercam.create(output_color="BGR",
                              region=(left, top, left + cs, top + cs))

    center = CAPTURE_SIZE / 2.0
    print("最小延迟视觉伺服启动。按 ESC 退出。")
    print(f"映射: dx=+{MOVE_K}*(x-{center:.0f}), dy=+{MOVE_K}*(y-{center:.0f})")

    clicked = False   # 是否已按下（防止未松开重复按下）

    try:
        while True:
            if esc_down():
                break

            # 1) 截取中心 640
            frame = camera.grab()
            if frame is None:
                continue

            # 2) YOLO 检测：取【任一】目标（不选最高conf，省时间）
            result = model(frame, verbose=False)[0]
            box_xyxy = None
            conf = 0.0
            for b in result.boxes:
                if int(b.cls.item()) != TARGET_CLASS:
                    continue
                c = b.conf.item()
                if c >= CONF:
                    box_xyxy = b.xyxy[0].tolist()
                    conf = c
                    break          # 取第一个就退出，不比较
                # 若第一个不达标，继续找；找到即停

            if box_xyxy is None:
                # 松开鼠标（如果按着）
                if clicked:
                    mouse_up()
                    clicked = False
                continue

            # 目标中心
            x1, y1, x2, y2 = box_xyxy
            tx = (x1 + x2) / 2.0
            ty = (y1 + y2) / 2.0

            # 3) 一步到位移动
            du = tx - center
            dv = ty - center
            send_move(int(MOVE_K * du), int(MOVE_K * dv))

            # 4) 按下/松开鼠标
            mouse_down()
            time.sleep(CLICK_HOLD)
            mouse_up()
            clicked = False

    except KeyboardInterrupt:
        pass

    # 确保松开鼠标
    mouse_up()
    camera.release()


if __name__ == "__main__":
    main()