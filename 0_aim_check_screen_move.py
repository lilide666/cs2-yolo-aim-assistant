"""
最小延迟视觉伺服 - YOLO 检测 + 鼠标一步移动 + 移动稳定后按下

循环（每 0.25s）：
  1. 截取中心640
  2. YOLO检测目标(取任一)
  3. 一步移动: dx=+2.36*(x-320), dy=+2.36*(y-320)
  4. 【变化】移动后不断截取中心32*32，对比画面是否变化
     - 若画面仍在变化 -> 继续等待
     - 若画面稳定(不再变化) -> 立刻按下/松开
  5. 补足到 0.25s -> 下轮
按 K 暂停/继续
"""
import ctypes
import time

import bettercam
import numpy as np
from ultralytics import YOLO

MODEL_PATH = r"D:\PycharmProjects\aim\head_body_640_yolo26n_screen_640\weights\best.engine"
CAPTURE_SIZE = 640
CONF = 0.35
TARGET_CLASS = 1
MOVE_K = 2.36

# ---- 画面变化判定参数 ----
CHANGE_WINDOW = 32          # 中心截取区域边长(像素)
CHANGE_POLL = 0.01          # 截取轮询间隔(秒)
DIFF_THRESHOLD = 8.0        # 判为"变化"的平均绝对差阈值
STABLE_COUNT = 3            # 连续 N 次判定为"稳定"才算真正稳定(防抖动)
MAX_SETTLE_WAIT = 0.5       # 最长等待稳定时间(秒)，超时也按下(安全阀)

CLICK_HOLD = 0.02
CLICK_INTERVAL = 0.3
K_VK = 0x4B

user32 = ctypes.windll.user32
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx",ctypes.c_long),("dy",ctypes.c_long),("mouseData",ctypes.c_ulong),
                ("dwFlags",ctypes.c_ulong),("time",ctypes.c_ulong),("dwExtraInfo",ctypes.POINTER(ctypes.c_ulong))]
class INPUT(ctypes.Structure):
    _fields_ = [("type",ctypes.c_ulong),("mi",MOUSEINPUT)]

def send_move(dx,dy):
    extra=ctypes.c_ulong(0); mi=MOUSEINPUT(int(dx),int(dy),0,MOUSEEVENTF_MOVE,0,ctypes.pointer(extra))
    inp=INPUT(0,mi); user32.SendInput(1,ctypes.byref(inp),ctypes.sizeof(inp))
def mouse_down():
    extra=ctypes.c_ulong(0); mi=MOUSEINPUT(0,0,0,MOUSEEVENTF_LEFTDOWN,0,ctypes.pointer(extra))
    inp=INPUT(0,mi); user32.SendInput(1,ctypes.byref(inp),ctypes.sizeof(inp))
def mouse_up():
    extra=ctypes.c_ulong(0); mi=MOUSEINPUT(0,0,0,MOUSEEVENTF_LEFTUP,0,ctypes.pointer(extra))
    inp=INPUT(0,mi); user32.SendInput(1,ctypes.byref(inp),ctypes.sizeof(inp))
def k_down(): return bool(user32.GetAsyncKeyState(K_VK)&0x8000)


def center_patch(frame, size=CHANGE_WINDOW):
    """从整帧中裁剪中心 size×size 区域，转成灰度 float32 便于比对"""
    h, w = frame.shape[:2]
    half = size // 2
    cxl = w // 2 - half
    cyt = h // 2 - half
    patch = frame[cyt:cyt+size, cxl:cxl+size]
    gray = patch[:, :, 0] if patch.ndim == 3 else patch   # 单通道(B)即可
    return gray.astype(np.float32)


def wait_until_settled(cam):
    """
    移动后调用：不断截取中心32×32，判断画面是否还在变化。
    返回 True 表示已稳定(可按下)；False 表示超时(安全阀)。
    逻辑：
      - 每次截取与上一次的 patch 求平均绝对差 MAD
      - MAD > DIFF_THRESHOLD  -> 仍在变化, 清零连续稳定计数
      - MAD <= DIFF_THRESHOLD -> 稳定一次, 连续达到 STABLE_COUNT 次才算真稳定
    """
    prev = None
    stable_cnt = 0
    deadline = time.perf_counter() + MAX_SETTLE_WAIT

    while True:
        frame = cam.grab()
        if frame is not None:
            patch = center_patch(frame)
            if prev is not None:
                mad = float(np.mean(np.abs(patch - prev)))
                if mad > DIFF_THRESHOLD:
                    stable_cnt = 0          # 画面还在动，重新计数
                else:
                    stable_cnt += 1         # 稳定采样 +1
                    if stable_cnt >= STABLE_COUNT:
                        return True         # 连续稳定多次 -> 到位
            prev = patch
        else:
            # grab 失败：当作没变化（避免死等）
            stable_cnt += 1
            if stable_cnt >= STABLE_COUNT:
                return True

        if time.perf_counter() >= deadline:
            return False                    # 超时安全阀

        time.sleep(CHANGE_POLL)


def main():
    model=YOLO(MODEL_PATH)
    sm_w=user32.GetSystemMetrics(0); sm_h=user32.GetSystemMetrics(1)
    cs=min(CAPTURE_SIZE,sm_w,sm_h); l=(sm_w-cs)//2; t=(sm_h-cs)//2
    cam=bettercam.create(output_color="BGR",region=(l,t,l+cs,t+cs))
    center=CAPTURE_SIZE/2.0

    paused=False; k_last=False

    print("启动：YOLO检测 + 鼠标一步移动 + 画面稳定后按下。按K暂停/继续。")

    try:
        while True:
            kk=k_down()
            if kk and not k_last:
                paused=not paused
                print("已暂停" if paused else "已继续")
            k_last=kk
            if paused:
                time.sleep(0.02); continue

            cycle_start=time.perf_counter()

            # 1) 截屏 + YOLO 检测
            frame=cam.grab()
            if frame is not None:
                result=model(frame,verbose=False)[0]
                box=None
                for b in result.boxes:
                    if int(b.cls.item())==TARGET_CLASS and b.conf.item()>=CONF:
                        box=b.xyxy[0].tolist(); break

                if box is not None:
                    x1,y1,x2,y2=box
                    tx=(x1+x2)/2; ty=(y1+y2)/2
                    du=tx-center; dv=ty-center
                    # ① 一步移动
                    send_move(int(MOVE_K*du), int(MOVE_K*dv))
                    # ② 移动后不断截取中心32×32判断画面是否变化，
                    #    稳定(不再变化)后立刻按下，不等待固定时间
                    if wait_until_settled(cam):
                        # ③ 画面稳定到位后才按下/松开
                        mouse_down()
                        time.sleep(CLICK_HOLD)
                        mouse_up()

            # 2) 补足到周期
            elapsed=time.perf_counter()-cycle_start
            if elapsed < CLICK_INTERVAL:
                time.sleep(CLICK_INTERVAL-elapsed)

    except KeyboardInterrupt:
        pass
    mouse_up(); cam.release()

if __name__=="__main__":
    main()