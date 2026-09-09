"""
最小延迟视觉伺服 - YOLO 检测 + 鼠标一步移动 + 移动稳定后按下

循环（每 0.25s）：
  1. 截取中心640
  2. YOLO检测目标(取任一)
  3. 一步移动: dx=+2.36*(x-320), dy=+2.36*(y-320)
  4. 等 MOVE_SETTLE 稳定
  5. 按下/松开鼠标
  6. 补足到 0.25s -> 下轮
按 K 暂停/继续
"""
import ctypes
import time

import bettercam
from ultralytics import YOLO

MODEL_PATH = r"D:\PycharmProjects\computer_controller\runs\18_model_aug\weights\last.engine"
CAPTURE_SIZE = 640
CONF = 0.35
TARGET_CLASS = 1
MOVE_K = 2.36
MOVE_SETTLE = 0.05       # 移动后、按下前稳定延迟(秒)
CLICK_HOLD = 0.02
CLICK_INTERVAL = 0.25
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

def main():
    model=YOLO(MODEL_PATH)
    sm_w=user32.GetSystemMetrics(0); sm_h=user32.GetSystemMetrics(1)
    cs=min(CAPTURE_SIZE,sm_w,sm_h); l=(sm_w-cs)//2; t=(sm_h-cs)//2
    cam=bettercam.create(output_color="BGR",region=(l,t,l+cs,t+cs))
    center=CAPTURE_SIZE/2.0

    paused=False; k_last=False

    print("启动：YOLO检测 + 鼠标一步移动 + 移动稳定后按下。按K暂停/继续。")

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
                    # ② 等移动稳定到位
                    time.sleep(MOVE_SETTLE)
                    # ③ 到位后才按下/松开
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