"""
最小延迟视觉伺服 - MOVE后帧差确认响应 + 时间点计时

时间线(perf_counter, 秒):
  t_grab_start : 截屏开始
  t_grab_end   : 截屏完成
  t_yolo_start : YOLO开始
  t_yolo_end   : YOLO完成
  t_calc       : 坐标计算完成
  t_move       : send_move 完成
  t_diff_start : 帧差确认开始
  t_diff_end   : 帧差确认结束(已响应/超时)
  t_down       : mouse_down 完成
  t_up         : mouse_up 完成

关键指标:
  截屏_->_按下 = t_down - t_grab_start   (核心延迟)
  截屏_->_抬起 = t_up   - t_grab_start
"""
import ctypes
import time

import cv2
import bettercam
from ultralytics import YOLO

MODEL_PATH = r"D:\PycharmProjects\computer_controller\runs\18_model_aug\weights\last.engine"
CAPTURE_SIZE = 640
CONF = 0.35
TARGET_CLASS = 1
MOVE_K = 2.36

FRAME_DIFF_THRESHOLD = 3.0
FRAME_DIFF_TIMEOUT = 0.3
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

def target_center(model, frame):
    r = model(frame, verbose=False)[0]
    for b in r.boxes:
        if int(b.cls.item())==TARGET_CLASS and b.conf.item()>=CONF:
            x1,y1,x2,y2=b.xyxy[0].tolist()
            return ((x1+x2)/2,(y1+y2)/2)
    return None

def frame_diff_responded(cam, baseline, threshold, timeout):
    """返回(是否响应, 耗时秒)。帧差>阈值→已响应。"""
    t0 = time.perf_counter()
    responded = False
    while time.perf_counter() - t0 < timeout:
        f = cam.grab()
        if f is not None:
            m = float(cv2.absdiff(f, baseline).mean())
            if m > threshold:
                responded = True
                break
        time.sleep(0.003)
    return responded, time.perf_counter() - t0

def main():
    model = YOLO(MODEL_PATH)
    sm_w=user32.GetSystemMetrics(0); sm_h=user32.GetSystemMetrics(1)
    cs=min(CAPTURE_SIZE,sm_w,sm_h); l=(sm_w-cs)//2; t=(sm_h-cs)//2
    cam=bettercam.create(output_color="BGR",region=(l,t,l+cs,t+cs))
    center=CAPTURE_SIZE/2.0
    paused=False; k_last=False

    print("启动：帧差确认 + 时间点计时。按K暂停/继续。")

    try:
        while True:
            kk=k_down()
            if kk and not k_last:
                paused=not paused; print("已暂停" if paused else "已继续")
            k_last=kk
            if paused:
                time.sleep(0.02); continue

            # ===== 截屏(基准帧) =====
            t_grab_start = time.perf_counter()
            baseline = cam.grab()
            t_grab_end = time.perf_counter()

            if baseline is None:
                continue

            # ===== YOLO =====
            t_yolo_start = time.perf_counter()
            p = target_center(model, baseline)
            t_yolo_end = time.perf_counter()

            if p is None:
                continue    # 无目标，本周不做移动

            # ===== 坐标计算 =====
            tx,ty = p
            du=tx-center; dv=ty-center
            dx=int(MOVE_K*du); dy=int(MOVE_K*dv)
            t_calc = time.perf_counter()

            # ===== send_move =====
            send_move(dx,dy)
            t_move = time.perf_counter()

            # ===== 帧差确认 =====
            t_diff_start = time.perf_counter()
            responded, diff_dur = frame_diff_responded(cam, baseline, FRAME_DIFF_THRESHOLD, FRAME_DIFF_TIMEOUT)
            t_diff_end = time.perf_counter()

            # ===== mouse_down / up =====
            mouse_down()
            t_down = time.perf_counter()
            time.sleep(CLICK_HOLD)
            mouse_up()
            t_up = time.perf_counter()

            # ===== 打印时间点 =====
            grab_ms   = (t_grab_end - t_grab_start)*1000
            yolo_ms   = (t_yolo_end - t_yolo_start)*1000
            calc_ms   = (t_calc - t_yolo_end)*1000
            move_ms   = (t_move - t_calc)*1000
            diff_ms   = diff_dur*1000
            down_ms   = (t_down - t_move)*1000    # 从MOVE发出到按下(含帧差确认)
            to_down   = (t_down - t_grab_start)*1000   # ★ 截屏->按下 总延迟
            to_up     = (t_up - t_grab_start)*1000      # 截屏->抬起

            print(
                f"[时间] 截屏{grab_ms:.1f} YOLO{yolo_ms:.1f} 计算{calc_ms:.1f} "
                f"移动{move_ms:.1f} 帧差确认{diff_ms:.1f} 按下{down_ms:.1f} | "
                f"截屏->按下 {to_down:.1f}ms 截屏->抬起 {to_up:.1f}ms"
                f" (响应={responded})"
            )

            # ===== 补足周期 =====
            cycle_elapsed = time.perf_counter() - t_grab_start
            if cycle_elapsed < CLICK_INTERVAL:
                time.sleep(CLICK_INTERVAL - cycle_elapsed)

    except KeyboardInterrupt:
        pass
    mouse_up(); cam.release()

if __name__=="__main__":
    main()