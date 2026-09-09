import ctypes
import time
import threading
import bettercam
import numpy as np
from ultralytics import YOLO


MODEL_PATH=r"D:\PycharmProjects\aim\18_model_aug\weights\best.engine"

CAPTURE_SIZE=640
CONF=0.35
TARGET_CLASS=1
MOVE_K=2.36

CHANGE_WINDOW=32
CHANGE_POLL=0.005
DIFF_THRESHOLD=10
MAX_MOTION_WAIT=0.15

CLICK_RELEASE_DELAY=0.005
CLICK_INTERVAL=0.3

K_VK=0x4B

# 移动线程的读取/执行周期(秒)
MOVE_INTERVAL=0.3


def ts():
    return time.perf_counter()*1000


def cost(name,start,end):
    print(f"{name:<22}{end-start:8.3f} ms")


user32=ctypes.windll.user32


MOUSEEVENTF_MOVE=0x0001
MOUSEEVENTF_LEFTDOWN=0x0002
MOUSEEVENTF_LEFTUP=0x0004


class MOUSEINPUT(ctypes.Structure):
    _fields_=[
        ("dx",ctypes.c_long),
        ("dy",ctypes.c_long),
        ("mouseData",ctypes.c_ulong),
        ("dwFlags",ctypes.c_ulong),
        ("time",ctypes.c_ulong),
        ("dwExtraInfo",ctypes.POINTER(ctypes.c_ulong))
    ]


class INPUT(ctypes.Structure):
    _fields_=[
        ("type",ctypes.c_ulong),
        ("mi",MOUSEINPUT)
    ]


def send_move(dx,dy):
    t0=ts()
    extra=ctypes.c_ulong(0)
    mi=MOUSEINPUT(
        int(dx),int(dy),0,MOUSEEVENTF_MOVE,0,ctypes.pointer(extra)
    )
    inp=INPUT(0,mi)
    user32.SendInput(1,ctypes.byref(inp),ctypes.sizeof(inp))
    return t0,ts()


def mouse_down():
    t0=ts()
    extra=ctypes.c_ulong(0)
    mi=MOUSEINPUT(0,0,0,MOUSEEVENTF_LEFTDOWN,0,ctypes.pointer(extra))
    inp=INPUT(0,mi)
    user32.SendInput(1,ctypes.byref(inp),ctypes.sizeof(inp))
    return t0,ts()


def mouse_up():
    extra=ctypes.c_ulong(0)
    mi=MOUSEINPUT(0,0,0,MOUSEEVENTF_LEFTUP,0,ctypes.pointer(extra))
    inp=INPUT(0,mi)
    user32.SendInput(1,ctypes.byref(inp),ctypes.sizeof(inp))


def delayed_mouse_up():
    time.sleep(CLICK_RELEASE_DELAY)
    mouse_up()


def k_down():
    return bool(user32.GetAsyncKeyState(K_VK)&0x8000)


def center_patch(frame,size=CHANGE_WINDOW):
    h,w=frame.shape[:2]
    half=size//2
    x=w//2-half
    y=h//2-half
    patch=frame[y:y+size,x:x+size]
    if patch.ndim==3:
        patch=patch[:,:,0]
    return patch.astype(np.float32)


def wait_until_motion(cam):
    start=ts()
    prev=None
    count=0
    MAX_COMPARE=10
    while count<MAX_COMPARE:
        frame=cam.grab()
        if frame is None:
            continue
        patch=center_patch(frame)
        if prev is not None:
            mad=float(np.mean(np.abs(patch-prev)))
            count+=1
            if mad>DIFF_THRESHOLD:
                return {
                    "success":True,
                    "time":ts()-start,
                    "mad":mad,
                    "compare":count
                }
        prev=patch
    return {
        "success":False,
        "time":ts()-start,
        "mad":mad if prev is not None else 0,
        "compare":count
    }


# ============================================================
# 共享目标缓冲区：检测线程写，移动线程读
# ============================================================
class TargetBuffer:
    def __init__(self):
        # (tx, ty, dx, dy, mx, my, fresh)
        self._lock=threading.Lock()
        self._tx=0.0
        self._ty=0.0
        self._dx=0.0
        self._dy=0.0
        self._mx=0
        self._my=0
        self._fresh=False   # 是否有有效目标
        self._count=0

    def update(self,tx,ty,center):
        with self._lock:
            self._tx=tx
            self._ty=ty
            dx=tx-center
            dy=ty-center
            self._dx=dx
            self._dy=dy
            self._mx=int(MOVE_K*dx)
            self._my=int(MOVE_K*dy)
            self._fresh=True
            self._count+=1

    def clear(self):
        with self._lock:
            self._fresh=False

    def get(self):
        with self._lock:
            return (
                self._tx,
                self._ty,
                self._dx,
                self._dy,
                self._mx,
                self._my,
                self._fresh,
                self._count
            )


# ============================================================
# 线程1：一直截屏 + YOLO，持续更新目标位置
# ============================================================
def detect_thread(cam, model, center, buf, stop_event):
    prev=None
    last_box=None
    cycle=0

    while not stop_event.is_set():
        kk=k_down()
        frame=cam.grab()
        if frame is None:
            continue

        result=model(frame, verbose=False)[0]

        box=None
        for b in result.boxes:
            cls=int(b.cls.item())
            conf=float(b.conf.item())
            if cls==TARGET_CLASS and conf>=CONF:
                box=b.xyxy[0].tolist()
                break

        if box is None:
            # 用上次的目标做平滑，若一段时间没有目标则清空
            buf.clear()
            prev=None
            last_box=None
            continue

        x1,y1,x2,y2=box
        tx=(x1+x2)/2
        ty=(y1+y2)/2

        buf.update(tx,ty,center)


# ============================================================
# 线程2：每0.3s读取目标位置并移动
# ============================================================
def move_thread(buf, stop_event):
    cycle=0
    total_time=0
    total=0

    while not stop_event.is_set():
        cycle_start=ts()

        tx,ty,dx,dy,mx,my,fresh,count=buf.get()

        if fresh:
            # 移动
            m0,m1=send_move(mx,my)

            # 点击
            d0,d1=mouse_down()
            threading.Thread(target=delayed_mouse_up, daemon=True).start()

            cycle+=1
            total+=1

            # 周期统计
            end=ts()
            elapsed=end-cycle_start
            total_time+=elapsed

            print("\n"+"="*55)
            print(f"Cycle {cycle}")
            print("="*55)

            print("[目标]")
            print(f"中心:({tx:.1f},{ty:.1f})")
            print(f"偏差:({dx:.1f},{dy:.1f})")
            print(f"移动:({mx},{my})")

            cost("SendInput移动",m0,m1)
            cost("mouse_down",d0,d1)

            print("[周期]")
            print(f"总耗时:{elapsed:.3f} ms")

        # 等待到下一个周期
        wait=MOVE_INTERVAL*1000-(ts()-cycle_start)
        if wait>0:
            stop_event.wait(wait/1000)

    # 统计
    if total:
        avg=total_time/total
        print("\n==========平均==========")
        print(f"平均周期:{avg:.3f} ms")
        print(f"平均FPS:{1000/avg:.3f}")


def main():
    model=YOLO(MODEL_PATH)

    sw=user32.GetSystemMetrics(0)
    sh=user32.GetSystemMetrics(1)
    cs=min(CAPTURE_SIZE,sw,sh)
    l=(sw-cs)//2
    t=(sh-cs)//2

    cam=bettercam.create(
        output_color="BGR",
        region=(l,t,l+cs,t+cs)
    )

    center=CAPTURE_SIZE/2

    print("启动：双线程（检测线程 + 每0.3s移动线程）")
    print("K暂停/继续")

    buf=TargetBuffer()
    stop_event=threading.Event()

    # 注意：bettercam 对象在多线程下使用需要小心，
    # 这里检测线程独占使用 cam，移动线程只做鼠标操作

    d_thread=threading.Thread(
        target=detect_thread,
        args=(cam, model, center, buf, stop_event),
        daemon=True
    )

    m_thread=threading.Thread(
        target=move_thread,
        args=(buf, stop_event),
        daemon=True
    )

    d_thread.start()
    m_thread.start()

    # 主线程：处理暂停/退出控制
    paused=False
    k_last=False

    try:
        while True:
            kk=k_down()
            if kk and not k_last:
                paused=not paused
                print("暂停" if paused else "继续")
                # 暂停时清掉目标，避免继续后位置残留
                if paused:
                    buf.clear()
            k_last=kk
            time.sleep(0.02)
    except KeyboardInterrupt:
        print("退出")
    finally:
        stop_event.set()
        mouse_up()
        cam.release()


if __name__=="__main__":
    main()