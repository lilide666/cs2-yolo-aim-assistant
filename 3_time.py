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
        int(dx),
        int(dy),
        0,
        MOUSEEVENTF_MOVE,
        0,
        ctypes.pointer(extra)
    )

    inp=INPUT(0,mi)


    user32.SendInput(
        1,
        ctypes.byref(inp),
        ctypes.sizeof(inp)
    )

    return t0,ts()



def mouse_down():

    t0=ts()

    extra=ctypes.c_ulong(0)

    mi=MOUSEINPUT(
        0,
        0,
        0,
        MOUSEEVENTF_LEFTDOWN,
        0,
        ctypes.pointer(extra)
    )

    inp=INPUT(0,mi)


    user32.SendInput(
        1,
        ctypes.byref(inp),
        ctypes.sizeof(inp)
    )

    return t0,ts()



def mouse_up():

    extra=ctypes.c_ulong(0)

    mi=MOUSEINPUT(
        0,
        0,
        0,
        MOUSEEVENTF_LEFTUP,
        0,
        ctypes.pointer(extra)
    )

    inp=INPUT(0,mi)


    user32.SendInput(
        1,
        ctypes.byref(inp),
        ctypes.sizeof(inp)
    )



def delayed_mouse_up():

    time.sleep(CLICK_RELEASE_DELAY)

    mouse_up()



def k_down():

    return bool(
        user32.GetAsyncKeyState(K_VK)&0x8000
    )



def center_patch(frame,size=CHANGE_WINDOW):

    h,w=frame.shape[:2]

    half=size//2

    x=w//2-half
    y=h//2-half


    patch=frame[
        y:y+size,
        x:x+size
    ]


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


            mad=float(
                np.mean(
                    np.abs(
                        patch-prev
                    )
                )
            )


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


    paused=False
    k_last=False


    cycle=0
    total_time=0
    total=0


    print("启动：YOLO + 移动 + 画面变化触发点击")
    print("K暂停/继续")



    try:

        while True:


            kk=k_down()


            if kk and not k_last:

                paused=not paused

                print(
                    "暂停"
                    if paused
                    else
                    "继续"
                )


            k_last=kk



            if paused:

                time.sleep(0.02)

                continue



            cycle_start=ts()



            # 截图

            t0=ts()

            frame=cam.grab()

            t1=ts()



            box=None



            if frame is not None:


                # YOLO

                y0=ts()

                result=model(
                    frame,
                    verbose=False
                )[0]

                y1=ts()



                # 筛选目标

                f0=ts()


                for b in result.boxes:


                    cls=int(b.cls.item())

                    conf=float(b.conf.item())


                    if (
                        cls==TARGET_CLASS
                        and
                        conf>=CONF
                    ):

                        box=b.xyxy[0].tolist()

                        break


                f1=ts()



            else:

                y0=y1=f0=f1=ts()



            # ==========================
            # 无目标不输出
            # ==========================

            if box is None:

                continue



            # ==========================
            # 有目标开始记录
            # ==========================


            cycle+=1

            total+=1



            print("\n"+"="*55)

            print(
                f"Cycle {cycle}"
            )

            print("="*55)



            print("\n[检测阶段]")


            cost(
                "截图",
                t0,
                t1
            )


            cost(
                "YOLO推理",
                y0,
                y1
            )


            cost(
                "目标筛选",
                f0,
                f1
            )



            x1,y1,x2,y2=box



            # 坐标计算

            c0=ts()


            tx=(x1+x2)/2

            ty=(y1+y2)/2


            dx=tx-center

            dy=ty-center


            mx=int(MOVE_K*dx)

            my=int(MOVE_K*dy)


            c1=ts()



            print("\n[目标]")


            print(
                f"中心:({tx:.1f},{ty:.1f})"
            )


            print(
                f"偏差:({dx:.1f},{dy:.1f})"
            )


            print(
                f"移动:({mx},{my})"
            )


            cost(
                "坐标计算",
                c0,
                c1
            )



            # 移动

            m0,m1=send_move(
                mx,
                my
            )


            cost(
                "SendInput移动",
                m0,
                m1
            )



            # 等待变化

            print("\n[变化检测]")


            motion=wait_until_motion(cam)



            print(
                "变化结果:",
                motion["success"]
            )


            print(
                "MAD:",
                motion["mad"]
            )


            print(
                "比较次数:",
                motion["compare"]
            )


            print(
                f"等待时间:"
                f"{motion['time']:.3f} ms"
            )



            # 点击

            d0,d1=mouse_down()


            print("\n[点击]")


            cost(
                "mouse_down",
                d0,
                d1
            )



            threading.Thread(
                target=delayed_mouse_up,
                daemon=True
            ).start()



            # 周期统计

            end=ts()

            elapsed=end-cycle_start


            total_time+=elapsed



            print("\n[周期]")


            print(
                f"总耗时:"
                f"{elapsed:.3f} ms"
            )


            remain=(
                CLICK_INTERVAL*1000
                -
                elapsed
            )


            print(
                f"剩余等待:"
                f"{max(remain,0):.3f} ms"
            )



            if remain>0:

                time.sleep(
                    remain/1000
                )



    except KeyboardInterrupt:

        print("退出")



    finally:

        mouse_up()

        cam.release()



        if total:

            avg=total_time/total


            print("\n==========平均==========")


            print(
                f"平均周期:"
                f"{avg:.3f} ms"
            )


            print(
                f"平均FPS:"
                f"{1000/avg:.3f}"
            )



if __name__=="__main__":

    main()