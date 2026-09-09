"""
最小延迟视觉伺服
YOLO + 一步移动 + 固定等待点击
"""

import ctypes
import time
import threading
import bettercam
from ultralytics import YOLO


MODEL_PATH = r"D:\PycharmProjects\aim\head_body_640_yolo26n_screen_640\weights\best.engine"


CAPTURE_SIZE = 640

CONF = 0.35
TARGET_CLASS = 1

MOVE_K = 2.36

# 移动后等待(ms)
# 0 = 立即点击
WAIT_AFTER_MOVE = 22


CLICK_RELEASE_DELAY = 0.005
CLICK_INTERVAL = 0.3


K_VK = 0x4B



def ts():
    return time.perf_counter()*1000



user32 = ctypes.windll.user32


MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004



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




def send_input(flag,dx=0,dy=0):

    extra=ctypes.c_ulong(0)


    mi=MOUSEINPUT(
        int(dx),
        int(dy),
        0,
        flag,
        0,
        ctypes.pointer(extra)
    )


    inp=INPUT(0,mi)


    user32.SendInput(
        1,
        ctypes.byref(inp),
        ctypes.sizeof(inp)
    )




def send_move(dx,dy):

    t0=ts()

    send_input(
        MOUSEEVENTF_MOVE,
        dx,
        dy
    )

    return t0,ts()




def mouse_down():

    t0=ts()

    send_input(
        MOUSEEVENTF_LEFTDOWN
    )

    return t0,ts()




def mouse_up():

    send_input(
        MOUSEEVENTF_LEFTUP
    )




def release_thread():

    time.sleep(
        CLICK_RELEASE_DELAY
    )

    mouse_up()




def k_down():

    return bool(
        user32.GetAsyncKeyState(K_VK)
        &
        0x8000
    )




def main():

    model=YOLO(MODEL_PATH)



    sw=user32.GetSystemMetrics(0)
    sh=user32.GetSystemMetrics(1)



    size=min(
        CAPTURE_SIZE,
        sw,
        sh
    )


    left=(sw-size)//2
    top=(sh-size)//2



    cam=bettercam.create(

        output_color="BGR",

        region=(

            left,
            top,

            left+size,
            top+size

        )
    )



    center=size/2



    cycle=0

    paused=False
    last_k=False



    print("启动")
    print("K暂停/继续")



    try:


        while True:



            k=k_down()



            if k and not last_k:

                paused=not paused

                print(
                    "暂停"
                    if paused
                    else
                    "继续"
                )



            last_k=k



            if paused:

                time.sleep(0.02)

                continue




            cycle+=1



            frame=cam.grab()



            box=None



            if frame is not None:



                result=model(
                    frame,
                    verbose=False
                )[0]



                for b in result.boxes:


                    cls=int(
                        b.cls.item()
                    )


                    conf=float(
                        b.conf.item()
                    )


                    if (

                        cls==TARGET_CLASS

                        and

                        conf>=CONF

                    ):


                        box=b.xyxy[0].tolist()

                        break




            # ======================
            # 有目标才执行和输出
            # ======================

            if box:


                x1,y1,x2,y2=box



                tx=(x1+x2)/2
                ty=(y1+y2)/2



                dx=tx-center
                dy=ty-center



                mx=int(
                    dx*MOVE_K
                )


                my=int(
                    dy*MOVE_K
                )



                print("\n"+"="*40)

                print(
                    f"Cycle {cycle}"
                )


                print(
                    f"目标中心:"
                    f"({tx:.1f},{ty:.1f})"
                )


                print(
                    f"移动:"
                    f"({mx},{my})"
                )



                a,b=send_move(
                    mx,
                    my
                )



                print(
                    f"移动耗时:"
                    f"{b-a:.3f} ms"
                )



                if WAIT_AFTER_MOVE>0:

                    time.sleep(
                        WAIT_AFTER_MOVE/1000
                    )



                a,b=mouse_down()



                print(
                    f"点击耗时:"
                    f"{b-a:.3f} ms"
                )



                threading.Thread(

                    target=release_thread,

                    daemon=True

                ).start()



            # 无目标不输出



            time.sleep(
                max(
                    0,
                    CLICK_INTERVAL
                    -
                    0
                )
            )




    except KeyboardInterrupt:

        print("退出")



    finally:

        mouse_up()

        cam.release()




if __name__=="__main__":

    main()