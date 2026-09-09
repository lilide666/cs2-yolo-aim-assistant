import ctypes
import time
import bettercam
import numpy as np


# ======================
# 参数
# ======================

MOVE_X = 1000
MOVE_Y = 1000

CAPTURE_SIZE = 640

TEST_COUNT = 100

DIFF_THRESHOLD = 5



def ts():
    return time.perf_counter()*1000



# ======================
# SendInput
# ======================

user32 = ctypes.windll.user32

MOUSEEVENTF_MOVE = 0x0001


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

    extra=ctypes.c_ulong(0)

    mi=MOUSEINPUT(
        int(dx),
        int(dy),
        0,
        MOUSEEVENTF_MOVE,
        0,
        ctypes.pointer(extra)
    )

    inp=INPUT(
        0,
        mi
    )

    user32.SendInput(
        1,
        ctypes.byref(inp),
        ctypes.sizeof(inp)
    )



# ======================
# 测试
# ======================

def main():

    sw=user32.GetSystemMetrics(0)
    sh=user32.GetSystemMetrics(1)


    size=CAPTURE_SIZE


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


    print("3秒后开始")

    time.sleep(3)



    results=[]


    for i in range(TEST_COUNT):


        # 移动前画面

        old=cam.grab()

        if old is None:
            continue


        old=old.astype(np.float32)



        # 发送移动

        send_move(
            MOVE_X,
            MOVE_Y
        )


        t0=ts()

        count=0



        while True:


            frame=cam.grab()


            if frame is None:
                continue



            diff=np.mean(
                np.abs(
                    frame.astype(np.float32)
                    -
                    old
                )
            )


            count+=1



            if diff > DIFF_THRESHOLD:


                delay=ts()-t0

                results.append(delay)


                print(
                    f"{i+1:03d}  "
                    f"{delay:8.3f} ms  "
                    f"截图:{count}  "
                    f"MAD:{diff:.2f}"
                )

                break



    cam.release()



    print("\n==========统计==========")

    if results:

        print(
            f"次数:{len(results)}"
        )

        print(
            f"平均:"
            f"{np.mean(results):.3f} ms"
        )

        print(
            f"最小:"
            f"{np.min(results):.3f} ms"
        )

        print(
            f"最大:"
            f"{np.max(results):.3f} ms"
        )

        print(
            f"标准差:"
            f"{np.std(results):.3f} ms"
        )



if __name__=="__main__":
    main()