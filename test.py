import time
import ctypes
import bettercam
import numpy as np
import gc


def ts():
    return time.perf_counter()*1000


def test(cam,name,n=100):

    times=[]

    for _ in range(10):
        cam.grab()


    for _ in range(n):

        t0=ts()

        frame=cam.grab()

        t1=ts()


        if frame is not None:
            times.append(t1-t0)


    print("\n",name)

    if len(times)==0:
        print("没有获取到帧")
        return


    print(
        f"平均:{np.mean(times):.3f} ms"
    )

    print(
        f"最小:{np.min(times):.3f} ms"
    )

    print(
        f"最大:{np.max(times):.3f} ms"
    )



# 删除旧实例
gc.collect()


user32=ctypes.windll.user32

w=user32.GetSystemMetrics(0)
h=user32.GetSystemMetrics(1)

print("屏幕:",w,h)



# 640中心

size=640

x=(w-size)//2
y=(h-size)//2


cam=bettercam.create(
    device_idx=0,
    output_idx=0,
    output_color="BGR",
    region=(
        x,
        y,
        x+size,
        y+size
    )
)


test(
    cam,
    "640x640"
)



del cam
gc.collect()



# 全屏

cam=bettercam.create(
    device_idx=0,
    output_idx=0,
    output_color="BGR",
    region=(
        0,
        0,
        w,
        h
    )
)


test(
    cam,
    "2560x1600 Full"
)


cam.release()