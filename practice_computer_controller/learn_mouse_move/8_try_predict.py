"""
目标运动数据采集 + 可视化
记录 每帧YOLO目标中心(px) + 时间戳, 保存CSV, 画出轨迹/速度曲线。
用真实数据判断: 目标运动是否有规律(能否预测)、噪音多大。
"""
import ctypes
import time
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
import bettercam
from ultralytics import YOLO

MODEL_PATH = r"D:\PycharmProjects\computer_controller\runs\18_model_aug\weights\last.engine"
CAPTURE_SIZE = 640
CONF = 0.35
TARGET_CLASS = 1
RECORD_SECONDS = 5.0       # 记录时长
SAVE_CSV = "motion_data.csv"
K_VK = 0x4B

user32 = ctypes.windll.user32
def k_down(): return bool(user32.GetAsyncKeyState(K_VK)&0x8000)

def target_center(model, frame):
    r = model(frame, verbose=False)[0]
    for b in r.boxes:
        if int(b.cls.item())==TARGET_CLASS and b.conf.item()>=CONF:
            x1,y1,x2,y2=b.xyxy[0].tolist()
            return ((x1+x2)/2,(y1+y2)/2)
    return None

def main():
    model=YOLO(MODEL_PATH)
    sm_w=user32.GetSystemMetrics(0); sm_h=user32.GetSystemMetrics(1)
    cs=min(CAPTURE_SIZE,sm_w,sm_h); l=(sm_w-cs)//2; t=(sm_h-cs)//2
    cam=bettercam.create(output_color="BGR",region=(l,t,l+cs,t+cs))

    print("采集目标运动数据:")
    print("  - 让目标【按你平时那样运动】(会变向、有加速度)")
    print("  - 按 K 开始记录, 再按 K 停止")
    print("  开始前请准备好,目标保持在画面内。")
    n=0
    while True:
        if k_down():
            n+=1; time.sleep(0.8)
            if n%2==1: print(">>> 开始记录...")
            else: break
        time.sleep(0.02)

    data=[]   # (time_s, x, y)
    t0=time.perf_counter()
    while True:
        frame=cam.grab()
        if frame is not None:
            p=target_center(model,frame)
            if p is not None:
                t=time.perf_counter()-t0
                data.append((t,p[0],p[1]))
        if k_down():
            break
        time.sleep(0.001)   # 尽量高频

    print(f"记录结束，收集 {len(data)} 个样本，时长 {data[-1][0] if data else 0:.1f}s")

    # 保存CSV
    with open(SAVE_CSV,"w",newline="") as f:
        w=csv.writer(f); w.writerow(["t_s","x_px","y_px"])
        w.writerows(data)
    print(f"已保存: {SAVE_CSV}")

    # 画图
    ts=np.array([d[0] for d in data]); xs=np.array([d[1] for d in data]); ys=np.array([d[2] for d in data])
    # 速度(帧间差分)
    vx=np.diff(xs)/np.diff(ts); vy=np.diff(ys)/np.diff(ts)
    spd=np.sqrt(vx**2+vy**2)

    fig,axes=plt.subplots(2,2,figsize=(12,8))
    axes[0][0].plot(xs,ys); axes[0][0].set_title("轨迹 (x vs y)"); axes[0][0].set_aspect('equal')
    axes[1][0].plot(ts,xs,'r',label='x'); axes[1][0].plot(ts,ys,'b',label='y'); axes[1][0].legend(); axes[1][0].set_title("位置 vs 时间")
    axes[0][1].plot(ts[1:],vx,'r',label='vx'); axes[0][1].plot(ts[1:],vy,'b',label='vy'); axes[0][1].legend(); axes[0][1].set_title("速度 vs 时间")
    axes[1][1].plot(ts[1:],spd); axes[1][1].set_title("合速度 |v|")
    plt.tight_layout(); plt.show()

    cam.release()

if __name__=="__main__":
    main()