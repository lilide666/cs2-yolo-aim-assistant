import bettercam
import cv2
from pathlib import Path
import time


# 保存目录
SAVE_DIR = Path(
    r"D:\cs2_all_frames"
)

SAVE_DIR.mkdir(
    exist_ok=True
)


# 创建捕获
camera = bettercam.create(
    output_idx=0,
    output_color="BGR"
)


print("开始捕获...")


# 开启后台采集
camera.start(
    target_fps=60
)


count = 0


while True:

    frame = camera.get_latest_frame()


    if frame is None:
        continue


    # 保存每一帧
    filename = SAVE_DIR / (
        f"{count:08d}.jpg"
    )


    cv2.imwrite(
        str(filename),
        frame,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            95
        ]
    )


    print(
        "保存:",
        count,
        frame.shape
    )


    count += 1