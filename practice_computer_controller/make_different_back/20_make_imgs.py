from pathlib import Path
import cv2
from tqdm import tqdm


# ============================
# 参数区域
# ============================

# 多个视频路径
VIDEO_PATHS = [

    r"D:\PycharmProjects\computer_controller\make_different_back\19_background_img\back.mp4",

]


# 输出数据集

SAVE_DIR = Path(
    r"D:\PycharmProjects\computer_controller\make_different_back\19_background_img\images"
)


# 图片格式

IMAGE_FORMAT = "jpg"


# 每隔多少帧保存一次

FRAME_INTERVAL = 10


# jpg质量

JPEG_QUALITY = 100



# ============================
# 文件夹
# ============================

IMAGE_DIR = SAVE_DIR / "images"

LABEL_DIR = SAVE_DIR / "labels"


IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LABEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)



# ============================
# 批量处理
# ============================


save_count = 0



for video_index, VIDEO_PATH in enumerate(VIDEO_PATHS):


    print()
    print("======================")
    print(f"开始处理第 {video_index+1} 个视频")
    print(VIDEO_PATH)
    print("======================")


    cap = cv2.VideoCapture(
        VIDEO_PATH
    )


    if not cap.isOpened():

        print("视频打开失败，跳过")
        continue



    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )


    fps = cap.get(
        cv2.CAP_PROP_FPS
    )


    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )


    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )


    print(
        f"分辨率: {width}x{height}"
    )

    print(
        f"FPS: {fps}"
    )

    print(
        f"总帧数: {total_frames}"
    )



    # 当前视频帧编号

    frame_id = 0



    for _ in tqdm(
        range(total_frames)
    ):


        ret, frame = cap.read()


        if not ret:
            break



        # 跳帧

        if frame_id % FRAME_INTERVAL != 0:

            frame_id += 1

            continue



        filename = (
            f"{save_count:06d}"
            f".{IMAGE_FORMAT}"
        )



        img_path = (
            IMAGE_DIR /
            filename
        )


        label_path = (
            LABEL_DIR /
            filename.replace(
                IMAGE_FORMAT,
                "txt"
            )
        )



        # 保存图片

        cv2.imwrite(
            str(img_path),
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                JPEG_QUALITY
            ]
        )


        # 创建空标签

        label_path.touch()



        save_count += 1


        frame_id += 1



    cap.release()



print()

print("======================")
print("全部视频处理完成")
print(
    f"总保存图片数量: {save_count}"
)

print(
    f"保存位置: {SAVE_DIR}"
)

print("======================")