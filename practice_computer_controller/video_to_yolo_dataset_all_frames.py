from pathlib import Path
import cv2
from tqdm import tqdm


# ============================
# 参数区域
# ============================

# 多个视频
VIDEO_PATHS = [
    r"D:\PycharmProjects\computer_controller\video3.mp4",
]


SAVE_DIR = Path(
    "dataset_all_frames"
)


IMAGE_FORMAT = "jpg"

JPEG_QUALITY = 95

CREATE_EMPTY_LABEL = True



# ============================
# 数据集目录
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
# 获取当前最大编号
# 防止重新运行覆盖
# ============================

existing = list(IMAGE_DIR.glob("*.jpg"))

if len(existing) > 0:

    save_count = max(
        int(x.stem)
        for x in existing
    ) + 1

else:

    save_count = 0



# ============================
# 逐个视频处理
# ============================

for VIDEO_PATH in VIDEO_PATHS:


    cap = cv2.VideoCapture(
        VIDEO_PATH
    )


    if not cap.isOpened():

        print(
            "视频打开失败:",
            VIDEO_PATH
        )

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



    print("======================")
    print("当前视频:")
    print(VIDEO_PATH)
    print("======================")

    print(
        f"分辨率: {width} x {height}"
    )

    print(
        f"FPS: {fps}"
    )

    print(
        f"总帧数: {total_frames}"
    )



    for frame_id in tqdm(
        range(total_frames),
        desc="正在抽取帧"
    ):


        ret, frame = cap.read()


        if not ret:
            break



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



        cv2.imwrite(
            str(img_path),
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                JPEG_QUALITY
            ]
        )



        if CREATE_EMPTY_LABEL:

            label_path.touch()



        save_count += 1



    cap.release()



print()
print("======================")
print("全部完成")
print("======================")

print(
    f"图片总数量: {save_count}"
)

print(
    f"数据集位置: {SAVE_DIR}"
)

print("======================")