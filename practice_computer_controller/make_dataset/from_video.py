import cv2
import os


# ============================================================
# 配置
# ============================================================

VIDEO_PATH = r"D:\PycharmProjects\computer_controller\video3.mp4"

OUTPUT_DIR = r"/make_dataset/train/images"

# 每隔多少帧保存一张
INTERVAL = 5


# ============================================================
# 创建输出目录
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 打开视频
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(
        f"无法打开视频:\n{VIDEO_PATH}"
    )


# ============================================================
# 视频信息
# ============================================================

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)


print("=" * 70)
print("视频:", VIDEO_PATH)
print(f"FPS: {fps}")
print(f"总帧数: {total_frames}")
print(f"分辨率: {width} x {height}")
print(f"截取间隔: 每 {INTERVAL} 帧")
print("输出目录:", OUTPUT_DIR)
print("=" * 70)


# ============================================================
# 开始截取
# ============================================================

frame_index = 0
image_index = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # --------------------------------------------------------
    # 每隔 INTERVAL 帧保存
    # --------------------------------------------------------

    if frame_index % INTERVAL == 0:

        filename = f"{image_index:06d}.jpg"

        output_path = os.path.join(
            OUTPUT_DIR,
            filename
        )

        success = cv2.imwrite(
            output_path,
            frame,
            [
                cv2.IMWRITE_JPEG_QUALITY,
                95
            ]
        )

        if success:

            print(
                f"帧 {frame_index:6d} "
                f"-> {filename}"
            )

            image_index += 1

        else:

            print(
                f"[ERROR] 保存失败: {output_path}"
            )

    frame_index += 1


# ============================================================
# 释放
# ============================================================

cap.release()


# ============================================================
# 统计
# ============================================================

print()
print("=" * 70)
print("完成")
print("=" * 70)

print(
    f"视频总帧数: {frame_index}"
)

print(
    f"截取图片数: {image_index}"
)

print(
    f"输出目录: {OUTPUT_DIR}"
)

print("=" * 70)