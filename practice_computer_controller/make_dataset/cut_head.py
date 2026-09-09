import os
import cv2
import json


# ============================================================
# 配置
# ============================================================

# 原始 person 数据集
SOURCE_DIR = r"D:\PycharmProjects\computer_controller\make_dataset\train_body"

IMAGE_DIR = os.path.join(
    SOURCE_DIR,
    "images"
)

LABEL_DIR = os.path.join(
    SOURCE_DIR,
    "labels"
)


# ============================================================
# 输出头部标注数据
# ============================================================

OUTPUT_DIR = r"D:\PycharmProjects\computer_controller\make_dataset\head_annotation"

OUTPUT_IMAGE_DIR = os.path.join(
    OUTPUT_DIR,
    "images"
)

MAPPING_PATH = os.path.join(
    OUTPUT_DIR,
    "mapping.json"
)


# ============================================================
# 创建目录
# ============================================================

os.makedirs(
    OUTPUT_IMAGE_DIR,
    exist_ok=True
)


# ============================================================
# 图片格式
# ============================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# person 类别
# ============================================================

PERSON_CLASS_ID = 0


# ============================================================
# 读取 YOLO 标签
# ============================================================

def read_yolo_label(label_path):

    boxes = []

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:

        lines = f.readlines()

    for line_index, line in enumerate(lines):

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            print(
                f"警告：标签格式错误："
                f"{label_path}"
                f" 第 {line_index + 1} 行"
            )
            continue

        class_id = int(parts[0])

        x_center = float(parts[1])
        y_center = float(parts[2])
        width = float(parts[3])
        height = float(parts[4])

        boxes.append({
            "class_id": class_id,
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height
        })

    return boxes


# ============================================================
# YOLO → 像素坐标
# ============================================================

def yolo_to_pixel(
        x_center,
        y_center,
        width,
        height,
        image_width,
        image_height
):

    x1 = int(
        (x_center - width / 2)
        * image_width
    )

    y1 = int(
        (y_center - height / 2)
        * image_height
    )

    x2 = int(
        (x_center + width / 2)
        * image_width
    )

    y2 = int(
        (y_center + height / 2)
        * image_height
    )

    return x1, y1, x2, y2


# ============================================================
# 获取图片
# ============================================================

image_files = [
    f
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(IMAGE_EXTENSIONS)
]

image_files.sort()


# ============================================================
# mapping
# ============================================================

mapping = {}


# ============================================================
# 统计
# ============================================================

total_images = 0
total_persons = 0
failed_images = 0


# ============================================================
# 开始处理
# ============================================================

print("=" * 80)
print("根据 person YOLO 标注裁剪人物")
print("=" * 80)

print(
    f"原始图片目录：{IMAGE_DIR}"
)

print(
    f"原始标签目录：{LABEL_DIR}"
)

print(
    f"输出目录：    {OUTPUT_IMAGE_DIR}"
)

print()


# ============================================================
# 遍历图片
# ============================================================

for image_name in image_files:

    image_path = os.path.join(
        IMAGE_DIR,
        image_name
    )

    stem = os.path.splitext(
        image_name
    )[0]

    label_path = os.path.join(
        LABEL_DIR,
        stem + ".txt"
    )

    # --------------------------------------------------------
    # 标签不存在
    # --------------------------------------------------------

    if not os.path.exists(label_path):

        print(
            f"[跳过] 没有标签：{image_name}"
        )

        continue

    # --------------------------------------------------------
    # 读取图片
    # --------------------------------------------------------

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            f"[失败] 无法读取图片：{image_name}"
        )

        failed_images += 1

        continue

    image_height, image_width = image.shape[:2]

    # --------------------------------------------------------
    # 读取标签
    # --------------------------------------------------------

    boxes = read_yolo_label(
        label_path
    )

    person_index = 0

    # --------------------------------------------------------
    # 遍历 person
    # --------------------------------------------------------

    for box in boxes:

        if box["class_id"] != PERSON_CLASS_ID:
            continue

        # ----------------------------------------------------
        # YOLO → 像素坐标
        # ----------------------------------------------------

        x1, y1, x2, y2 = yolo_to_pixel(
            box["x_center"],
            box["y_center"],
            box["width"],
            box["height"],
            image_width,
            image_height
        )

        # ----------------------------------------------------
        # 防止越界
        # ----------------------------------------------------

        x1 = max(
            0,
            min(
                x1,
                image_width - 1
            )
        )

        y1 = max(
            0,
            min(
                y1,
                image_height - 1
            )
        )

        x2 = max(
            1,
            min(
                x2,
                image_width
            )
        )

        y2 = max(
            1,
            min(
                y2,
                image_height
            )
        )

        # ----------------------------------------------------
        # 检查框
        # ----------------------------------------------------

        if x2 <= x1 or y2 <= y1:

            print(
                f"[跳过] 无效 person 框："
                f"{image_name}"
                f" person{person_index}"
            )

            person_index += 1

            continue

        # ----------------------------------------------------
        # 裁剪
        #
        # 注意：
        # 不 padding
        # 完全按照 person bbox 裁剪
        # ----------------------------------------------------

        crop = image[
            y1:y2,
            x1:x2
        ]

        # ----------------------------------------------------
        # 输出文件名
        # ----------------------------------------------------

        crop_name = (
            f"{stem}"
            f"_person{person_index}"
            f".jpg"
        )

        crop_path = os.path.join(
            OUTPUT_IMAGE_DIR,
            crop_name
        )

        # ----------------------------------------------------
        # 保存 crop
        # ----------------------------------------------------

        success = cv2.imwrite(
            crop_path,
            crop
        )

        if not success:

            print(
                f"[失败] 保存失败："
                f"{crop_name}"
            )

            failed_images += 1

            person_index += 1

            continue

        # ----------------------------------------------------
        # 保存映射关系
        # ----------------------------------------------------

        mapping[crop_name] = {

            "original_image": image_name,

            "original_width": image_width,

            "original_height": image_height,

            "person_index": person_index,

            "person_bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            },

            "crop_width": x2 - x1,

            "crop_height": y2 - y1
        }

        print(
            f"[OK] "
            f"{crop_name}"
            f"  "
            f"crop={x2 - x1}x{y2 - y1}"
        )

        total_persons += 1

        person_index += 1

    total_images += 1


# ============================================================
# 保存 mapping.json
# ============================================================

with open(
    MAPPING_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        mapping,
        f,
        ensure_ascii=False,
        indent=4
    )


# ============================================================
# 完成
# ============================================================

print()
print("=" * 80)
print("裁剪完成")
print("=" * 80)

print(
    f"处理图片：      {total_images}"
)

print(
    f"裁剪 person：   {total_persons}"
)

print(
    f"失败数量：      {failed_images}"
)

print()

print(
    f"Crop 图片目录："
)

print(
    f"  {OUTPUT_IMAGE_DIR}"
)

print()

print(
    f"Mapping："
)

print(
    f"  {MAPPING_PATH}"
)

print("=" * 80)