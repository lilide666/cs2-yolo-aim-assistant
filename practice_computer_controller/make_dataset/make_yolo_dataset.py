import os
import shutil
import random


# ============================================================
# 配置
# ============================================================

# 原始数据
SOURCE_DIR = r"D:\PycharmProjects\computer_controller\make_dataset\different_back"

IMAGE_DIR = os.path.join(
    SOURCE_DIR,
    "images"
)

LABEL_DIR = os.path.join(
    SOURCE_DIR,
    "labels"
)


# ============================================================
# 输出数据集
#
# body_head_dataset/
#
# ├── images/
# │   ├── train/
# │   └── val/
# │
# ├── labels/
# │   ├── train/
# │   └── val/
# │
# └── data.yaml
# ============================================================

OUTPUT_DIR = r"D:\PycharmProjects\computer_controller\make_dataset\different_back_dataset"


# ============================================================
# Train
# ============================================================

TRAIN_IMAGE_DIR = os.path.join(
    OUTPUT_DIR,
    "images",

    "train"
)

TRAIN_LABEL_DIR = os.path.join(
    OUTPUT_DIR,
    "labels",
    "train"
)


# ============================================================
# Val
# ============================================================

VAL_IMAGE_DIR = os.path.join(
    OUTPUT_DIR,
    "images",
    "val"
)

VAL_LABEL_DIR = os.path.join(
    OUTPUT_DIR,
    "labels",
    "val"
)


# ============================================================
# 划分比例
# ============================================================

TRAIN_RATIO = 0.8
VAL_RATIO = 0.2


# ============================================================
# 选择前多少张图片
#
# None = 使用全部图片
# ============================================================

MAX_IMAGES = None


# ============================================================
# 随机种子
# ============================================================

RANDOM_SEED = 42


# ============================================================
# 类别
#
# 必须和 YOLO 标签中的 class_id 对应
#
# 0 = body
# 1 = head
# ============================================================

CLASS_NAMES = [
    "body",
    "head"
]


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
# 创建目录
# ============================================================

os.makedirs(
    TRAIN_IMAGE_DIR,
    exist_ok=True
)

os.makedirs(
    TRAIN_LABEL_DIR,
    exist_ok=True
)

os.makedirs(
    VAL_IMAGE_DIR,
    exist_ok=True
)

os.makedirs(
    VAL_LABEL_DIR,
    exist_ok=True
)


# ============================================================
# 获取图片
# ============================================================

image_files = [
    f
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(IMAGE_EXTENSIONS)
]

# 按文件名排序
image_files.sort()


# ============================================================
# 基本信息
# ============================================================

print("=" * 80)
print("YOLO 多类别数据集划分")
print("=" * 80)

print(
    f"原始图片目录：{IMAGE_DIR}"
)

print(
    f"原始标签目录：{LABEL_DIR}"
)

print(
    f"输出目录：    {OUTPUT_DIR}"
)

print()

print(
    f"类别数量：    {len(CLASS_NAMES)}"
)

print(
    "类别："
)

for class_id, class_name in enumerate(CLASS_NAMES):

    print(
        f"  {class_id}: {class_name}"
    )

print()

print(
    f"原始图片数量：{len(image_files)}"
)

print(
    f"Train 比例：  {TRAIN_RATIO}"
)

print(
    f"Val 比例：    {VAL_RATIO}"
)

print(
    f"最多使用图片："
    f"{MAX_IMAGES if MAX_IMAGES is not None else '全部'}"
)

print("=" * 80)


# ============================================================
# 检查图片
# ============================================================

if not image_files:

    print()
    print("错误：没有找到图片")

    input("按回车退出...")
    exit()


# ============================================================
# 选择前 N 张
# ============================================================

if MAX_IMAGES is not None:

    if MAX_IMAGES <= 0:

        print(
            "错误：MAX_IMAGES 必须大于 0"
        )

        input("按回车退出...")
        exit()

    image_files = image_files[
        :MAX_IMAGES
    ]


print()

print(
    f"实际选择图片：{len(image_files)}"
)


# ============================================================
# 检查标签
# ============================================================

valid_images = []

missing_labels = []

invalid_labels = []


for image_name in image_files:

    stem = os.path.splitext(
        image_name
    )[0]

    label_name = stem + ".txt"

    label_path = os.path.join(
        LABEL_DIR,
        label_name
    )


    # --------------------------------------------------------
    # 标签不存在
    # --------------------------------------------------------

    if not os.path.exists(label_path):

        missing_labels.append(
            image_name
        )

        continue


    # --------------------------------------------------------
    # 检查标签内容
    # --------------------------------------------------------

    label_valid = True

    try:

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as f:

            lines = f.readlines()


        for line_number, line in enumerate(
            lines,
            start=1
        ):

            line = line.strip()

            # 空行允许
            if not line:
                continue


            parts = line.split()


            # YOLO 标签必须有 5 个参数
            #
            # class x_center y_center width height
            #

            if len(parts) != 5:

                print(
                    f"标签格式错误："
                    f"{label_name} "
                    f"第 {line_number} 行"
                )

                label_valid = False

                break


            class_id = int(
                float(parts[0])
            )


            # ------------------------------------------------
            # 检查类别 ID
            # ------------------------------------------------

            if class_id < 0 or class_id >= len(CLASS_NAMES):

                print(
                    f"类别 ID 错误："
                    f"{label_name} "
                    f"第 {line_number} 行 "
                    f"class={class_id}"
                )

                label_valid = False

                break


            # ------------------------------------------------
            # 检查 bbox
            # ------------------------------------------------

            x = float(parts[1])
            y = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])


            if not (
                0 <= x <= 1
                and
                0 <= y <= 1
                and
                0 < w <= 1
                and
                0 < h <= 1
            ):

                print(
                    f"BBox 越界："
                    f"{label_name} "
                    f"第 {line_number} 行"
                )

                label_valid = False

                break


    except Exception as e:

        print(
            f"读取标签失败："
            f"{label_name}"
        )

        print(
            f"原因：{e}"
        )

        label_valid = False


    # --------------------------------------------------------
    # 记录
    # --------------------------------------------------------

    if label_valid:

        valid_images.append(
            image_name
        )

    else:

        invalid_labels.append(
            image_name
        )


# ============================================================
# 缺少标签
# ============================================================

if missing_labels:

    print()
    print("=" * 80)
    print("警告：以下图片没有对应标签")
    print("=" * 80)

    for image_name in missing_labels:

        print(
            f"  {image_name}"
        )


# ============================================================
# 无效标签
# ============================================================

if invalid_labels:

    print()
    print("=" * 80)
    print("警告：以下图片标签无效")
    print("=" * 80)

    for image_name in invalid_labels:

        print(
            f"  {image_name}"
        )


# ============================================================
# 没有有效图片
# ============================================================

if not valid_images:

    print()
    print(
        "错误：没有找到有效图片"
    )

    input("按回车退出...")
    exit()


# ============================================================
# 随机打乱
# ============================================================

random.seed(
    RANDOM_SEED
)

random.shuffle(
    valid_images
)


# ============================================================
# 划分
# ============================================================

total = len(valid_images)


train_count = int(
    total * TRAIN_RATIO
)


val_count = (
    total - train_count
)


train_files = valid_images[
    :train_count
]

val_files = valid_images[
    train_count:
]


# ============================================================
# 显示划分结果
# ============================================================

print()
print("=" * 80)
print("数据集划分结果")
print("=" * 80)

print(
    f"有效图片：{total}"
)

print(
    f"Train：   {len(train_files)}"
)

print(
    f"Val：     {len(val_files)}"
)

print("=" * 80)


# ============================================================
# 复制数据
# ============================================================

def copy_dataset(
    files,
    image_output_dir,
    label_output_dir
):

    for image_name in files:

        # ----------------------------------------------------
        # 图片
        # ----------------------------------------------------

        source_image = os.path.join(
            IMAGE_DIR,
            image_name
        )

        target_image = os.path.join(
            image_output_dir,
            image_name
        )

        shutil.copy2(
            source_image,
            target_image
        )


        # ----------------------------------------------------
        # 标签
        # ----------------------------------------------------

        stem = os.path.splitext(
            image_name
        )[0]

        label_name = stem + ".txt"


        source_label = os.path.join(
            LABEL_DIR,
            label_name
        )

        target_label = os.path.join(
            label_output_dir,
            label_name
        )

        shutil.copy2(
            source_label,
            target_label
        )


# ============================================================
# Train
# ============================================================

print()
print("开始复制 Train...")

copy_dataset(
    train_files,
    TRAIN_IMAGE_DIR,
    TRAIN_LABEL_DIR
)

print(
    f"Train 完成：{len(train_files)} 张"
)


# ============================================================
# Val
# ============================================================

print()
print("开始复制 Val...")

copy_dataset(
    val_files,
    VAL_IMAGE_DIR,
    VAL_LABEL_DIR
)

print(
    f"Val 完成：{len(val_files)} 张"
)


# ============================================================
# 生成 data.yaml
# ============================================================

yaml_path = os.path.join(
    OUTPUT_DIR,
    "data.yaml"
)


# Windows 路径转换
yaml_root = OUTPUT_DIR.replace(
    "\\",
    "/"
)


# ------------------------------------------------------------
# 自动生成 names
# ------------------------------------------------------------

names_yaml = ""

for class_id, class_name in enumerate(CLASS_NAMES):

    names_yaml += (
        f"  {class_id}: {class_name}\n"
    )


yaml_content = f"""path: {yaml_root}

train: images/train
val: images/val

names:
{names_yaml}"""


with open(
    yaml_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        yaml_content
    )


# ============================================================
# 统计类别
# ============================================================

class_counts = {
    class_id: 0
    for class_id in range(
        len(CLASS_NAMES)
    )
}


def count_classes(files):

    for image_name in files:

        stem = os.path.splitext(
            image_name
        )[0]

        label_name = stem + ".txt"

        label_path = os.path.join(
            LABEL_DIR,
            label_name
        )

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                parts = line.split()

                class_id = int(
                    float(parts[0])
                )

                if class_id in class_counts:

                    class_counts[
                        class_id
                    ] += 1


count_classes(
    train_files
)

count_classes(
    val_files
)


# ============================================================
# 完成
# ============================================================

print()
print("=" * 80)
print("数据集划分完成")
print("=" * 80)

print(
    f"原始图片：  {len(image_files)}"
)

print(
    f"有效图片：  {total}"
)

print()

print(
    f"Train：     {len(train_files)}"
)

print(
    f"Val：       {len(val_files)}"
)

print()

print("类别统计：")

for class_id, class_name in enumerate(CLASS_NAMES):

    print(
        f"  {class_id}: "
        f"{class_name:<10} "
        f"{class_counts[class_id]} 个"
    )

print()

print("Train 图片：")

print(
    f"  {TRAIN_IMAGE_DIR}"
)

print()

print("Train 标签：")

print(
    f"  {TRAIN_LABEL_DIR}"
)

print()

print("Val 图片：")

print(
    f"  {VAL_IMAGE_DIR}"
)

print()

print("Val 标签：")

print(
    f"  {VAL_LABEL_DIR}"
)

print()

print("data.yaml：")

print(
    f"  {yaml_path}"
)

print()

print("data.yaml 内容：")
print("-" * 80)

print(
    yaml_content
)

print("-" * 80)

print()
print("全部完成。")