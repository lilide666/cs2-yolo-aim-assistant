import cv2
from pathlib import Path


# ============================================================
# 配置
# ============================================================

# 图片目录
IMAGE_DIR = Path(
    r"D:\PycharmProjects\computer_controller\make_different_back\7_dataset_procress\images"
)

# 原始 YOLO 标签目录
LABEL_DIR = Path(
    r"D:\PycharmProjects\computer_controller\make_different_back\7_dataset_procress\labels"
)

# 清理后的标签输出目录
OUTPUT_LABEL_DIR = LABEL_DIR.parent / "labels_clean"


# ============================================================
# RMBG 白色区域判断
# ============================================================

# RGB 三个通道都 >= 245
# 就认为是白色
WHITE_THRESHOLD = 245


# bbox 内白色区域超过 80%
# 删除该标注框
WHITE_RATIO_THRESHOLD = 0.80


# ============================================================
# 是否覆盖原标签
# ============================================================

# False：
#   原 labels 不动
#   保存到 labels_clean
#
# True：
#   直接修改原 labels
#
# 强烈建议第一次使用 False
OVERWRITE = False


# ============================================================
# 支持的图片格式
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# 计算 bbox 内白色比例
# ============================================================

def calculate_white_ratio(image, x1, y1, x2, y2):
    """
    计算 bbox 内白色像素占比

    白色定义：
        B >= WHITE_THRESHOLD
        G >= WHITE_THRESHOLD
        R >= WHITE_THRESHOLD

    返回：
        0.0 ~ 1.0
    """

    h, w = image.shape[:2]

    # --------------------------------------------------------
    # 防止 bbox 越界
    # --------------------------------------------------------

    x1 = max(0, min(x1, w))
    y1 = max(0, min(y1, h))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    # --------------------------------------------------------
    # 无效框
    # --------------------------------------------------------

    if x2 <= x1 or y2 <= y1:
        return 1.0

    crop = image[y1:y2, x1:x2]

    if crop.size == 0:
        return 1.0

    # --------------------------------------------------------
    # OpenCV 是 BGR
    # --------------------------------------------------------

    b = crop[:, :, 0]
    g = crop[:, :, 1]
    r = crop[:, :, 2]

    # --------------------------------------------------------
    # 判断白色像素
    # --------------------------------------------------------

    white_mask = (
        (b >= WHITE_THRESHOLD) &
        (g >= WHITE_THRESHOLD) &
        (r >= WHITE_THRESHOLD)
    )

    white_ratio = white_mask.mean()

    return float(white_ratio)


# ============================================================
# YOLO 坐标转换
# ============================================================

def yolo_to_xyxy(
    x_center,
    y_center,
    box_width,
    box_height,
    image_width,
    image_height,
):
    """
    YOLO：
        class x_center y_center width height

    坐标均为归一化坐标

    转换为：
        x1 y1 x2 y2
    """

    x1 = int(
        (x_center - box_width / 2)
        * image_width
    )

    y1 = int(
        (y_center - box_height / 2)
        * image_height
    )

    x2 = int(
        (x_center + box_width / 2)
        * image_width
    )

    y2 = int(
        (y_center + box_height / 2)
        * image_height
    )

    return x1, y1, x2, y2


# ============================================================
# 处理单张图片
# ============================================================

def process_image(image_path):

    # 对应标签
    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    # --------------------------------------------------------
    # 没有标签
    # --------------------------------------------------------

    if not label_path.exists():

        return {
            "total": 0,
            "deleted": 0,
            "kept": 0,
            "skip": True,
        }

    # --------------------------------------------------------
    # 读取图片
    # --------------------------------------------------------

    image = cv2.imread(str(image_path))

    if image is None:

        print(
            f"[错误] 无法读取图片："
            f"{image_path.name}"
        )

        return {
            "total": 0,
            "deleted": 0,
            "kept": 0,
            "skip": True,
        }

    image_height, image_width = image.shape[:2]

    # --------------------------------------------------------
    # 读取 YOLO 标签
    # --------------------------------------------------------

    try:

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as f:

            lines = [
                line.strip()
                for line in f
                if line.strip()
            ]

    except Exception as e:

        print(
            f"[错误] 读取标签失败："
            f"{label_path.name}"
        )

        print(e)

        return {
            "total": 0,
            "deleted": 0,
            "kept": 0,
            "skip": True,
        }

    # --------------------------------------------------------
    # 新标签
    # --------------------------------------------------------

    new_lines = []

    deleted_count = 0
    kept_count = 0

    # ========================================================
    # 遍历 bbox
    # ========================================================

    for line_index, line in enumerate(lines):

        parts = line.split()

        # ----------------------------------------------------
        # YOLO 标准检测格式
        # class x y w h
        # ----------------------------------------------------

        if len(parts) < 5:

            print(
                f"[警告] 标签格式异常："
                f"{label_path.name} "
                f"第 {line_index + 1} 行"
            )

            # 异常标签不删除
            new_lines.append(line)
            kept_count += 1

            continue

        try:

            class_id = parts[0]

            x_center = float(parts[1])
            y_center = float(parts[2])
            box_width = float(parts[3])
            box_height = float(parts[4])

        except ValueError:

            print(
                f"[警告] 标签数据异常："
                f"{label_path.name} "
                f"第 {line_index + 1} 行"
            )

            # 异常标签保留
            new_lines.append(line)
            kept_count += 1

            continue

        # ----------------------------------------------------
        # YOLO -> 像素坐标
        # ----------------------------------------------------

        x1, y1, x2, y2 = yolo_to_xyxy(
            x_center,
            y_center,
            box_width,
            box_height,
            image_width,
            image_height,
        )

        # ----------------------------------------------------
        # 计算白色比例
        # ----------------------------------------------------

        white_ratio = calculate_white_ratio(
            image,
            x1,
            y1,
            x2,
            y2,
        )

        # ----------------------------------------------------
        # 删除条件
        # ----------------------------------------------------

        if white_ratio > WHITE_RATIO_THRESHOLD:

            deleted_count += 1

            print(
                f"[删除] "
                f"{image_path.name} | "
                f"框 #{line_index + 1} | "
                f"class={class_id} | "
                f"白色={white_ratio * 100:.2f}%"
            )

        else:

            # 保留原始标签
            new_lines.append(line)

            kept_count += 1

    # ========================================================
    # 确定输出路径
    # ========================================================

    if OVERWRITE:

        save_path = label_path

    else:

        OUTPUT_LABEL_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        save_path = (
            OUTPUT_LABEL_DIR
            / label_path.name
        )

    # ========================================================
    # 保存
    # ========================================================

    with open(
        save_path,
        "w",
        encoding="utf-8"
    ) as f:

        if new_lines:

            f.write(
                "\n".join(new_lines)
            )

            f.write("\n")

    return {
        "total": len(lines),
        "deleted": deleted_count,
        "kept": kept_count,
        "skip": False,
    }


# ============================================================
# 主程序
# ============================================================

def main():

    print()
    print("=" * 80)
    print("RMBG 白色区域 YOLO 标注清理工具")
    print("=" * 80)

    print()
    print(f"图片目录：")
    print(IMAGE_DIR)

    print()
    print(f"标签目录：")
    print(LABEL_DIR)

    print()
    print(f"输出目录：")
    print(OUTPUT_LABEL_DIR)

    print()
    print("-" * 80)

    print(
        f"白色判定："
        f"R/G/B >= {WHITE_THRESHOLD}"
    )

    print(
        f"删除条件："
        f"bbox 内白色比例 > "
        f"{WHITE_RATIO_THRESHOLD * 100:.1f}%"
    )

    print("-" * 80)

    if OVERWRITE:

        print()
        print("⚠️ 当前模式：直接覆盖原始 labels")

    else:

        print()
        print("当前模式：安全模式")
        print("原始 labels 不会修改")

    print()
    print("=" * 80)

    # ========================================================
    # 检查目录
    # ========================================================

    if not IMAGE_DIR.exists():

        print()
        print("[错误] 图片目录不存在：")
        print(IMAGE_DIR)

        return

    if not LABEL_DIR.exists():

        print()
        print("[错误] 标签目录不存在：")
        print(LABEL_DIR)

        return

    # ========================================================
    # 获取所有图片
    # ========================================================

    image_paths = [
        p
        for p in IMAGE_DIR.iterdir()
        if (
            p.is_file()
            and p.suffix.lower()
            in IMAGE_EXTENSIONS
        )
    ]

    image_paths.sort()

    print()
    print(
        f"发现图片：{len(image_paths)} 张"
    )

    if len(image_paths) == 0:

        print()
        print("[错误] 没有找到图片")

        return

    # ========================================================
    # 总统计
    # ========================================================

    total_images = 0
    total_boxes = 0
    total_deleted = 0
    total_kept = 0
    skipped_images = 0

    # ========================================================
    # 开始处理
    # ========================================================

    print()
    print("开始处理...")
    print()

    for index, image_path in enumerate(
        image_paths,
        start=1
    ):

        result = process_image(
            image_path
        )

        total_images += 1

        total_boxes += result["total"]

        total_deleted += result["deleted"]

        total_kept += result["kept"]

        if result["skip"]:

            skipped_images += 1

        # ----------------------------------------------------
        # 每 100 张显示进度
        # ----------------------------------------------------

        if (
            index % 100 == 0
            or index == len(image_paths)
        ):

            progress = (
                index
                / len(image_paths)
                * 100
            )

            print()
            print(
                f"进度："
                f"{index}/{len(image_paths)} "
                f"({progress:.1f}%)"
            )

            print(
                f"累计："
                f"原始框={total_boxes} | "
                f"删除={total_deleted} | "
                f"保留={total_kept}"
            )

    # ========================================================
    # 最终统计
    # ========================================================

    print()
    print()
    print("=" * 80)
    print("处理完成")
    print("=" * 80)

    print()
    print(
        f"处理图片：        "
        f"{total_images}"
    )

    print(
        f"跳过图片：        "
        f"{skipped_images}"
    )

    print(
        f"原始标注框：      "
        f"{total_boxes}"
    )

    print(
        f"删除标注框：      "
        f"{total_deleted}"
    )

    print(
        f"保留标注框：      "
        f"{total_kept}"
    )

    if total_boxes > 0:

        delete_ratio = (
            total_deleted
            / total_boxes
            * 100
        )

        print(
            f"删除比例：        "
            f"{delete_ratio:.2f}%"
        )

    print()

    if OVERWRITE:

        print(
            "原始标签已修改："
        )

        print(LABEL_DIR)

    else:

        print(
            "清理后的标签："
        )

        print(OUTPUT_LABEL_DIR)

    print()
    print("=" * 80)


# ============================================================
# 程序入口
# ============================================================

if __name__ == "__main__":

    main()