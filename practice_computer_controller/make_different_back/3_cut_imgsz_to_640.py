import os
import cv2
import shutil
from collections import Counter


# ============================================================
# 配置
# ============================================================

# ------------------------------------------------------------
# 原始数据集
# ------------------------------------------------------------

IMAGE_DIR = r"D:\PycharmProjects\computer_controller\make_different_back\1_train_body_head_save\images"

LABEL_DIR = r"D:\PycharmProjects\computer_controller\make_different_back\1_train_body_head_save\labels"


# ------------------------------------------------------------
# 输出数据集
# ------------------------------------------------------------

OUTPUT_DIR = r"D:\PycharmProjects\computer_controller\make_different_back\4_640_imgsz_dataset"

OUTPUT_IMAGE_DIR = os.path.join(
    OUTPUT_DIR,
    "images"
)

OUTPUT_LABEL_DIR = os.path.join(
    OUTPUT_DIR,
    "labels"
)


# ============================================================
# 裁剪参数
# ============================================================

TILE_SIZE = 640


# ------------------------------------------------------------
# 目标周围希望保留多少背景
#
# 例如：
#
# body bbox:
#  x1=500
#  x2=900
#
# crop 会尽量让 body 周围还有一些背景。
#
# 但是最终优先保证 bbox 完整。
# ------------------------------------------------------------

TARGET_MARGIN = 80


# ------------------------------------------------------------
# 两个 crop 如果高度/宽度非常接近，可以认为是重复 crop
# ------------------------------------------------------------

CROP_DUPLICATE_DISTANCE = 80


# ------------------------------------------------------------
# 如果一个 bbox 小于这个尺寸，不影响。
#
# 这是最终裁剪 bbox 的最低像素尺寸。
# ------------------------------------------------------------

MIN_BOX_WIDTH = 2
MIN_BOX_HEIGHT = 2


# ------------------------------------------------------------
# 是否删除旧输出
# ------------------------------------------------------------

CLEAR_OUTPUT = True


# ============================================================
# 类别
# ============================================================

CLASS_NAMES = {
    0: "body",
    1: "head",
}


# ============================================================
# 图片格式
# ============================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
)


# ============================================================
# 读取 YOLO 标签
# ============================================================

def read_yolo_labels(
    label_path,
    image_width,
    image_height
):

    boxes = []

    if not os.path.exists(label_path):
        return boxes

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:

        for line_num, line in enumerate(f, 1):

            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if len(parts) < 5:

                print(
                    f"[警告] 标签格式错误："
                    f"{label_path} "
                    f"第 {line_num} 行"
                )

                continue

            try:

                cls = int(parts[0])

                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

            except ValueError:

                print(
                    f"[警告] 标签数值错误："
                    f"{label_path} "
                    f"第 {line_num} 行"
                )

                continue

            # ------------------------------------------------
            # YOLO normalized
            # ->
            # 像素坐标
            # ------------------------------------------------

            x_center *= image_width
            y_center *= image_height

            width *= image_width
            height *= image_height

            x1 = x_center - width / 2
            y1 = y_center - height / 2

            x2 = x_center + width / 2
            y2 = y_center + height / 2

            boxes.append({
                "class": cls,

                "x1": x1,
                "y1": y1,

                "x2": x2,
                "y2": y2,

                "index": len(boxes),
            })

    return boxes


# ============================================================
# 计算一个 bbox 的中心
# ============================================================

def box_center(box):

    cx = (
        box["x1"] +
        box["x2"]
    ) / 2

    cy = (
        box["y1"] +
        box["y2"]
    ) / 2

    return cx, cy


# ============================================================
# 判断 bbox 是否能够完整放入 640x640
# ============================================================

def can_fit_box(
    box,
    image_width,
    image_height
):

    width = (
        box["x2"] -
        box["x1"]
    )

    height = (
        box["y2"] -
        box["y1"]
    )

    return (
        width <= TILE_SIZE
        and
        height <= TILE_SIZE
        and
        image_width >= TILE_SIZE
        and
        image_height >= TILE_SIZE
    )


# ============================================================
# 根据 bbox 生成一个 640x640 crop
#
# 核心函数
# ============================================================

def make_crop_for_box(
    box,
    image_width,
    image_height
):

    box_x1 = box["x1"]
    box_y1 = box["y1"]

    box_x2 = box["x2"]
    box_y2 = box["y2"]

    box_width = (
        box_x2 -
        box_x1
    )

    box_height = (
        box_y2 -
        box_y1
    )

    # ========================================================
    # bbox 太大
    # ========================================================

    if box_width > TILE_SIZE:

        return None

    if box_height > TILE_SIZE:

        return None

    # ========================================================
    # 目标中心
    # ========================================================

    cx = (
        box_x1 +
        box_x2
    ) / 2

    cy = (
        box_y1 +
        box_y2
    ) / 2

    # ========================================================
    # 理想 crop
    #
    # 让 bbox 中心处于 640 crop 中心
    # ========================================================

    crop_x1 = cx - TILE_SIZE / 2
    crop_y1 = cy - TILE_SIZE / 2

    # ========================================================
    # 首先保证 bbox 完整
    #
    # crop 必须满足：
    #
    # crop_x1 <= box_x1
    # crop_x2 >= box_x2
    #
    # crop_y1 <= box_y1
    # crop_y2 >= box_y2
    # ========================================================

    # --------------------------------------------------------
    # 如果 bbox 左边超过 crop
    # --------------------------------------------------------

    if crop_x1 > box_x1:

        crop_x1 = box_x1

    # --------------------------------------------------------
    # 如果 bbox 右边超过 crop
    # --------------------------------------------------------

    if crop_x1 + TILE_SIZE < box_x2:

        crop_x1 = (
            box_x2 -
            TILE_SIZE
        )

    # --------------------------------------------------------
    # Y
    # --------------------------------------------------------

    if crop_y1 > box_y1:

        crop_y1 = box_y1

    if crop_y1 + TILE_SIZE < box_y2:

        crop_y1 = (
            box_y2 -
            TILE_SIZE
        )

    # ========================================================
    # 尽量让 crop 不贴边
    # ========================================================

    # --------------------------------------------------------
    # 限制到原图
    # --------------------------------------------------------

    crop_x1 = max(
        0,
        crop_x1
    )

    crop_y1 = max(
        0,
        crop_y1
    )

    crop_x1 = min(
        crop_x1,
        image_width - TILE_SIZE
    )

    crop_y1 = min(
        crop_y1,
        image_height - TILE_SIZE
    )

    # ========================================================
    # 最终整数坐标
    # ========================================================

    crop_x1 = int(round(crop_x1))
    crop_y1 = int(round(crop_y1))

    crop_x2 = (
        crop_x1 +
        TILE_SIZE
    )

    crop_y2 = (
        crop_y1 +
        TILE_SIZE
    )

    return (
        crop_x1,
        crop_y1,
        crop_x2,
        crop_y2
    )


# ============================================================
# 判断 bbox 是否完整位于 crop 中
# ============================================================

def box_fully_inside_crop(
    box,
    crop
):

    crop_x1, crop_y1, crop_x2, crop_y2 = crop

    epsilon = 0.01

    return (
        box["x1"] >= crop_x1 - epsilon
        and
        box["y1"] >= crop_y1 - epsilon
        and
        box["x2"] <= crop_x2 + epsilon
        and
        box["y2"] <= crop_y2 + epsilon
    )


# ============================================================
# 判断两个 crop 是否非常接近
# ============================================================

def crops_are_similar(
    crop_a,
    crop_b
):

    ax1, ay1, ax2, ay2 = crop_a
    bx1, by1, bx2, by2 = crop_b

    distance = max(
        abs(ax1 - bx1),
        abs(ay1 - by1)
    )

    return (
        distance <=
        CROP_DUPLICATE_DISTANCE
    )


# ============================================================
# 计算两个 bbox 是否有交集
# ============================================================

def boxes_intersect(
    box,
    crop
):

    crop_x1, crop_y1, crop_x2, crop_y2 = crop

    ix1 = max(
        box["x1"],
        crop_x1
    )

    iy1 = max(
        box["y1"],
        crop_y1
    )

    ix2 = min(
        box["x2"],
        crop_x2
    )

    iy2 = min(
        box["y2"],
        crop_y2
    )

    return (
        ix2 > ix1
        and
        iy2 > iy1
    )


# ============================================================
# 计算 bbox 与 crop 的交集
# ============================================================

def clip_box_to_crop(
    box,
    crop
):

    crop_x1, crop_y1, crop_x2, crop_y2 = crop

    x1 = max(
        box["x1"],
        crop_x1
    )

    y1 = max(
        box["y1"],
        crop_y1
    )

    x2 = min(
        box["x2"],
        crop_x2
    )

    y2 = min(
        box["y2"],
        crop_y2
    )

    if x2 <= x1 or y2 <= y1:

        return None

    return (
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
# bbox -> 当前 640 crop 的 YOLO
# ============================================================

def box_to_yolo(
    box,
    crop
):

    crop_x1, crop_y1, _, _ = crop

    x1, y1, x2, y2 = box

    # --------------------------------------------------------
    # 转到 crop 局部坐标
    # --------------------------------------------------------

    local_x1 = (
        x1 -
        crop_x1
    )

    local_y1 = (
        y1 -
        crop_y1
    )

    local_x2 = (
        x2 -
        crop_x1
    )

    local_y2 = (
        y2 -
        crop_y1
    )

    width = (
        local_x2 -
        local_x1
    )

    height = (
        local_y2 -
        local_y1
    )

    x_center = (
        local_x1 +
        local_x2
    ) / 2

    y_center = (
        local_y1 +
        local_y2
    ) / 2

    # --------------------------------------------------------
    # normalized
    # --------------------------------------------------------

    x_center /= TILE_SIZE
    y_center /= TILE_SIZE

    width /= TILE_SIZE
    height /= TILE_SIZE

    return (
        x_center,
        y_center,
        width,
        height
    )


# ============================================================
# 生成一个 crop 中的所有标签
#
# 注意：
#
# 这里不只保存“目标中心所在的 bbox”
# 而是把 crop 中所有相交的 bbox 都保存。
#
# 这样一张 crop 可以包含很多人。
# ============================================================

def generate_labels_for_crop(
    boxes,
    crop,
    stats
):

    labels = []

    crop_x1, crop_y1, crop_x2, crop_y2 = crop

    for box in boxes:

        if not boxes_intersect(
            box,
            crop
        ):
            continue

        clipped = clip_box_to_crop(
            box,
            crop
        )

        if clipped is None:
            continue

        x1, y1, x2, y2 = clipped

        width = x2 - x1
        height = y2 - y1

        if (
            width < MIN_BOX_WIDTH
            or
            height < MIN_BOX_HEIGHT
        ):
            continue

        # ----------------------------------------------------
        # 如果完整包含
        # ----------------------------------------------------

        full = box_fully_inside_crop(
            box,
            crop
        )

        if full:

            stats["full_box_labels"] += 1

        else:

            stats["clipped_box_labels"] += 1

        # ----------------------------------------------------
        # 转 YOLO
        # ----------------------------------------------------

        x_center, y_center, w, h = (
            box_to_yolo(
                clipped,
                crop
            )
        )

        # ----------------------------------------------------
        # 安全限制
        # ----------------------------------------------------

        x_center = max(
            0.0,
            min(1.0, x_center)
        )

        y_center = max(
            0.0,
            min(1.0, y_center)
        )

        w = max(
            0.0,
            min(1.0, w)
        )

        h = max(
            0.0,
            min(1.0, h)
        )

        labels.append(
            f"{box['class']} "
            f"{x_center:.6f} "
            f"{y_center:.6f} "
            f"{w:.6f} "
            f"{h:.6f}"
        )

    return labels


# ============================================================
# 选择 crop
#
# 目标：
#
# 1. 每个 bbox 都有一个完整 crop
# 2. 尽量复用 crop
# 3. 减少最终图片数量
# ============================================================

def generate_crops(
    boxes,
    image_width,
    image_height,
    stats
):

    candidate_crops = []

    # ========================================================
    # 第一阶段
    #
    # 每个 bbox 生成一个“保证完整”的 crop
    # ========================================================

    for box in boxes:

        crop = make_crop_for_box(
            box,
            image_width,
            image_height
        )

        if crop is None:

            stats[
                "impossible_boxes"
            ].append(
                box["index"]
            )

            continue

        candidate_crops.append(
            crop
        )

    # ========================================================
    # 去重
    # ========================================================

    unique_crops = []

    for crop in candidate_crops:

        duplicate = False

        for existing in unique_crops:

            if crops_are_similar(
                crop,
                existing
            ):

                duplicate = True
                break

        if not duplicate:

            unique_crops.append(
                crop
            )

    # ========================================================
    # 第二阶段：
    #
    # 尝试删除没有“独占目标”的重复 crop
    #
    # 但是必须保证每个 bbox 至少有一个
    # 完整 crop。
    #
    # 所以这里做覆盖检查。
    # ========================================================

    changed = True

    while changed:

        changed = False

        for i in range(
            len(unique_crops)
        ):

            crop = unique_crops[i]

            # ------------------------------------------------
            # 假设删除这个 crop
            # ------------------------------------------------

            remaining = (
                unique_crops[:i]
                +
                unique_crops[i + 1:]
            )

            all_safe = True

            # ------------------------------------------------
            # 检查所有 bbox
            # ------------------------------------------------

            for box in boxes:

                # 已经在其他 crop 中完整保留
                covered = False

                for other in remaining:

                    if box_fully_inside_crop(
                        box,
                        other
                    ):

                        covered = True
                        break

                if not covered:

                    all_safe = False
                    break

            # ------------------------------------------------
            # 如果删除后所有 bbox
            # 仍然至少有一个完整 crop
            # ------------------------------------------------

            if all_safe:

                unique_crops.pop(i)

                changed = True

                break

    return unique_crops


# ============================================================
# 处理单张图片
# ============================================================

def process_image(
    image_path,
    label_path,
    stats
):

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            f"[错误] 无法读取："
            f"{image_path}"
        )

        stats["read_error"] += 1

        return 0

    image_height, image_width = image.shape[:2]

    # ========================================================
    # 检查图片是否足够大
    # ========================================================

    if (
        image_width < TILE_SIZE
        or
        image_height < TILE_SIZE
    ):

        print(
            f"[错误] 图片小于 640："
            f"{os.path.basename(image_path)} "
            f"{image_width}x{image_height}"
        )

        stats["small_images"] += 1

        return 0

    # ========================================================
    # 读取标签
    # ========================================================

    boxes = read_yolo_labels(
        label_path,
        image_width,
        image_height
    )

    # ========================================================
    # 原始 bbox 统计
    # ========================================================

    stats["input_boxes"] += len(boxes)

    for box in boxes:

        stats[
            "input_by_class"
        ][box["class"]] += 1

        box_width = (
            box["x2"] -
            box["x1"]
        )

        box_height = (
            box["y2"] -
            box["y1"]
        )

        if box_width > TILE_SIZE:

            stats[
                "too_large_boxes"
            ] += 1

        if box_height > TILE_SIZE:

            stats[
                "too_large_boxes"
            ] += 1

    # ========================================================
    # 没有标签
    # ========================================================

    if not boxes:

        stats["empty_source_images"] += 1

        return 0

    # ========================================================
    # 生成 crop
    # ========================================================

    crops = generate_crops(
        boxes,
        image_width,
        image_height,
        stats
    )

    # ========================================================
    # 保存 crop
    # ========================================================

    base_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    generated = 0

    # --------------------------------------------------------
    # 用于最终检查
    #
    # box index -> 是否存在完整 crop
    # --------------------------------------------------------

    box_full_coverage = {
        box["index"]: False
        for box in boxes
    }

    # ========================================================
    # 逐个 crop
    # ========================================================

    for crop_index, crop in enumerate(
        crops
    ):

        crop_x1, crop_y1, crop_x2, crop_y2 = crop

        # ----------------------------------------------------
        # 图像
        # ----------------------------------------------------

        tile = image[
            crop_y1:crop_y2,
            crop_x1:crop_x2
        ]

        # ----------------------------------------------------
        # 安全检查
        # ----------------------------------------------------

        if (
            tile.shape[0] != TILE_SIZE
            or
            tile.shape[1] != TILE_SIZE
        ):

            print(
                f"[错误] crop 尺寸错误："
                f"{base_name} "
                f"{crop}"
            )

            continue

        # ----------------------------------------------------
        # 标签
        # ----------------------------------------------------

        labels = generate_labels_for_crop(
            boxes,
            crop,
            stats
        )

        if not labels:

            continue

        # ----------------------------------------------------
        # 更新“完整覆盖”状态
        # ----------------------------------------------------

        for box in boxes:

            if box_fully_inside_crop(
                box,
                crop
            ):

                box_full_coverage[
                    box["index"]
                ] = True

        # ----------------------------------------------------
        # 文件名
        # ----------------------------------------------------

        tile_name = (
            f"{base_name}"
            f"_target_{crop_index:03d}"
        )

        output_image_path = os.path.join(
            OUTPUT_IMAGE_DIR,
            tile_name + ".jpg"
        )

        output_label_path = os.path.join(
            OUTPUT_LABEL_DIR,
            tile_name + ".txt"
        )

        # ----------------------------------------------------
        # 保存图片
        # ----------------------------------------------------

        success = cv2.imwrite(
            output_image_path,
            tile
        )

        if not success:

            print(
                f"[错误] 保存图片失败："
                f"{output_image_path}"
            )

            continue

        # ----------------------------------------------------
        # 保存标签
        # ----------------------------------------------------

        with open(
            output_label_path,
            "w",
            encoding="utf-8"
        ) as f:

            f.write(
                "\n".join(labels)
            )

        generated += 1

        stats["output_images"] += 1

        stats[
            "output_boxes"
        ] += len(labels)

    # ========================================================
    # 最终检查
    # ========================================================

    for box in boxes:

        if box_full_coverage[
            box["index"]
        ]:

            stats[
                "guaranteed_boxes"
            ] += 1

        else:

            stats[
                "missing_boxes"
            ].append(
                (
                    base_name,
                    box["index"],
                    box["class"],
                    box["x1"],
                    box["y1"],
                    box["x2"],
                    box["y2"]
                )
            )

    return generated


# ============================================================
# 创建统计器
# ============================================================

def create_stats():

    return {

        "input_images": 0,

        "input_boxes": 0,

        "input_by_class": Counter(),

        "output_images": 0,

        "output_boxes": 0,

        "full_box_labels": 0,

        "clipped_box_labels": 0,

        "guaranteed_boxes": 0,

        "missing_boxes": [],

        "impossible_boxes": [],

        "too_large_boxes": 0,

        "read_error": 0,

        "small_images": 0,

        "empty_source_images": 0,
    }


# ============================================================
# 打印统计
# ============================================================

def print_statistics(
    stats
):

    print()
    print("=" * 80)
    print("最终统计")
    print("=" * 80)

    print()
    print("【图片】")

    print(
        f"原始图片：          "
        f"{stats['input_images']}"
    )

    print(
        f"输出 640x640：      "
        f"{stats['output_images']}"
    )

    print()
    print("【BBox】")

    print(
        f"原始 BBox：         "
        f"{stats['input_boxes']}"
    )

    print(
        f"输出 BBox：         "
        f"{stats['output_boxes']}"
    )

    print()
    print("【类别】")

    classes = sorted(
        stats[
            "input_by_class"
        ].keys()
    )

    for cls in classes:

        name = CLASS_NAMES.get(
            cls,
            f"class_{cls}"
        )

        print(
            f"class {cls} ({name})："
            f"{stats['input_by_class'][cls]}"
        )

    print()
    print("【BBox 保留情况】")

    print(
        f"完整保留的标签实例："
        f"{stats['full_box_labels']}"
    )

    print(
        f"被裁剪的标签实例："
        f"{stats['clipped_box_labels']}"
    )

    print(
        f"至少存在一个完整 crop："
        f"{stats['guaranteed_boxes']}"
    )

    print(
        f"没有完整 crop："
        f"{len(stats['missing_boxes'])}"
    )

    print()
    print("【异常】")

    print(
        f"大于 640 的 BBox："
        f"{stats['too_large_boxes']}"
    )

    print(
        f"无法完整保留的 BBox："
        f"{len(stats['impossible_boxes'])}"
    )

    print(
        f"读取失败图片："
        f"{stats['read_error']}"
    )

    print(
        f"小于 640 的图片："
        f"{stats['small_images']}"
    )

    print(
        f"原始空标签图片："
        f"{stats['empty_source_images']}"
    )

    # ========================================================
    # 最终结论
    # ========================================================

    print()
    print("=" * 80)

    if (
        len(stats["missing_boxes"]) == 0
        and
        len(stats["impossible_boxes"]) == 0
    ):

        print(
            "✓ 完美：每一个原始 BBox "
            "都有至少一个 640x640 crop 完整保留"
        )

    else:

        print(
            "⚠ 注意：存在无法完整保留的 BBox"
        )

        if stats["missing_boxes"]:

            print()
            print(
                "没有完整 crop 的 BBox："
            )

            for item in stats[
                "missing_boxes"
            ][:50]:

                (
                    name,
                    index,
                    cls,
                    x1,
                    y1,
                    x2,
                    y2
                ) = item

                print(
                    f"  {name} "
                    f"box={index} "
                    f"class={cls} "
                    f"xyxy=("
                    f"{x1:.1f}, "
                    f"{y1:.1f}, "
                    f"{x2:.1f}, "
                    f"{y2:.1f})"
                )

        if stats[
            "impossible_boxes"
        ]:

            print()
            print(
                "BBox 本身大于 640，"
                "无法放进 640x640："
            )

            for index in stats[
                "impossible_boxes"
            ][:50]:

                print(
                    f"  box index = {index}"
                )

    print("=" * 80)


# ============================================================
# 主程序
# ============================================================

def main():

    print()
    print("=" * 80)
    print("目标驱动 640x640 YOLO 数据集生成器")
    print("=" * 80)

    print()
    print("输入图片：")
    print(IMAGE_DIR)

    print()
    print("输入标签：")
    print(LABEL_DIR)

    print()
    print("输出：")
    print(OUTPUT_DIR)

    print()
    print("裁剪尺寸：640x640")

    print(
        "核心规则："
        "每个原始 BBox 至少拥有一个完整 crop"
    )

    print()

    # ========================================================
    # 清空输出
    # ========================================================

    if CLEAR_OUTPUT:

        if os.path.exists(
            OUTPUT_DIR
        ):

            print(
                "正在删除旧数据集..."
            )

            shutil.rmtree(
                OUTPUT_DIR
            )

            print(
                "旧数据集删除完成"
            )

    # ========================================================
    # 创建目录
    # ========================================================

    os.makedirs(
        OUTPUT_IMAGE_DIR,
        exist_ok=True
    )

    os.makedirs(
        OUTPUT_LABEL_DIR,
        exist_ok=True
    )

    # ========================================================
    # 获取图片
    # ========================================================

    image_files = [
        f
        for f in os.listdir(
            IMAGE_DIR
        )
        if f.lower().endswith(
            IMAGE_EXTENSIONS
        )
    ]

    image_files.sort()

    if not image_files:

        print(
            "[错误] 没有找到图片"
        )

        return

    # ========================================================
    # 统计
    # ========================================================

    stats = create_stats()

    stats[
        "input_images"
    ] = len(image_files)

    # ========================================================
    # 开始
    # ========================================================

    print(
        f"发现 {len(image_files)} 张图片"
    )

    print()

    for index, image_file in enumerate(
        image_files,
        1
    ):

        image_path = os.path.join(
            IMAGE_DIR,
            image_file
        )

        base_name = os.path.splitext(
            image_file
        )[0]

        label_path = os.path.join(
            LABEL_DIR,
            base_name + ".txt"
        )

        before = (
            stats["output_images"]
        )

        generated = process_image(
            image_path,
            label_path,
            stats
        )

        after = (
            stats["output_images"]
        )

        real_generated = (
            after -
            before
        )

        print(
            f"[{index:03d}/{len(image_files):03d}] "
            f"{image_file:<20} "
            f"-> {real_generated:3d} 张"
        )

    # ========================================================
    # 最终统计
    # ========================================================

    print_statistics(
        stats
    )

    # ========================================================
    # 最终文件检查
    # ========================================================

    output_images = [
        f
        for f in os.listdir(
            OUTPUT_IMAGE_DIR
        )
        if f.lower().endswith(
            IMAGE_EXTENSIONS
        )
    ]

    output_labels = [
        f
        for f in os.listdir(
            OUTPUT_LABEL_DIR
        )
        if f.lower().endswith(
            ".txt"
        )
    ]

    print()
    print("=" * 80)
    print("文件检查")
    print("=" * 80)

    print(
        f"实际图片："
        f"{len(output_images)}"
    )

    print(
        f"实际标签："
        f"{len(output_labels)}"
    )

    if (
        len(output_images)
        ==
        len(output_labels)
    ):

        print(
            "✓ 图片 / 标签数量一致"
        )

    else:

        print(
            "✗ 图片 / 标签数量不一致"
        )

    print()
    print("输出目录：")

    print(
        OUTPUT_IMAGE_DIR
    )

    print(
        OUTPUT_LABEL_DIR
    )

    print()
    print("=" * 80)
    print("处理完成")
    print("=" * 80)


# ============================================================
# 启动
# ============================================================

if __name__ == "__main__":

    main()