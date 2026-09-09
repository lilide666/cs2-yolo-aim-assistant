import os
import cv2


# ============================================================
# 配置
# ============================================================

# ============================================================
# 原图
# ============================================================

ORIGINAL_IMAGE_DIR = (
    r"D:\PycharmProjects\computer_controller\make_dataset\train_body\images"
)

# ============================================================
# 原图人物标签
#
# 注意：
# 这个目录只读
# 程序绝对不会修改这里面的 txt
# ============================================================

ORIGINAL_LABEL_DIR = (
    r"D:\PycharmProjects\computer_controller\make_dataset\train_body\labels"
)

# ============================================================
# 头部裁剪图
# ============================================================

HEAD_IMAGE_DIR = (
    r"D:\PycharmProjects\computer_controller\make_dataset\head_annotation\images"
)

# ============================================================
# 头部裁剪图标签
# ============================================================

HEAD_LABEL_DIR = (
    r"D:\PycharmProjects\computer_controller\make_dataset\head_annotation\labels"
)

# ============================================================
# 输出最终标签
#
# 这里会生成新的标签
# 不会修改 ORIGINAL_LABEL_DIR
# ============================================================

OUTPUT_LABEL_DIR = (
    r"D:\PycharmProjects\computer_controller\make_dataset\head_to_original\labels"
)

# ============================================================
# 类别
# ============================================================

HEAD_CLASS_ID = 1
PERSON_CLASS_ID = 0


# ============================================================
# 去重参数
# ============================================================

# ------------------------------------------------------------
# 原来：
#
# DUPLICATE_IOU = 0.50
#
# 太容易误删
#
# 现在改成：
#
# 只有两个框高度重叠才认为是重复
# ------------------------------------------------------------

DUPLICATE_IOU = 0.98


# ============================================================
# 创建输出目录
# ============================================================

os.makedirs(
    OUTPUT_LABEL_DIR,
    exist_ok=True
)


# ============================================================
# 图片扩展名
# ============================================================

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# YOLO -> XYXY
# ============================================================

def yolo_to_xyxy(
    cx,
    cy,
    w,
    h,
    image_width,
    image_height
):

    x1 = (
        cx - w / 2
    ) * image_width

    y1 = (
        cy - h / 2
    ) * image_height

    x2 = (
        cx + w / 2
    ) * image_width

    y2 = (
        cy + h / 2
    ) * image_height

    return (
        x1,
        y1,
        x2,
        y2
    )


# ============================================================
# XYXY -> YOLO
# ============================================================

def xyxy_to_yolo(
    x1,
    y1,
    x2,
    y2,
    image_width,
    image_height
):

    # --------------------------------------------------------
    # 限制在图片范围
    # --------------------------------------------------------

    x1 = max(
        0,
        min(
            image_width,
            x1
        )
    )

    y1 = max(
        0,
        min(
            image_height,
            y1
        )
    )

    x2 = max(
        0,
        min(
            image_width,
            x2
        )
    )

    y2 = max(
        0,
        min(
            image_height,
            y2
        )
    )

    cx = (
        x1 + x2
    ) / 2

    cy = (
        y1 + y2
    ) / 2

    w = (
        x2 - x1
    )

    h = (
        y2 - y1
    )

    cx /= image_width
    cy /= image_height

    w /= image_width
    h /= image_height

    return (
        cx,
        cy,
        w,
        h
    )


# ============================================================
# IoU
# ============================================================

def calculate_iou(
    box1,
    box2
):

    x1 = max(
        box1[0],
        box2[0]
    )

    y1 = max(
        box1[1],
        box2[1]
    )

    x2 = min(
        box1[2],
        box2[2]
    )

    y2 = min(
        box1[3],
        box2[3]
    )

    intersection_width = max(
        0,
        x2 - x1
    )

    intersection_height = max(
        0,
        y2 - y1
    )

    intersection = (
        intersection_width *
        intersection_height
    )

    area1 = (
        max(
            0,
            box1[2] - box1[0]
        )
        *
        max(
            0,
            box1[3] - box1[1]
        )
    )

    area2 = (
        max(
            0,
            box2[2] - box2[0]
        )
        *
        max(
            0,
            box2[3] - box2[1]
        )
    )

    union = (
        area1 +
        area2 -
        intersection
    )

    if union <= 0:
        return 0.0

    return (
        intersection /
        union
    )


# ============================================================
# Box 面积
# ============================================================

def box_area(box):

    width = max(
        0,
        box[2] - box[0]
    )

    height = max(
        0,
        box[3] - box[1]
    )

    return (
        width *
        height
    )


# ============================================================
# 读取 YOLO 标签
# ============================================================

def read_yolo_labels(
    label_path
):

    labels = []

    if not os.path.exists(
        label_path
    ):
        return labels

    with open(
        label_path,
        "r",
        encoding="utf-8"
    ) as f:

        lines = f.readlines()

    for line in lines:

        parts = line.strip().split()

        if len(parts) != 5:
            continue

        try:

            class_id = int(
                parts[0]
            )

            cx = float(
                parts[1]
            )

            cy = float(
                parts[2]
            )

            w = float(
                parts[3]
            )

            h = float(
                parts[4]
            )

        except ValueError:

            continue

        labels.append(
            (
                class_id,
                cx,
                cy,
                w,
                h
            )
        )

    return labels


# ============================================================
# 获取原图人物框
# ============================================================

def get_person_boxes(
    label_path,
    image_width,
    image_height
):

    labels = read_yolo_labels(
        label_path
    )

    boxes = []

    for (
        class_id,
        cx,
        cy,
        w,
        h
    ) in labels:

        if class_id != PERSON_CLASS_ID:
            continue

        box = yolo_to_xyxy(
            cx,
            cy,
            w,
            h,
            image_width,
            image_height
        )

        boxes.append(
            box
        )

    return boxes


# ============================================================
# 解析裁剪图文件名
#
# 例如：
#
# 000000_person0.jpg
# 000000_person1.jpg
# 000000_person10.jpg
#
# 得到：
#
# original_stem = 000000
# person_index = 0
# ============================================================

def parse_crop_name(
    filename
):

    stem = os.path.splitext(
        filename
    )[0]

    marker = "_person"

    if marker not in stem:
        return None, None

    original_stem, index_str = (
        stem.rsplit(
            marker,
            1
        )
    )

    try:

        person_index = int(
            index_str
        )

    except ValueError:

        return None, None

    return (
        original_stem,
        person_index
    )


# ============================================================
# 去重
#
# 重要：
#
# 标全优先
#
# 只有 IoU >= 0.80
# 才认为两个 head 是同一个头
#
# 如果重复：
# 保留面积更大的框
#
# 这样比“先到先得”更加稳定
# ============================================================

def remove_duplicates(
    boxes
):

    if len(boxes) <= 1:

        return boxes.copy()

    # --------------------------------------------------------
    # 按面积从大到小处理
    #
    # 大框优先保留
    # --------------------------------------------------------

    boxes_sorted = sorted(
        boxes,
        key=box_area,
        reverse=True
    )

    result = []

    for box in boxes_sorted:

        is_duplicate = False

        for old_box in result:

            iou = calculate_iou(
                box,
                old_box
            )

            # ------------------------------------------------
            # 只有非常高的 IoU 才删除
            # ------------------------------------------------

            if iou >= DUPLICATE_IOU:

                is_duplicate = True

                break

        if not is_duplicate:

            result.append(
                box
            )

    return result


# ============================================================
# 获取原图
# ============================================================

def find_original_image(
    original_stem
):

    for ext in IMAGE_EXTENSIONS:

        image_path = os.path.join(
            ORIGINAL_IMAGE_DIR,
            original_stem + ext
        )

        if os.path.exists(
            image_path
        ):

            return image_path

    return None


# ============================================================
# 获取所有 Head 图片
# ============================================================

head_images = [
    f
    for f in os.listdir(
        HEAD_IMAGE_DIR
    )
    if f.lower().endswith(
        IMAGE_EXTENSIONS
    )
]

head_images.sort()


# ============================================================
# 开始
# ============================================================

print(
    "=" * 80
)

print(
    "开始将 Head 放回原图"
)

print(
    "=" * 80
)

print(
    f"原图目录:       {ORIGINAL_IMAGE_DIR}"
)

print(
    f"原始标签目录:   {ORIGINAL_LABEL_DIR}"
)

print(
    f"Head 图片目录:  {HEAD_IMAGE_DIR}"
)

print(
    f"Head 标签目录:  {HEAD_LABEL_DIR}"
)

print(
    f"输出目录:       {OUTPUT_LABEL_DIR}"
)

print(
    f"Head 类别:      {HEAD_CLASS_ID}"
)

print(
    f"Person 类别:    {PERSON_CLASS_ID}"
)

print(
    f"去重 IoU:       {DUPLICATE_IOU}"
)

print(
    f"Head 图片数量:  {len(head_images)}"
)

print(
    "=" * 80
)


# ============================================================
# 汇总
#
# {
#     "000001": [
#         box,
#         box,
#         box
#     ]
# }
# ============================================================

all_head_boxes = {}


# ============================================================
# 统计
# ============================================================

mapped_count = 0
failed_count = 0

head_count = 0

total_person_boxes = 0


# ============================================================
# 处理 Head 图片
# ============================================================

for crop_filename in head_images:

    # --------------------------------------------------------
    # 解析文件名
    # --------------------------------------------------------

    original_stem, person_index = (
        parse_crop_name(
            crop_filename
        )
    )

    if original_stem is None:

        print(
            f"[跳过] 无法解析文件名: "
            f"{crop_filename}"
        )

        failed_count += 1

        continue

    # --------------------------------------------------------
    # 找原图
    # --------------------------------------------------------

    original_image_path = (
        find_original_image(
            original_stem
        )
    )

    if original_image_path is None:

        print(
            f"[失败] 找不到原图: "
            f"{original_stem}"
        )

        failed_count += 1

        continue

    # --------------------------------------------------------
    # 原图标签
    # --------------------------------------------------------

    original_label_path = os.path.join(
        ORIGINAL_LABEL_DIR,
        original_stem + ".txt"
    )

    # --------------------------------------------------------
    # 读取原图
    # --------------------------------------------------------

    original_image = cv2.imread(
        original_image_path
    )

    if original_image is None:

        print(
            f"[失败] 无法读取原图: "
            f"{original_image_path}"
        )

        failed_count += 1

        continue

    original_height, original_width = (
        original_image.shape[:2]
    )

    # --------------------------------------------------------
    # 获取人物框
    # --------------------------------------------------------

    person_boxes = get_person_boxes(
        original_label_path,
        original_width,
        original_height
    )

    total_person_boxes += len(
        person_boxes
    )

    # --------------------------------------------------------
    # 检查人物编号
    # --------------------------------------------------------

    if person_index >= len(
        person_boxes
    ):

        print(
            f"[失败] {crop_filename} "
            f"人物编号 {person_index} "
            f"超过人物数量 "
            f"{len(person_boxes)}"
        )

        failed_count += 1

        continue

    # --------------------------------------------------------
    # 当前人物框
    # --------------------------------------------------------

    person_box = person_boxes[
        person_index
    ]

    person_x1 = person_box[0]
    person_y1 = person_box[1]

    person_x2 = person_box[2]
    person_y2 = person_box[3]

    person_width = (
        person_x2 -
        person_x1
    )

    person_height = (
        person_y2 -
        person_y1
    )

    # --------------------------------------------------------
    # Head 裁剪图
    # --------------------------------------------------------

    crop_image_path = os.path.join(
        HEAD_IMAGE_DIR,
        crop_filename
    )

    crop_image = cv2.imread(
        crop_image_path
    )

    if crop_image is None:

        print(
            f"[失败] 无法读取裁剪图: "
            f"{crop_filename}"
        )

        failed_count += 1

        continue

    crop_height, crop_width = (
        crop_image.shape[:2]
    )

    # --------------------------------------------------------
    # Head 标签
    # --------------------------------------------------------

    crop_label_path = os.path.join(
        HEAD_LABEL_DIR,
        os.path.splitext(
            crop_filename
        )[0] + ".txt"
    )

    head_labels = read_yolo_labels(
        crop_label_path
    )

    # --------------------------------------------------------
    # 映射
    # --------------------------------------------------------

    mapped_this_image = 0

    for (
        class_id,
        cx,
        cy,
        w,
        h
    ) in head_labels:

        # ----------------------------------------------------
        # 不管裁剪图里的 class 是多少
        #
        # 最终统一为 Head = 1
        # ----------------------------------------------------

        (
            crop_x1,
            crop_y1,
            crop_x2,
            crop_y2
        ) = yolo_to_xyxy(
            cx,
            cy,
            w,
            h,
            crop_width,
            crop_height
        )

        # ----------------------------------------------------
        # 裁剪图坐标
        #
        # ->
        # 人物框内部相对比例
        # ----------------------------------------------------

        relative_x1 = (
            crop_x1 /
            crop_width
        )

        relative_y1 = (
            crop_y1 /
            crop_height
        )

        relative_x2 = (
            crop_x2 /
            crop_width
        )

        relative_y2 = (
            crop_y2 /
            crop_height
        )

        # ----------------------------------------------------
        # 映射回原图
        # ----------------------------------------------------

        original_x1 = (
            person_x1 +
            relative_x1 *
            person_width
        )

        original_y1 = (
            person_y1 +
            relative_y1 *
            person_height
        )

        original_x2 = (
            person_x1 +
            relative_x2 *
            person_width
        )

        original_y2 = (
            person_y1 +
            relative_y2 *
            person_height
        )

        box = (
            original_x1,
            original_y1,
            original_x2,
            original_y2
        )

        # ----------------------------------------------------
        # 保存到汇总
        # ----------------------------------------------------

        if original_stem not in all_head_boxes:

            all_head_boxes[
                original_stem
            ] = []

        all_head_boxes[
            original_stem
        ].append(
            box
        )

        mapped_this_image += 1
        head_count += 1

    if mapped_this_image > 0:

        mapped_count += 1

        print(
            f"[OK] {crop_filename:30s} "
            f"person={person_index:<3d} "
            f"head={mapped_this_image}"
        )


# ============================================================
# 保存
# ============================================================

total_before_dedup = 0
total_after_dedup = 0
total_duplicates = 0

output_files = 0


print()

print(
    "=" * 80
)

print(
    "开始生成新的标签"
)

print(
    "=" * 80
)


# ============================================================
# 遍历所有原图
#
# 这里非常重要：
#
# 即使某张图没有 Head
# 也复制原来的 Person 标签
#
# 所以不会丢失原始 person 标注
# ============================================================

original_label_files = [
    f
    for f in os.listdir(
        ORIGINAL_LABEL_DIR
    )
    if f.lower().endswith(".txt")
]

original_label_files.sort()


for label_filename in original_label_files:

    original_stem = os.path.splitext(
        label_filename
    )[0]

    original_label_path = os.path.join(
        ORIGINAL_LABEL_DIR,
        label_filename
    )

    output_path = os.path.join(
        OUTPUT_LABEL_DIR,
        label_filename
    )

    # --------------------------------------------------------
    # 读取原始标签
    #
    # 注意：
    # 只读，不修改
    # --------------------------------------------------------

    original_labels = read_yolo_labels(
        original_label_path
    )

    # --------------------------------------------------------
    # 统计 Person
    # --------------------------------------------------------

    person_labels = []

    for label in original_labels:

        if label[0] == PERSON_CLASS_ID:

            person_labels.append(
                label
            )

    # --------------------------------------------------------
    # 获取 Head
    # --------------------------------------------------------

    head_boxes = all_head_boxes.get(
        original_stem,
        []
    )

    total_before_dedup += len(
        head_boxes
    )

    # --------------------------------------------------------
    # 去重
    # --------------------------------------------------------

    unique_head_boxes = (
        remove_duplicates(
            head_boxes
        )
    )

    total_after_dedup += len(
        unique_head_boxes
    )

    removed = (
        len(head_boxes) -
        len(unique_head_boxes)
    )

    total_duplicates += removed

    # --------------------------------------------------------
    # 生成最终标签
    #
    # 先放原来的 Person
    # 再放 Head
    # --------------------------------------------------------

    output_lines = []

    # ========================================================
    # Person
    # ========================================================

    for (
        class_id,
        cx,
        cy,
        w,
        h
    ) in original_labels:

        output_lines.append(
            f"{class_id} "
            f"{cx:.6f} "
            f"{cy:.6f} "
            f"{w:.6f} "
            f"{h:.6f}"
        )

    # ========================================================
    # Head
    # ========================================================

    original_image_path = (
        find_original_image(
            original_stem
        )
    )

    if original_image_path is not None:

        image = cv2.imread(
            original_image_path
        )

        if image is not None:

            image_height, image_width = (
                image.shape[:2]
            )

            for box in unique_head_boxes:

                x1, y1, x2, y2 = box

                (
                    cx,
                    cy,
                    w,
                    h
                ) = xyxy_to_yolo(
                    x1,
                    y1,
                    x2,
                    y2,
                    image_width,
                    image_height
                )

                output_lines.append(
                    f"{HEAD_CLASS_ID} "
                    f"{cx:.6f} "
                    f"{cy:.6f} "
                    f"{w:.6f} "
                    f"{h:.6f}"
                )

    # --------------------------------------------------------
    # 写入新的标签
    #
    # 只写 OUTPUT_LABEL_DIR
    #
    # ORIGINAL_LABEL_DIR 完全不动
    # --------------------------------------------------------

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        if output_lines:

            f.write(
                "\n".join(
                    output_lines
                )
            )

            f.write("\n")

    output_files += 1

    # --------------------------------------------------------
    # 输出
    # --------------------------------------------------------

    if len(head_boxes) > 0:

        print(
            f"{original_stem:30s} "
            f"Person={len(person_labels):2d} "
            f"Head={len(head_boxes):3d} -> "
            f"{len(unique_head_boxes):3d} "
            f"删除={removed:2d}"
        )


# ============================================================
# 完成
# ============================================================

print()

print(
    "=" * 80
)

print(
    "全部完成"
)

print(
    "=" * 80
)

print(
    f"Head 图片数量:        {len(head_images)}"
)

print(
    f"成功映射图片:         {mapped_count}"
)

print(
    f"映射失败:             {failed_count}"
)

print(
    f"原始 Person 数量:     {total_person_boxes}"
)

print(
    f"Head 原始数量:        {total_before_dedup}"
)

print(
    f"去重删除:             {total_duplicates}"
)

print(
    f"最终 Head 数量:       {total_after_dedup}"
)

print(
    f"输出标签数量:         {output_files}"
)

print()

print(
    "原始标签目录没有被修改："
)

print(
    ORIGINAL_LABEL_DIR
)

print()

print(
    "新的完整标签目录："
)

print(
    OUTPUT_LABEL_DIR
)

print()

print(
    "=" * 80
)