import os
import shutil
import math


# ============================================================
# 配置
# ============================================================

IMAGE_DIR = r"D:\PycharmProjects\computer_controller\make_dataset\train_body_head\images"

LABEL_DIR = r"D:\PycharmProjects\computer_controller\make_dataset\train_body_head\labels"


# ============================================================
# 是否创建备份
#
# True：
#     第一次运行时，将原 labels 备份到 backup_labels
#
# False：
#     不备份
# ============================================================

CREATE_BACKUP = True


BACKUP_DIR = os.path.join(
    LABEL_DIR,
    "backup_labels"
)


# ============================================================
# 类别
#
# 0 = body
# 1 = head
# ============================================================

CLASS_NAMES = {
    0: "body",
    1: "head",
}


# ============================================================
# 去重参数
# ============================================================

# ------------------------------------------------------------
# 非常保守
#
# 两个同类别框 IoU 必须达到这个值，
# 才有资格被认为是重复框
# ------------------------------------------------------------

DUPLICATE_IOU = 0.4


# ------------------------------------------------------------
# 中心距离阈值
#
# 中心点距离 / 较大框对角线
#
# 越小越严格
#
# 0.15 = 中心必须非常接近
# ------------------------------------------------------------

MAX_CENTER_DISTANCE_RATIO = 0.4


# ------------------------------------------------------------
# 面积比例
#
# 两个框面积必须比较接近
#
# 例如：
#
# 0.70
#
# 表示：
#
# 小框面积 / 大框面积 >= 0.70
#
# 防止把一个大 body 和一个很小的异常框误认为重复
# ------------------------------------------------------------

MIN_AREA_RATIO = 0.30


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
# IoU
# ============================================================

def calculate_iou(box1, box2):

    """
    box:
        [class_id, cx, cy, w, h]
    """

    _, cx1, cy1, w1, h1 = box1
    _, cx2, cy2, w2, h2 = box2


    # ========================================================
    # 转 XYXY
    # ========================================================

    x1_min = cx1 - w1 / 2
    y1_min = cy1 - h1 / 2

    x1_max = cx1 + w1 / 2
    y1_max = cy1 + h1 / 2


    x2_min = cx2 - w2 / 2
    y2_min = cy2 - h2 / 2

    x2_max = cx2 + w2 / 2
    y2_max = cy2 + h2 / 2


    # ========================================================
    # 交集
    # ========================================================

    inter_x1 = max(
        x1_min,
        x2_min
    )

    inter_y1 = max(
        y1_min,
        y2_min
    )

    inter_x2 = min(
        x1_max,
        x2_max
    )

    inter_y2 = min(
        y1_max,
        y2_max
    )


    inter_w = max(
        0.0,
        inter_x2 - inter_x1
    )

    inter_h = max(
        0.0,
        inter_y2 - inter_y1
    )


    intersection = (
        inter_w *
        inter_h
    )


    # ========================================================
    # 面积
    # ========================================================

    area1 = (
        w1 *
        h1
    )

    area2 = (
        w2 *
        h2
    )


    # ========================================================
    # IoU
    # ========================================================

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
# 中心距离
# ============================================================

def calculate_center_distance(box1, box2):

    _, cx1, cy1, w1, h1 = box1
    _, cx2, cy2, w2, h2 = box2


    distance = math.sqrt(
        (cx1 - cx2) ** 2 +
        (cy1 - cy2) ** 2
    )


    # 使用较大的框对角线作为尺度
    diagonal = max(
        math.sqrt(
            w1 ** 2 +
            h1 ** 2
        ),
        math.sqrt(
            w2 ** 2 +
            h2 ** 2
        )
    )


    if diagonal <= 0:

        return 999.0


    return (
        distance /
        diagonal
    )


# ============================================================
# 面积比例
# ============================================================

def calculate_area_ratio(box1, box2):

    _, _, _, w1, h1 = box1
    _, _, _, w2, h2 = box2


    area1 = w1 * h1
    area2 = w2 * h2


    if area1 <= 0 or area2 <= 0:

        return 0.0


    small_area = min(
        area1,
        area2
    )

    large_area = max(
        area1,
        area2
    )


    return (
        small_area /
        large_area
    )


# ============================================================
# 判断是否属于重复框
# ============================================================

def is_duplicate(
    box1,
    box2
):

    # ========================================================
    # 类别必须一样
    #
    # 这是最重要的一层保护
    #
    # body 和 head 永远不会互相删除
    # ========================================================

    if box1[0] != box2[0]:

        return False


    # ========================================================
    # IoU
    # ========================================================

    iou = calculate_iou(
        box1,
        box2
    )


    if iou < DUPLICATE_IOU:

        return False


    # ========================================================
    # 中心距离
    # ========================================================

    center_distance = calculate_center_distance(
        box1,
        box2
    )


    if (
        center_distance
        >
        MAX_CENTER_DISTANCE_RATIO
    ):

        return False


    # ========================================================
    # 面积比例
    # ========================================================

    area_ratio = calculate_area_ratio(
        box1,
        box2
    )


    if area_ratio < MIN_AREA_RATIO:

        return False


    # ========================================================
    # 三个条件全部满足
    #
    # 才认为是重复
    # ========================================================

    return True


# ============================================================
# 读取标签
# ============================================================

def read_labels(label_path):

    labels = []


    if not os.path.exists(label_path):

        return labels


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


            if len(parts) != 5:

                # 非标准标签
                # 保留
                labels.append({
                    "raw": line,
                    "box": None
                })

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


                box = [
                    class_id,
                    cx,
                    cy,
                    w,
                    h
                ]


                labels.append({
                    "raw": line,
                    "box": box
                })


            except ValueError:

                # 解析失败
                # 原样保留
                labels.append({
                    "raw": line,
                    "box": None
                })


    return labels


# ============================================================
# 创建备份
# ============================================================

if CREATE_BACKUP:

    os.makedirs(
        BACKUP_DIR,
        exist_ok=True
    )


# ============================================================
# 获取标签文件
# ============================================================

label_files = [
    f
    for f in os.listdir(LABEL_DIR)
    if f.lower().endswith(".txt")
]


label_files.sort()


# ============================================================
# 统计
# ============================================================

total_files = 0

files_changed = 0

files_unchanged = 0


total_boxes_before = 0

total_boxes_after = 0

total_removed = 0


removed_body = 0

removed_head = 0


# ============================================================
# 开始
# ============================================================

print()
print("=" * 80)
print("开始轻度去除重复 YOLO 标注")
print("=" * 80)

print(
    f"标签目录: {LABEL_DIR}"
)

print()

print("类别：")

for class_id, class_name in CLASS_NAMES.items():

    print(
        f"  {class_id} = {class_name}"
    )

print()

print(
    f"IoU 阈值: "
    f"{DUPLICATE_IOU}"
)

print(
    f"中心距离阈值: "
    f"{MAX_CENTER_DISTANCE_RATIO}"
)

print(
    f"面积比例阈值: "
    f"{MIN_AREA_RATIO}"
)

print()

print(
    "注意："
)

print(
    "  body 和 head 永远不会互相去重"
)

print(
    "  只有同类别高度重叠的框才可能被删除"
)

print(
    "  原始标签不会被重新生成"
)

print("=" * 80)


# ============================================================
# 处理每一个标签文件
# ============================================================

for file_index, label_file in enumerate(
    label_files,
    start=1
):

    total_files += 1


    label_path = os.path.join(
        LABEL_DIR,
        label_file
    )


    # ========================================================
    # 读取
    # ========================================================

    labels = read_labels(
        label_path
    )


    total_boxes_before += len(
        labels
    )


    # ========================================================
    # 备份
    # ========================================================

    if CREATE_BACKUP:

        backup_path = os.path.join(
            BACKUP_DIR,
            label_file
        )


        # 只在不存在时备份
        #
        # 防止第二次运行把第一次的原始数据覆盖掉

        if not os.path.exists(
            backup_path
        ):

            shutil.copy2(
                label_path,
                backup_path
            )


    # ========================================================
    # 删除标记
    # ========================================================

    removed = set()


    # ========================================================
    # 两两比较
    # ========================================================

    for i in range(
        len(labels)
    ):

        # 已经删除
        if i in removed:

            continue


        box_i = labels[i]["box"]


        # 无法解析的标签
        #
        # 不参与去重
        if box_i is None:

            continue


        for j in range(
            i + 1,
            len(labels)
        ):

            if j in removed:

                continue


            box_j = labels[j]["box"]


            if box_j is None:

                continue


            # =================================================
            # 判断重复
            # =================================================

            if not is_duplicate(
                box_i,
                box_j
            ):

                continue


            # =================================================
            # 两个框被认为是重复
            #
            # 哪一个删除？
            #
            # 因为没有 confidence，
            # 所以采用非常保守的策略：
            #
            # 保留面积稍大的那个。
            #
            # 如果面积非常接近：
            # 保留前面的。
            # =================================================

            area_i = (
                box_i[3] *
                box_i[4]
            )

            area_j = (
                box_j[3] *
                box_j[4]
            )


            if area_j > area_i:

                remove_index = i

            else:

                remove_index = j


            # =================================================
            # 标记删除
            # =================================================

            removed.add(
                remove_index
            )


            # =================================================
            # 统计
            # =================================================

            removed_class = labels[
                remove_index
            ]["box"][0]


            total_removed += 1


            if removed_class == 0:

                removed_body += 1

            elif removed_class == 1:

                removed_head += 1


            # =================================================
            # 如果 i 被删除
            #
            # 当前 i 后面不再继续比较
            # =================================================

            if remove_index == i:

                break


    # ========================================================
    # 生成最终标签
    # ========================================================

    final_labels = []


    for i, item in enumerate(
        labels
    ):

        if i in removed:

            continue


        final_labels.append(
            item["raw"]
        )


    total_boxes_after += len(
        final_labels
    )


    # ========================================================
    # 判断文件是否变化
    # ========================================================

    if len(final_labels) != len(labels):

        files_changed += 1


    else:

        files_unchanged += 1


    # ========================================================
    # 保存
    # ========================================================

    with open(
        label_path,
        "w",
        encoding="utf-8"
    ) as f:

        if final_labels:

            f.write(
                "\n".join(
                    final_labels
                )
            )

            f.write("\n")


    # ========================================================
    # 输出
    # ========================================================

    if len(removed) > 0:

        print(
            f"[{file_index:5d}/{len(label_files)}] "
            f"{label_file:30s} "
            f"{len(labels):3d} → "
            f"{len(final_labels):3d} "
            f"删除={len(removed):2d}"
        )

    else:

        print(
            f"[{file_index:5d}/{len(label_files)}] "
            f"{label_file:30s} "
            f"{len(labels):3d} → "
            f"{len(final_labels):3d} "
            f"无变化"
        )


# ============================================================
# 完成
# ============================================================

print()
print("=" * 80)
print("去重完成")
print("=" * 80)

print(
    f"标签文件数量: "
    f"{total_files}"
)

print(
    f"发生变化的文件: "
    f"{files_changed}"
)

print(
    f"没有变化的文件: "
    f"{files_unchanged}"
)

print()

print(
    f"去重前框数量: "
    f"{total_boxes_before}"
)

print(
    f"去重后框数量: "
    f"{total_boxes_after}"
)

print(
    f"总删除框数量: "
    f"{total_removed}"
)

print()

print(
    f"删除 body 重复框: "
    f"{removed_body}"
)

print(
    f"删除 head 重复框: "
    f"{removed_head}"
)

print()

if total_boxes_before > 0:

    reduction = (
        total_removed /
        total_boxes_before
        *
        100
    )

    print(
        f"减少比例: "
        f"{reduction:.2f}%"
    )

print()

if CREATE_BACKUP:

    print(
        f"原始标签备份: "
        f"{BACKUP_DIR}"
    )

print()

print("=" * 80)