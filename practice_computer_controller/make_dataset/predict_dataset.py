from ultralytics import YOLO
import os


# ============================================================
# 配置
# ============================================================

MODEL_PATH = r"D:\PycharmProjects\computer_controller\best.engine"

IMAGE_DIR = r"D:\PycharmProjects\computer_controller\make_dataset\train_body_head\images"

LABEL_DIR = r"D:\PycharmProjects\computer_controller\make_dataset\train_body_head\labels"


# ============================================================
# 前多少张图片保留原始标签
#
# 前 KEEP_FIRST_N 张：
#     完全不修改
#
# 从 KEEP_FIRST_N + 1 张开始：
#     原始标签 + 模型预测
#     但会对模型预测进行轻度去重
# ============================================================

KEEP_FIRST_N = 100


# ============================================================
# 类别
#
# 必须和训练时 data.yaml 的类别顺序一致
#
# 0 = person
# 1 = head
# ============================================================

CLASS_NAMES = {
    0: "person",
    1: "head",
}


# ============================================================
# 推理参数
# ============================================================

CONF = 0.25

IOU = 0.8

IMGSZ = 640


# ============================================================
# 去重参数
#
# 注意：
#
# 这个值故意设置得比较高。
#
# IoU >= 0.80
#     才认为预测框和原标注是重复的
#
# 这样可以尽量避免误删预测框。
# ============================================================

DUPLICATE_IOU = 0.80


# ============================================================
# 是否要求类别相同
#
# True：
#     只有同类别才会去重
#
# 例如：
#
# person vs person → 可以去重
# head vs head     → 可以去重
#
# person vs head   → 永远不去重
# ============================================================

DUPLICATE_SAME_CLASS_ONLY = True


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
# 创建 labels 文件夹
# ============================================================

os.makedirs(
    LABEL_DIR,
    exist_ok=True
)


# ============================================================
# IoU 函数
#
# 输入：
#     两个 YOLO 格式框
#
# 格式：
#     [class_id, cx, cy, w, h]
#
# 返回：
#     IoU
# ============================================================

def calculate_iou(
    box1,
    box2
):

    # --------------------------------------------------------
    # box1
    # --------------------------------------------------------

    _, cx1, cy1, w1, h1 = box1

    # --------------------------------------------------------
    # box2
    # --------------------------------------------------------

    _, cx2, cy2, w2, h2 = box2


    # ========================================================
    # 转 XYXY
    # ========================================================

    box1_x1 = cx1 - w1 / 2
    box1_y1 = cy1 - h1 / 2

    box1_x2 = cx1 + w1 / 2
    box1_y2 = cy1 + h1 / 2


    box2_x1 = cx2 - w2 / 2
    box2_y1 = cy2 - h2 / 2

    box2_x2 = cx2 + w2 / 2
    box2_y2 = cy2 + h2 / 2


    # ========================================================
    # 交集
    # ========================================================

    intersection_x1 = max(
        box1_x1,
        box2_x1
    )

    intersection_y1 = max(
        box1_y1,
        box2_y1
    )

    intersection_x2 = min(
        box1_x2,
        box2_x2
    )

    intersection_y2 = min(
        box1_y2,
        box2_y2
    )


    intersection_width = max(
        0.0,
        intersection_x2 - intersection_x1
    )

    intersection_height = max(
        0.0,
        intersection_y2 - intersection_y1
    )


    intersection_area = (
        intersection_width
        *
        intersection_height
    )


    # ========================================================
    # 两个框面积
    # ========================================================

    area1 = (
        w1
        *
        h1
    )

    area2 = (
        w2
        *
        h2
    )


    # ========================================================
    # 并集
    # ========================================================

    union_area = (
        area1
        +
        area2
        -
        intersection_area
    )


    if union_area <= 0:

        return 0.0


    return (
        intersection_area
        /
        union_area
    )


# ============================================================
# 加载 YOLO
# ============================================================

print("=" * 80)
print("加载 YOLO 模型")
print("=" * 80)

print(
    f"模型: {MODEL_PATH}"
)

model = YOLO(
    MODEL_PATH
)


# ============================================================
# 显示模型类别
# ============================================================

print()
print("=" * 80)
print("模型类别")
print("=" * 80)

print(
    model.names
)

print("=" * 80)


# ============================================================
# 检查类别
# ============================================================

model_class_ids = set(
    int(k)
    for k in model.names.keys()
)

config_class_ids = set(
    CLASS_NAMES.keys()
)


if model_class_ids != config_class_ids:

    print()
    print("警告：模型类别和 CLASS_NAMES 不完全一致！")
    print()

    print(
        f"模型类别: {model.names}"
    )

    print(
        f"配置类别: {CLASS_NAMES}"
    )

    print()


# ============================================================
# 获取图片
# ============================================================

image_files = [
    f
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(
        IMAGE_EXTENSIONS
    )
]


# 按文件名排序
image_files.sort()


# ============================================================
# 基本信息
# ============================================================

print()
print("=" * 80)
print("开始生成 YOLO 数据集")
print("=" * 80)

print(
    f"图片数量: {len(image_files)}"
)

print(
    f"模型: {MODEL_PATH}"
)

print(
    f"图片目录: {IMAGE_DIR}"
)

print(
    f"标签目录: {LABEL_DIR}"
)

print()

print(
    f"前 {KEEP_FIRST_N} 张:"
)

print(
    "    完全保留原始标签"
)

print()

print(
    f"第 {KEEP_FIRST_N + 1} 张开始:"
)

print(
    "    原始标签 + 模型预测"
)

print(
    f"    重复判断 IoU >= {DUPLICATE_IOU}"
)

print(
    "    原始标签永远不会被删除"
)

print(
    "    去重只针对模型新增预测"
)

print()

print("类别：")

for class_id, class_name in CLASS_NAMES.items():

    print(
        f"  {class_id}: {class_name}"
    )

print("=" * 80)


# ============================================================
# 推理
# ============================================================

print()
print("开始模型推理...")
print()


results = model.predict(
    source=IMAGE_DIR,
    conf=CONF,
    iou=IOU,
    imgsz=IMGSZ,
    save=False,
    show=False,
    verbose=False,
)


# ============================================================
# 统计
# ============================================================

total_images = 0

kept_original_images = 0

auto_generated_images = 0

images_with_detection = 0

images_without_detection = 0


# ============================================================
# 原始框数量
# ============================================================

original_box_count = 0


# ============================================================
# 模型预测框数量
# ============================================================

prediction_box_count = 0


# ============================================================
# 被认为重复而丢弃的预测框
# ============================================================

duplicate_box_count = 0


# ============================================================
# 最终新增框数量
# ============================================================

added_prediction_box_count = 0


# ============================================================
# 最终框数量
# ============================================================

final_box_count = 0


# ============================================================
# 每个类别：
#
# 模型原始预测数量
# ============================================================

class_prediction_counts = {
    class_id: 0
    for class_id in CLASS_NAMES
}


# ============================================================
# 每个类别：
#
# 被去重掉的预测数量
# ============================================================

class_duplicate_counts = {
    class_id: 0
    for class_id in CLASS_NAMES
}


# ============================================================
# 每个类别：
#
# 最终新增预测数量
# ============================================================

class_added_counts = {
    class_id: 0
    for class_id in CLASS_NAMES
}


# ============================================================
# 每个类别：
#
# 出现过的图片数量
# ============================================================

class_image_counts = {
    class_id: 0
    for class_id in CLASS_NAMES
}


# ============================================================
# 处理结果
# ============================================================

for result in results:

    total_images += 1


    # --------------------------------------------------------
    # 当前图片序号
    # --------------------------------------------------------

    image_index = total_images


    # --------------------------------------------------------
    # 图片文件名
    # --------------------------------------------------------

    image_path = result.path

    image_name = os.path.basename(
        image_path
    )

    stem = os.path.splitext(
        image_name
    )[0]


    # --------------------------------------------------------
    # 标签路径
    # --------------------------------------------------------

    label_path = os.path.join(
        LABEL_DIR,
        stem + ".txt"
    )


    # ========================================================
    # 前 KEEP_FIRST_N 张
    #
    # 完全不修改
    # ========================================================

    if image_index <= KEEP_FIRST_N:

        kept_original_images += 1

        print(
            f"[{image_index:4d}/{len(image_files)}] "
            f"{image_name:25s} "
            f"保留原始标签"
        )

        continue


    # ========================================================
    # 后续图片
    # ========================================================

    auto_generated_images += 1


    # ========================================================
    # 读取原始标签
    # ========================================================

    original_labels = []


    if os.path.exists(label_path):

        with open(
            label_path,
            "r",
            encoding="utf-8"
        ) as f:

            for line in f:

                line = line.strip()

                if not line:
                    continue

                original_labels.append(
                    line
                )


    # --------------------------------------------------------
    # 原始标签统计
    # --------------------------------------------------------

    original_box_count += len(
        original_labels
    )


    # ========================================================
    # 将原始标签转换成数组
    #
    # 用于后面 IoU 判断
    # ========================================================

    original_boxes = []


    for label in original_labels:

        parts = label.split()

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


        if class_id not in CLASS_NAMES:

            continue


        original_boxes.append(
            [
                class_id,
                cx,
                cy,
                w,
                h
            ]
        )


    # ========================================================
    # 原始图片尺寸
    # ========================================================

    image_height, image_width = result.orig_shape


    # ========================================================
    # 模型预测标签
    # ========================================================

    prediction_labels = []


    # ========================================================
    # 没有检测结果
    # ========================================================

    if (
        result.boxes is None
        or len(result.boxes) == 0
    ):

        images_without_detection += 1


        # ----------------------------------------------------
        # 原始标签全部保留
        # ----------------------------------------------------

        final_labels = original_labels


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


        final_box_count += len(
            final_labels
        )


        print(
            f"[{image_index:4d}/{len(image_files)}] "
            f"{image_name:25s} "
            f"预测=0 "
            f"原标注={len(original_labels):3d} "
            f"→ 全部保留"
        )

        continue


    # ========================================================
    # 当前图片最终新增的类别
    # ========================================================

    current_added_classes = set()


    # ========================================================
    # 遍历模型检测框
    # ========================================================

    for box in result.boxes:

        # ----------------------------------------------------
        # 类别
        # ----------------------------------------------------

        class_id = int(
            box.cls.item()
        )


        # ----------------------------------------------------
        # 只保留配置中的类别
        # ----------------------------------------------------

        if class_id not in CLASS_NAMES:

            continue


        # ====================================================
        # XYXY
        # ====================================================

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
        )


        # ====================================================
        # 转换成 YOLO 格式
        # ====================================================

        center_x = (
            (x1 + x2) / 2
        )

        center_y = (
            (y1 + y2) / 2
        )

        box_width = (
            x2 - x1
        )

        box_height = (
            y2 - y1
        )


        # ====================================================
        # 归一化
        # ====================================================

        center_x /= image_width

        center_y /= image_height

        box_width /= image_width

        box_height /= image_height


        # ====================================================
        # 限制范围
        # ====================================================

        center_x = max(
            0.0,
            min(1.0, center_x)
        )

        center_y = max(
            0.0,
            min(1.0, center_y)
        )

        box_width = max(
            0.0,
            min(1.0, box_width)
        )

        box_height = max(
            0.0,
            min(1.0, box_height)
        )


        # ====================================================
        # 当前预测框
        # ====================================================

        prediction_box = [
            class_id,
            center_x,
            center_y,
            box_width,
            box_height
        ]


        # ====================================================
        # 统计模型原始预测
        # ====================================================

        prediction_box_count += 1

        class_prediction_counts[
            class_id
        ] += 1


        # ====================================================
        # 和原始标签比较
        #
        # 注意：
        #
        # 只要没有达到 DUPLICATE_IOU，
        # 就认为不是重复。
        #
        # 因此这个去重非常保守。
        # ========================================================

        is_duplicate = False


        for original_box in original_boxes:

            # ------------------------------------------------
            # 不同类别
            #
            # 永远不认为重复
            # ------------------------------------------------

            if DUPLICATE_SAME_CLASS_ONLY:

                if (
                    class_id
                    !=
                    original_box[0]
                ):

                    continue


            # ------------------------------------------------
            # 计算 IoU
            # ------------------------------------------------

            iou = calculate_iou(
                prediction_box,
                original_box
            )


            # ------------------------------------------------
            # 高 IoU
            #
            # 才认为重复
            # ------------------------------------------------

            if iou >= DUPLICATE_IOU:

                is_duplicate = True

                break


        # ====================================================
        # 如果重复
        # ====================================================

        if is_duplicate:

            duplicate_box_count += 1

            class_duplicate_counts[
                class_id
            ] += 1

            continue


        # ====================================================
        # 不是重复
        #
        # 加入最终标签
        # ====================================================

        prediction_labels.append(
            f"{class_id} "
            f"{center_x:.6f} "
            f"{center_y:.6f} "
            f"{box_width:.6f} "
            f"{box_height:.6f}"
        )


        # ----------------------------------------------------
        # 统计新增
        # ----------------------------------------------------

        added_prediction_box_count += 1

        class_added_counts[
            class_id
        ] += 1


        current_added_classes.add(
            class_id
        )


    # ========================================================
    # 最终标签
    #
    # 原始标签永远在前面
    #
    # 预测标签全部追加到后面
    # ========================================================

    final_labels = (
        original_labels
        +
        prediction_labels
    )


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


    # ========================================================
    # 最终统计
    # ========================================================

    final_box_count += len(
        final_labels
    )


    if prediction_labels:

        images_with_detection += 1

    else:

        images_without_detection += 1


    # ========================================================
    # 统计每个类别出现过的图片数量
    #
    # 注意：
    # 这里统计的是新增预测类别
    # ========================================================

    for class_id in current_added_classes:

        class_image_counts[
            class_id
        ] += 1


    # ========================================================
    # 输出类别信息
    # ========================================================

    class_text = []


    for class_id in CLASS_NAMES:

        prediction_count = sum(
            1
            for label in prediction_labels
            if label.startswith(
                f"{class_id} "
            )
        )


        duplicate_count = (
            class_duplicate_counts[
                class_id
            ]
        )


        if prediction_count > 0:

            class_text.append(
                f"{CLASS_NAMES[class_id]}+{prediction_count}"
            )


    if class_text:

        detection_text = ", ".join(
            class_text
        )

    else:

        detection_text = "无新增"


    # ========================================================
    # 输出
    # ========================================================

    print(
        f"[{image_index:4d}/{len(image_files)}] "
        f"{image_name:25s} "
        f"原标注={len(original_labels):3d} "
        f"预测={len(result.boxes):3d} "
        f"去重={len(result.boxes) - len(prediction_labels):3d} "
        f"新增={len(prediction_labels):3d} "
        f"→ {detection_text}"
    )


# ============================================================
# 完成
# ============================================================

print()
print("=" * 80)
print("完成")
print("=" * 80)

print(
    f"总图片: "
    f"{total_images}"
)

print(
    f"保留原始标签图片: "
    f"{kept_original_images}"
)

print(
    f"执行自动追加图片: "
    f"{auto_generated_images}"
)

print()

print(
    f"原始标签框总数: "
    f"{original_box_count}"
)

print(
    f"模型原始预测框总数: "
    f"{prediction_box_count}"
)

print(
    f"判定为重复的预测框: "
    f"{duplicate_box_count}"
)

print(
    f"最终新增预测框: "
    f"{added_prediction_box_count}"
)

print(
    f"最终标签框总数: "
    f"{final_box_count}"
)

print()

print(
    f"有新增预测的图片: "
    f"{images_with_detection}"
)

print(
    f"没有新增预测的图片: "
    f"{images_without_detection}"
)


# ============================================================
# 模型原始预测统计
# ============================================================

print()
print("=" * 80)
print("模型原始预测统计")
print("=" * 80)

for class_id, class_name in CLASS_NAMES.items():

    print(
        f"Class {class_id} "
        f"({class_name:10s}) : "
        f"{class_prediction_counts[class_id]:6d}"
    )


# ============================================================
# 去重统计
# ============================================================

print()
print("=" * 80)
print("被去重的预测统计")
print("=" * 80)

for class_id, class_name in CLASS_NAMES.items():

    print(
        f"Class {class_id} "
        f"({class_name:10s}) : "
        f"{class_duplicate_counts[class_id]:6d}"
    )


# ============================================================
# 最终新增统计
# ============================================================

print()
print("=" * 80)
print("最终新增预测统计")
print("=" * 80)

for class_id, class_name in CLASS_NAMES.items():

    print(
        f"Class {class_id} "
        f"({class_name:10s}) : "
        f"{class_added_counts[class_id]:6d}"
    )

    print(
        f"    出现图片数 : "
        f"{class_image_counts[class_id]:6d}"
    )


print("=" * 80)

print()

print(
    f"Labels: {LABEL_DIR}"
)

print("=" * 80)