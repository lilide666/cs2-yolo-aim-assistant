import cv2
import os
import shutil
import numpy as np


# ============================================================
# 配置
# ============================================================

# 图片目录
IMAGE_DIR = r"D:\PycharmProjects\computer_controller\make_different_back\10_different_back_dataset\images\train"

# 原始 YOLO 标签
LABEL_DIR = r"D:\PycharmProjects\computer_controller\make_different_back\10_different_back_dataset\labels\train"

# 修改后的 YOLO 标签
#
# 注意：
# 你现在这里和 LABEL_DIR 相同
# 所以会直接修改原始 labels
#
NEW_LABEL_DIR = LABEL_DIR


# ============================================================
# 类别配置
# ============================================================

CLASS_NAMES = {
    0: "person",
    1: "head",
    2: "?",
    3: "?",
    4: "?",
    5: "?",
}


# ============================================================
# 默认新建框类别
# ============================================================

CURRENT_CLASS_ID = 1


# ============================================================
# 默认显示类别
#
# 1 = 只显示 head
# 0 = 只显示 person
# None = 显示全部
# ============================================================

DISPLAY_CLASS_ID = 1


# ============================================================
# 界面配置
# ============================================================

WINDOW_NAME = "YOLO Annotation Tool"

WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 900

# 最小框尺寸
MIN_BOX_SIZE = 5

# 最小缩放
MIN_SCALE = 0.10

# 最大缩放
MAX_SCALE = 8.0

# 滚轮每次缩放倍率
ZOOM_FACTOR = 1.15


# ============================================================
# 跳转窗口配置
# ============================================================

JUMP_WINDOW_NAME = "Jump to Image"

JUMP_WINDOW_WIDTH = 520
JUMP_WINDOW_HEIGHT = 220

# 最大输入数字长度
JUMP_MAX_DIGITS = 8

# 当前跳转输入
jump_input = ""

# 跳转窗口是否打开
jump_window_active = False


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
# 创建输出目录
# ============================================================

os.makedirs(
    NEW_LABEL_DIR,
    exist_ok=True
)


# ============================================================
# 获取图片
# ============================================================

image_files = sorted([
    f
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(IMAGE_EXTENSIONS)
])


if not image_files:

    print("=" * 70)
    print("错误：没有找到图片")
    print("=" * 70)

    input("按回车退出...")
    exit()


print("=" * 70)
print(f"找到图片：{len(image_files)} 张")
print("=" * 70)


# ============================================================
# 全局状态
# ============================================================

current_index = 0

image = None

image_width = 0
image_height = 0


# ============================================================
# 框
# ============================================================

boxes = []

# 刚进入图片时的状态
original_boxes = []

# 当前选中的框
selected_box_index = -1


# ============================================================
# 鼠标状态
# ============================================================

drawing = False
deleting = False
panning = False

start_x = 0
start_y = 0

current_mouse_x = 0
current_mouse_y = 0

temp_box = None


# ============================================================
# 缩放 / 平移
# ============================================================

display_scale = 1.0

display_offset_x = 0
display_offset_y = 0

display_width = 0
display_height = 0


# 中键拖动
pan_start_mouse_x = 0
pan_start_mouse_y = 0

pan_start_offset_x = 0
pan_start_offset_y = 0


# ============================================================
# 是否修改
# ============================================================

modified = False


# ============================================================
# 当前鼠标所在图片坐标
# ============================================================

last_image_x = 0
last_image_y = 0


# ============================================================
# 获取类别名称
# ============================================================

def get_class_name(class_id):

    return CLASS_NAMES.get(
        class_id,
        f"class_{class_id}"
    )


# ============================================================
# 获取标签路径
# ============================================================

def get_label_path(
    label_dir,
    index
):

    image_name = image_files[index]

    stem = os.path.splitext(
        image_name
    )[0]

    return os.path.join(
        label_dir,
        stem + ".txt"
    )


# ============================================================
# 读取 YOLO 标签
# ============================================================

def load_labels(path):

    result = []

    if not os.path.exists(path):
        return result

    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            lines = f.readlines()

    except Exception as e:

        print(
            f"读取标签失败：{path}"
        )

        print(e)

        return result


    for line in lines:

        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            continue

        try:

            cls = int(parts[0])

            cx = float(parts[1])
            cy = float(parts[2])

            w = float(parts[3])
            h = float(parts[4])

        except ValueError:

            continue


        # ====================================================
        # YOLO -> XYXY
        # ====================================================

        x1 = int(
            (cx - w / 2)
            * image_width
        )

        y1 = int(
            (cy - h / 2)
            * image_height
        )

        x2 = int(
            (cx + w / 2)
            * image_width
        )

        y2 = int(
            (cy + h / 2)
            * image_height
        )


        # ====================================================
        # 限制到图片范围
        # ====================================================

        x1 = max(
            0,
            min(
                image_width - 1,
                x1
            )
        )

        y1 = max(
            0,
            min(
                image_height - 1,
                y1
            )
        )

        x2 = max(
            0,
            min(
                image_width - 1,
                x2
            )
        )

        y2 = max(
            0,
            min(
                image_height - 1,
                y2
            )
        )


        # ====================================================
        # 防止反向
        # ====================================================

        if x1 > x2:

            x1, x2 = x2, x1

        if y1 > y2:

            y1, y2 = y2, y1


        result.append({

            "class": cls,

            "x1": x1,
            "y1": y1,

            "x2": x2,
            "y2": y2

        })


    return result


# ============================================================
# 保存 YOLO 标签
# ============================================================

def save_labels():

    global modified

    label_path = get_label_path(
        NEW_LABEL_DIR,
        current_index
    )


    try:

        with open(
            label_path,
            "w",
            encoding="utf-8"
        ) as f:

            for box in boxes:

                x1 = box["x1"]
                y1 = box["y1"]

                x2 = box["x2"]
                y2 = box["y2"]


                # =================================================
                # XYXY -> YOLO
                # =================================================

                cx = (
                    (x1 + x2) / 2
                ) / image_width

                cy = (
                    (y1 + y2) / 2
                ) / image_height

                w = (
                    x2 - x1
                ) / image_width

                h = (
                    y2 - y1
                ) / image_height


                # =================================================
                # 限制范围
                # =================================================

                cx = max(
                    0.0,
                    min(
                        1.0,
                        cx
                    )
                )

                cy = max(
                    0.0,
                    min(
                        1.0,
                        cy
                    )
                )

                w = max(
                    0.0,
                    min(
                        1.0,
                        w
                    )
                )

                h = max(
                    0.0,
                    min(
                        1.0,
                        h
                    )
                )


                f.write(
                    f"{box['class']} "
                    f"{cx:.6f} "
                    f"{cy:.6f} "
                    f"{w:.6f} "
                    f"{h:.6f}\n"
                )


        modified = False


        print(
            f"[保存] "
            f"{os.path.basename(label_path)} "
            f"Boxes={len(boxes)}"
        )


    except Exception as e:

        print(
            f"[保存失败] {label_path}"
        )

        print(e)


# ============================================================
# 准备标签
# ============================================================

def prepare_label():

    original_path = get_label_path(
        LABEL_DIR,
        current_index
    )

    new_path = get_label_path(
        NEW_LABEL_DIR,
        current_index
    )


    # ========================================================
    # 已经有修改版本
    # ========================================================

    if os.path.exists(new_path):

        return new_path


    # ========================================================
    # 原始标签存在
    # ========================================================

    if os.path.exists(original_path):

        # 如果两个路径相同
        # 不需要复制

        if os.path.abspath(
            original_path
        ) == os.path.abspath(
            new_path
        ):

            return new_path


        try:

            shutil.copy2(
                original_path,
                new_path
            )


            print(
                "[复制原始标签]"
            )

            print(
                f"  FROM: {original_path}"
            )

            print(
                f"  TO:   {new_path}"
            )


        except Exception as e:

            print(
                f"复制失败：{e}"
            )


        return new_path


    # ========================================================
    # 没有标签
    # ========================================================

    print(
        "[无原始标签] 从空白开始"
    )

    return new_path


# ============================================================
# 计算适合窗口的缩放
# ============================================================

def fit_image():

    global display_scale

    global display_offset_x
    global display_offset_y

    global display_width
    global display_height


    try:

        window_width, window_height = (
            cv2.getWindowImageRect(
                WINDOW_NAME
            )[2:4]
        )

    except Exception:

        window_width = WINDOW_WIDTH
        window_height = WINDOW_HEIGHT


    window_width = max(
        100,
        window_width
    )

    window_height = max(
        100,
        window_height
    )


    # ========================================================
    # 顶部信息栏
    # ========================================================

    usable_width = max(
        100,
        window_width
    )

    usable_height = max(
        100,
        window_height - 50
    )


    # ========================================================
    # 计算缩放
    # ========================================================

    scale_x = (
        usable_width
        / image_width
    )

    scale_y = (
        usable_height
        / image_height
    )


    display_scale = min(
        scale_x,
        scale_y
    )


    display_scale = max(
        MIN_SCALE,
        min(
            MAX_SCALE,
            display_scale
        )
    )


    # ========================================================
    # 显示尺寸
    # ========================================================

    display_width = int(
        image_width
        * display_scale
    )

    display_height = int(
        image_height
        * display_scale
    )


    # ========================================================
    # 居中
    # ========================================================

    display_offset_x = (
        window_width
        - display_width
    ) // 2


    display_offset_y = (
        50
        + (
            usable_height
            - display_height
        ) // 2
    )


# ============================================================
# 加载图片
# ============================================================

def load_image(index):

    global image

    global image_width
    global image_height

    global boxes
    global original_boxes

    global modified

    global selected_box_index

    global temp_box

    global drawing
    global deleting
    global panning


    image_path = os.path.join(
        IMAGE_DIR,
        image_files[index]
    )


    print()
    print("=" * 70)

    print(
        f"[图片] "
        f"{index + 1}/{len(image_files)}"
    )

    print(
        image_files[index]
    )


    # ========================================================
    # 读取图片
    # ========================================================

    image = cv2.imread(
        image_path
    )


    if image is None:

        print(
            f"无法读取：{image_path}"
        )

        return False


    image_height, image_width = (
        image.shape[:2]
    )


    # ========================================================
    # 准备标签
    # ========================================================

    label_path = prepare_label()


    # ========================================================
    # 读取标签
    # ========================================================

    boxes = load_labels(
        label_path
    )


    # ========================================================
    # 保存进入当前图片时的状态
    # ========================================================

    original_boxes = [
        box.copy()
        for box in boxes
    ]


    selected_box_index = -1

    temp_box = None

    drawing = False
    deleting = False
    panning = False

    modified = False


    # ========================================================
    # 恢复适合窗口
    # ========================================================

    fit_image()


    print(
        f"尺寸："
        f"{image_width} x "
        f"{image_height}"
    )

    print(
        f"Boxes：{len(boxes)}"
    )


    if DISPLAY_CLASS_ID is None:

        print(
            "显示类别：ALL"
        )

    else:

        print(
            "显示类别："
            f"{DISPLAY_CLASS_ID}:"
            f"{get_class_name(DISPLAY_CLASS_ID)}"
        )


    return True


# ============================================================
# 跳转到指定图片
# ============================================================

def jump_to_image():

    global current_index
    global jump_input
    global jump_window_active


    # ========================================================
    # 如果已经打开
    # ========================================================

    if jump_window_active:

        return


    jump_window_active = True

    jump_input = ""


    # ========================================================
    # 创建窗口
    # ========================================================

    cv2.namedWindow(
        JUMP_WINDOW_NAME,
        cv2.WINDOW_NORMAL
    )


    cv2.resizeWindow(
        JUMP_WINDOW_NAME,
        JUMP_WINDOW_WIDTH,
        JUMP_WINDOW_HEIGHT
    )


    # ========================================================
    # 尝试设置窗口位置
    # ========================================================

    try:

        cv2.moveWindow(
            JUMP_WINDOW_NAME,
            700,
            350
        )

    except Exception:

        pass


    # ========================================================
    # 输入循环
    # ========================================================

    while True:

        # ----------------------------------------------------
        # 创建背景
        # ----------------------------------------------------

        canvas = np.zeros(
            (
                JUMP_WINDOW_HEIGHT,
                JUMP_WINDOW_WIDTH,
                3
            ),
            dtype=np.uint8
        )


        # ----------------------------------------------------
        # 顶部标题栏
        # ----------------------------------------------------

        cv2.rectangle(
            canvas,
            (0, 0),
            (
                JUMP_WINDOW_WIDTH,
                55
            ),
            (25, 25, 25),
            -1
        )


        cv2.putText(
            canvas,
            "Jump to Image",
            (20, 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # 提示
        # ----------------------------------------------------

        cv2.putText(
            canvas,
            f"Enter image number (1 - {len(image_files)})",
            (20, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 200),
            1,
            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # 输入框
        # ----------------------------------------------------

        input_x1 = 20
        input_y1 = 105

        input_x2 = JUMP_WINDOW_WIDTH - 20
        input_y2 = 155


        cv2.rectangle(
            canvas,
            (input_x1, input_y1),
            (input_x2, input_y2),
            (60, 60, 60),
            -1
        )


        cv2.rectangle(
            canvas,
            (input_x1, input_y1),
            (input_x2, input_y2),
            (0, 200, 255),
            2
        )


        # ----------------------------------------------------
        # 输入文字
        # ----------------------------------------------------

        display_text = jump_input + "_"


        cv2.putText(
            canvas,
            display_text,
            (
                input_x1 + 12,
                input_y1 + 34
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # 底部提示
        # ----------------------------------------------------

        cv2.putText(
            canvas,
            "ENTER: Confirm    ESC: Cancel",
            (
                20,
                190
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1,
            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # 显示
        # ----------------------------------------------------

        cv2.imshow(
            JUMP_WINDOW_NAME,
            canvas
        )


        # ----------------------------------------------------
        # 键盘
        # ----------------------------------------------------

        key = cv2.waitKey(30) & 0xFF


        # ====================================================
        # ESC：取消
        # ====================================================

        if key == 27:

            print(
                "[跳转] 已取消"
            )

            break


        # ====================================================
        # ENTER：确认
        # ====================================================

        elif key in (
            13,
            10
        ):

            if not jump_input:

                print(
                    "[跳转] 请输入图片序号"
                )

                continue


            try:

                target_number = int(
                    jump_input
                )

            except ValueError:

                print(
                    "[跳转] 输入无效"
                )

                jump_input = ""

                continue


            # ------------------------------------------------
            # 检查范围
            # ------------------------------------------------

            if target_number < 1:

                print(
                    "[跳转] 序号不能小于 1"
                )

                jump_input = ""

                continue


            if target_number > len(image_files):

                print(
                    "[跳转] 序号不能大于 "
                    f"{len(image_files)}"
                )

                jump_input = ""

                continue


            # ------------------------------------------------
            # 保存当前修改
            # ------------------------------------------------

            if modified:

                save_labels()


            # ------------------------------------------------
            # 转换 index
            # ------------------------------------------------

            new_index = (
                target_number - 1
            )


            # ------------------------------------------------
            # 执行跳转
            # ------------------------------------------------

            if new_index != current_index:

                current_index = new_index

                if not load_image(
                    current_index
                ):

                    print(
                        "[跳转] 图片加载失败"
                    )

                    break


            print(
                "[跳转] 已跳转到 "
                f"{current_index + 1}/"
                f"{len(image_files)}"
            )

            break


        # ====================================================
        # BACKSPACE：删除
        # ====================================================

        elif key in (
            8,
            127
        ):

            jump_input = (
                jump_input[:-1]
            )


        # ====================================================
        # 数字
        # ====================================================

        elif (
            ord("0")
            <= key
            <= ord("9")
        ):

            if len(jump_input) < JUMP_MAX_DIGITS:

                jump_input += chr(key)


    # ========================================================
    # 关闭窗口
    # ========================================================

    cv2.destroyWindow(
        JUMP_WINDOW_NAME
    )


    jump_window_active = False

    jump_input = ""


# ============================================================
# 原图 -> 显示坐标
# ============================================================

def image_to_display(
    x,
    y
):

    display_x = int(
        x
        * display_scale
        + display_offset_x
    )

    display_y = int(
        y
        * display_scale
        + display_offset_y
    )


    return (
        display_x,
        display_y
    )


# ============================================================
# 显示坐标 -> 原图坐标
#
# 鼠标可以拖出图片
# 坐标会自动限制在图片边缘
# ============================================================

def display_to_image(
    x,
    y
):

    image_x = int(
        (
            x
            - display_offset_x
        )
        / display_scale
    )

    image_y = int(
        (
            y
            - display_offset_y
        )
        / display_scale
    )


    # ========================================================
    # 限制到图片边界
    # ========================================================

    image_x = max(
        0,
        min(
            image_width - 1,
            image_x
        )
    )

    image_y = max(
        0,
        min(
            image_height - 1,
            image_y
        )
    )


    return (
        image_x,
        image_y
    )


# ============================================================
# 判断点是否在框
# ============================================================

def point_inside_box(
    x,
    y,
    box
):

    return (
        box["x1"] <= x <= box["x2"]
        and
        box["y1"] <= y <= box["y2"]
    )


# ============================================================
# 找到鼠标下面的框
# ============================================================

def find_box_at(
    image_x,
    image_y
):

    # 从后往前
    # 后创建的框优先

    for i in range(
        len(boxes) - 1,
        -1,
        -1
    ):

        box = boxes[i]


        # ====================================================
        # 过滤类别
        # ====================================================

        if (
            DISPLAY_CLASS_ID is not None
            and
            box["class"] != DISPLAY_CLASS_ID
        ):

            continue


        if point_inside_box(
            image_x,
            image_y,
            box
        ):

            return i


    return -1


# ============================================================
# 删除框
# ============================================================

def delete_box_at(
    image_x,
    image_y
):

    global boxes
    global modified
    global selected_box_index


    index = find_box_at(
        image_x,
        image_y
    )


    if index < 0:

        return


    box = boxes[index]


    print(
        "[删除] "
        f"{get_class_name(box['class'])} "
        f"({box['x1']},{box['y1']}) "
        f"-> "
        f"({box['x2']},{box['y2']})"
    )


    boxes.pop(index)


    # ========================================================
    # 修正选中框索引
    # ========================================================

    selected_box_index = -1


    modified = True


    # ========================================================
    # 实时保存
    # ========================================================

    save_labels()


# ============================================================
# 修改选中框类别
# ============================================================

def change_selected_class(
    class_id
):

    global modified


    if selected_box_index < 0:

        print(
            "[类别] 当前没有选中的框"
        )

        return


    if selected_box_index >= len(boxes):

        return


    old_class = boxes[
        selected_box_index
    ]["class"]


    boxes[
        selected_box_index
    ]["class"] = class_id


    modified = True


    print(
        "[修改类别] "
        f"Box {selected_box_index}: "
        f"{old_class}:"
        f"{get_class_name(old_class)}"
        f" -> "
        f"{class_id}:"
        f"{get_class_name(class_id)}"
    )


    # 实时保存

    save_labels()


# ============================================================
# 鼠标滚轮缩放
# ============================================================

def zoom_at_mouse(
    mouse_x,
    mouse_y,
    direction
):

    global display_scale

    global display_offset_x
    global display_offset_y


    # ========================================================
    # 鼠标当前对应的原图坐标
    # ========================================================

    image_x_before, image_y_before = (
        display_to_image(
            mouse_x,
            mouse_y
        )
    )


    old_scale = display_scale


    # ========================================================
    # 新缩放
    # ========================================================

    if direction > 0:

        new_scale = (
            old_scale
            * ZOOM_FACTOR
        )

    else:

        new_scale = (
            old_scale
            / ZOOM_FACTOR
        )


    new_scale = max(
        MIN_SCALE,
        min(
            MAX_SCALE,
            new_scale
        )
    )


    if abs(
        new_scale
        - old_scale
    ) < 0.0001:

        return


    display_scale = new_scale


    # ========================================================
    # 保持鼠标下面的图片位置不动
    # ========================================================

    display_offset_x = int(
        mouse_x
        - image_x_before
        * display_scale
    )

    display_offset_y = int(
        mouse_y
        - image_y_before
        * display_scale
    )


    update_display_size()


# ============================================================
# 更新显示尺寸
# ============================================================

def update_display_size():

    global display_width
    global display_height


    display_width = int(
        image_width
        * display_scale
    )

    display_height = int(
        image_height
        * display_scale
    )


# ============================================================
# 鼠标回调
# ============================================================

def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):

    global drawing
    global deleting
    global panning

    global start_x
    global start_y

    global current_mouse_x
    global current_mouse_y

    global temp_box

    global selected_box_index

    global last_image_x
    global last_image_y

    global pan_start_mouse_x
    global pan_start_mouse_y

    global pan_start_offset_x
    global pan_start_offset_y


    current_mouse_x = x
    current_mouse_y = y


    # ========================================================
    # 显示坐标 -> 图片坐标
    # ========================================================

    image_x, image_y = display_to_image(
        x,
        y
    )


    last_image_x = image_x
    last_image_y = image_y


    # ========================================================
    # 鼠标滚轮
    # ========================================================

    if event == cv2.EVENT_MOUSEWHEEL:

        if flags > 0:

            zoom_at_mouse(
                x,
                y,
                +1
            )

        else:

            zoom_at_mouse(
                x,
                y,
                -1
            )

        return


    # ========================================================
    # 中键按下
    # ========================================================

    if event == cv2.EVENT_MBUTTONDOWN:

        panning = True

        pan_start_mouse_x = x
        pan_start_mouse_y = y

        pan_start_offset_x = (
            display_offset_x
        )

        pan_start_offset_y = (
            display_offset_y
        )

        return


    # ========================================================
    # 中键移动
    # ========================================================

    if (
        event == cv2.EVENT_MOUSEMOVE
        and panning
    ):

        dx = (
            x
            - pan_start_mouse_x
        )

        dy = (
            y
            - pan_start_mouse_y
        )


        globals()[
            "display_offset_x"
        ] = (
            pan_start_offset_x
            + dx
        )


        globals()[
            "display_offset_y"
        ] = (
            pan_start_offset_y
            + dy
        )

        return


    # ========================================================
    # 中键释放
    # ========================================================

    if event == cv2.EVENT_MBUTTONUP:

        panning = False

        return


    # ========================================================
    # 右键按下
    # ========================================================

    if event == cv2.EVENT_RBUTTONDOWN:

        deleting = True

        delete_box_at(
            image_x,
            image_y
        )

        return


    # ========================================================
    # 右键移动
    # ========================================================

    if (
        event == cv2.EVENT_MOUSEMOVE
        and deleting
    ):

        delete_box_at(
            image_x,
            image_y
        )

        return


    # ========================================================
    # 右键释放
    # ========================================================

    if event == cv2.EVENT_RBUTTONUP:

        deleting = False

        return


    # ========================================================
    # 左键按下
    # ========================================================

    if event == cv2.EVENT_LBUTTONDOWN:

        # ----------------------------------------------------
        # 如果点到已有框
        # ----------------------------------------------------

        hit_index = find_box_at(
            image_x,
            image_y
        )


        if hit_index >= 0:

            selected_box_index = (
                hit_index
            )


            print(
                "[选中] "
                f"Box {hit_index} "
                f"{get_class_name(boxes[hit_index]['class'])}"
            )

            return


        # ----------------------------------------------------
        # 没有点到框
        # ----------------------------------------------------

        selected_box_index = -1

        drawing = True

        start_x = image_x
        start_y = image_y

        temp_box = (
            start_x,
            start_y,
            start_x,
            start_y
        )

        return


    # ========================================================
    # 左键移动
    # ========================================================

    if (
        event == cv2.EVENT_MOUSEMOVE
        and drawing
    ):

        x1 = min(
            start_x,
            image_x
        )

        y1 = min(
            start_y,
            image_y
        )

        x2 = max(
            start_x,
            image_x
        )

        y2 = max(
            start_y,
            image_y
        )


        temp_box = (
            x1,
            y1,
            x2,
            y2
        )

        return


    # ========================================================
    # 左键释放
    # ========================================================

    if event == cv2.EVENT_LBUTTONUP:

        if not drawing:

            return


        drawing = False


        # ====================================================
        # 最终坐标
        # ====================================================

        x1 = min(
            start_x,
            image_x
        )

        y1 = min(
            start_y,
            image_y
        )

        x2 = max(
            start_x,
            image_x
        )

        y2 = max(
            start_y,
            image_y
        )


        # ====================================================
        # 限制到图片
        # ====================================================

        x1 = max(
            0,
            min(
                image_width - 1,
                x1
            )
        )

        y1 = max(
            0,
            min(
                image_height - 1,
                y1
            )
        )

        x2 = max(
            0,
            min(
                image_width - 1,
                x2
            )
        )

        y2 = max(
            0,
            min(
                image_height - 1,
                y2
            )
        )


        # ====================================================
        # 防止反向
        # ====================================================

        if x1 > x2:

            x1, x2 = x2, x1

        if y1 > y2:

            y1, y2 = y2, y1


        # ====================================================
        # 检查尺寸
        # ====================================================

        if (
            x2 - x1 >= MIN_BOX_SIZE
            and
            y2 - y1 >= MIN_BOX_SIZE
        ):

            boxes.append({

                "class": CURRENT_CLASS_ID,

                "x1": x1,
                "y1": y1,

                "x2": x2,
                "y2": y2

            })


            selected_box_index = (
                len(boxes) - 1
            )


            print(
                "[新增] "
                f"{CURRENT_CLASS_ID}:"
                f"{get_class_name(CURRENT_CLASS_ID)} "
                f"({x1},{y1}) "
                f"-> "
                f"({x2},{y2})"
            )


            # =================================================
            # 标记修改
            # =================================================

            globals()[
                "modified"
            ] = True


            # =================================================
            # 实时保存
            # =================================================

            save_labels()


        temp_box = None

        return


# ============================================================
# 绘制
# ============================================================

def draw():

    global display_width
    global display_height


    # ========================================================
    # 更新尺寸
    # ========================================================

    update_display_size()


    # ========================================================
    # 获取窗口大小
    # ========================================================

    try:

        window_width, window_height = (
            cv2.getWindowImageRect(
                WINDOW_NAME
            )[2:4]
        )

    except Exception:

        window_width = WINDOW_WIDTH
        window_height = WINDOW_HEIGHT


    window_width = max(
        100,
        window_width
    )

    window_height = max(
        100,
        window_height
    )


    # ========================================================
    # 缩放图片
    # ========================================================

    if (
        display_width > 0
        and
        display_height > 0
    ):

        display_image = cv2.resize(
            image,
            (
                display_width,
                display_height
            ),
            interpolation=(
                cv2.INTER_AREA
                if display_scale < 1.0
                else cv2.INTER_LINEAR
            )
        )

    else:

        display_image = image.copy()


    # ========================================================
    # 创建画布
    # ========================================================

    canvas = np.zeros(
        (
            window_height,
            window_width,
            3
        ),
        dtype=np.uint8
    )


    # ========================================================
    # 图片位置
    # ========================================================

    x1 = display_offset_x
    y1 = display_offset_y

    x2 = x1 + display_width
    y2 = y1 + display_height


    # ========================================================
    # 计算裁剪区域
    # ========================================================

    src_x1 = max(
        0,
        -x1
    )

    src_y1 = max(
        0,
        -y1
    )

    src_x2 = min(
        display_width,
        window_width - x1
    )

    src_y2 = min(
        display_height,
        window_height - y1
    )


    if (
        src_x1 < src_x2
        and
        src_y1 < src_y2
    ):

        dst_x1 = max(
            0,
            x1
        )

        dst_y1 = max(
            0,
            y1
        )

        dst_x2 = (
            dst_x1
            + src_x2
            - src_x1
        )

        dst_y2 = (
            dst_y1
            + src_y2
            - src_y1
        )


        canvas[
            dst_y1:dst_y2,
            dst_x1:dst_x2
        ] = display_image[
            src_y1:src_y2,
            src_x1:src_x2
        ]


    # ========================================================
    # 绘制已有框
    # ========================================================

    for i, box in enumerate(boxes):

        # ====================================================
        # 只显示当前类别
        # ====================================================

        if (
            DISPLAY_CLASS_ID is not None
            and
            box["class"] != DISPLAY_CLASS_ID
        ):

            continue


        box_x1, box_y1 = image_to_display(
            box["x1"],
            box["y1"]
        )

        box_x2, box_y2 = image_to_display(
            box["x2"],
            box["y2"]
        )


        # ====================================================
        # 选中框
        # ====================================================

        if i == selected_box_index:

            box_color = (
                0,
                255,
                255
            )

            thickness = 3

        else:

            box_color = (
                0,
                255,
                0
            )

            thickness = 2


        cv2.rectangle(
            canvas,
            (
                box_x1,
                box_y1
            ),
            (
                box_x2,
                box_y2
            ),
            box_color,
            thickness
        )


        # ====================================================
        # 类别
        # ====================================================

        class_id = box["class"]

        class_name = get_class_name(
            class_id
        )


        text = (
            f"{class_id}:"
            f"{class_name}"
        )


        text_y = max(
            65,
            box_y1 - 6
        )


        cv2.putText(
            canvas,
            text,
            (
                box_x1,
                text_y
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            box_color,
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # 正在创建的框
    # ========================================================

    if temp_box is not None:

        temp_x1, temp_y1 = image_to_display(
            temp_box[0],
            temp_box[1]
        )

        temp_x2, temp_y2 = image_to_display(
            temp_box[2],
            temp_box[3]
        )


        cv2.rectangle(
            canvas,
            (
                temp_x1,
                temp_y1
            ),
            (
                temp_x2,
                temp_y2
            ),
            (255, 0, 0),
            2
        )


        text = (
            f"NEW: "
            f"{CURRENT_CLASS_ID}:"
            f"{get_class_name(CURRENT_CLASS_ID)}"
        )


        cv2.putText(
            canvas,
            text,
            (
                temp_x1,
                max(
                    65,
                    temp_y1 - 6
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 0, 0),
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # 顶部信息栏
    # ========================================================

    cv2.rectangle(
        canvas,
        (0, 0),
        (
            window_width,
            50
        ),
        (25, 25, 25),
        -1
    )


    # ========================================================
    # 当前新建类别
    # ========================================================

    current_class_text = (
        f"{CURRENT_CLASS_ID}:"
        f"{get_class_name(CURRENT_CLASS_ID)}"
    )


    # ========================================================
    # 当前显示类别
    # ========================================================

    if DISPLAY_CLASS_ID is None:

        display_class_text = "ALL"

    else:

        display_class_text = (
            f"{DISPLAY_CLASS_ID}:"
            f"{get_class_name(DISPLAY_CLASS_ID)}"
        )


    # ========================================================
    # 状态栏
    # ========================================================

    status = (
        f"{current_index + 1}/"
        f"{len(image_files)}"
        f"    "
        f"{image_files[current_index]}"
        f"    "
        f"Boxes: {len(boxes)}"
        f"    "
        f"Current: {current_class_text}"
        f"    "
        f"Display: {display_class_text}"
        f"    "
        f"Zoom: "
        f"{display_scale * 100:.0f}%"
    )


    if selected_box_index >= 0:

        status += (
            f"    "
            f"Selected: "
            f"{selected_box_index}"
        )


    if modified:

        status += (
            "    "
            "[MODIFIED]"
        )


    cv2.putText(
        canvas,
        status,
        (12, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )


    return canvas


# ============================================================
# 创建窗口
# ============================================================

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)


cv2.resizeWindow(
    WINDOW_NAME,
    WINDOW_WIDTH,
    WINDOW_HEIGHT
)


cv2.setMouseCallback(
    WINDOW_NAME,
    mouse_callback
)


# ============================================================
# 加载第一张
# ============================================================

if not load_image(
    current_index
):

    cv2.destroyAllWindows()

    exit()


# ============================================================
# 操作提示
# ============================================================

print()
print("=" * 70)
print("YOLO Annotation Tool")
print("=" * 70)

print()

print("【图片】")

print(
    "A              上一张"
)

print(
    "D              下一张"
)

print(
    "SPACE          下一张"
)

print(
    "J              跳转到指定图片"
)


print()

print("【框】")

print(
    "左键拖动       新建框"
)

print(
    "左键点击框     选中框"
)

print(
    "右键按住       删除框"
)


print()

print("【类别 / 显示】")

for class_id in sorted(
    CLASS_NAMES.keys()
):

    print(
        f"{class_id}              "
        f"{CLASS_NAMES[class_id]}"
    )


print()

print(
    "数字键 0~9      "
    "切换显示类别"
)

print(
    "                "
    "同时设置新建框类别"
)

print(
    "                "
    "如果选中框，则修改选中框类别"
)

print(
    "                "
    "注意：Space 不再切换 ALL"
)


print()

print("【缩放 / 移动】")

print(
    "滚轮向上        放大"
)

print(
    "滚轮向下        缩小"
)

print(
    "中键拖动        移动画面"
)

print(
    "F               适合窗口"
)


print()

print("【其他】")

print(
    "S               保存"
)

print(
    "R               恢复"
)

print(
    "J               跳转到指定图片"
)

print(
    "ESC             退出"
)


print()

print(
    f"原始标签："
    f"{LABEL_DIR}"
)

print(
    f"修改标签："
    f"{NEW_LABEL_DIR}"
)


print()

if os.path.abspath(
    LABEL_DIR
) == os.path.abspath(
    NEW_LABEL_DIR
):

    print(
        "⚠ 注意：原始标签目录和修改标签目录相同"
    )

    print(
        "⚠ 修改会直接写入原始 labels"
    )

else:

    print(
        "原始 labels 不会被修改。"
    )


print()

print(
    "当前显示："
    f"{DISPLAY_CLASS_ID}:"
    f"{get_class_name(DISPLAY_CLASS_ID)}"
)


print()

print("=" * 70)


# ============================================================
# 主循环
# ============================================================

while True:

    # ========================================================
    # 绘制
    # ========================================================

    canvas = draw()


    cv2.imshow(
        WINDOW_NAME,
        canvas
    )


    # ========================================================
    # 键盘
    # ========================================================

    key = cv2.waitKey(10) & 0xFF


    # ========================================================
    # ESC
    # ========================================================

    if key == 27:

        if modified:

            save_labels()

        break


    # ========================================================
    # A：上一张
    # ========================================================

    elif key in (
        ord("a"),
        ord("A")
    ):

        if modified:

            save_labels()


        if current_index > 0:

            current_index -= 1

            load_image(
                current_index
            )

        else:

            print(
                "[已经是第一张]"
            )


    # ========================================================
    # D：下一张
    # ========================================================

    elif key in (
        ord("d"),
        ord("D")
    ):

        if modified:

            save_labels()


        if current_index < (
            len(image_files) - 1
        ):

            current_index += 1

            load_image(
                current_index
            )

        else:

            print(
                "[已经是最后一张]"
            )


    # ========================================================
    # SPACE：下一张
    # ========================================================

    elif key == 32:

        if modified:

            save_labels()


        if current_index < (
            len(image_files) - 1
        ):

            current_index += 1

            load_image(
                current_index
            )

        else:

            print(
                "[已经是最后一张]"
            )


    # ========================================================
    # S：保存
    # ========================================================

    elif key in (
        ord("s"),
        ord("S")
    ):

        save_labels()


    # ========================================================
    # R：恢复
    # ========================================================

    elif key in (
        ord("r"),
        ord("R")
    ):

        boxes = [
            box.copy()
            for box in original_boxes
        ]


        selected_box_index = -1


        modified = True


        save_labels()


        print(
            "[恢复] "
            "已恢复到进入当前图片时的状态"
        )


    # ========================================================
    # F：适合窗口
    # ========================================================

    elif key in (
        ord("f"),
        ord("F")
    ):

        fit_image()


        print(
            "[缩放] "
            "适合窗口"
        )


    # ========================================================
    # J：跳转到指定图片
    # ========================================================

    elif key in (
        ord("j"),
        ord("J")
    ):

        jump_to_image()


    # ========================================================
    # 数字键 0~9
    #
    # 作用：
    #
    # 1. 切换显示类别
    #
    # 2. 设置新建框类别
    #
    # 3. 如果选中框，修改选中框类别
    # ========================================================

    elif (
        ord("0")
        <= key
        <= ord("9")
    ):

        class_id = (
            key
            - ord("0")
        )


        # ----------------------------------------------------
        # 检查类别是否存在
        # ----------------------------------------------------

        if class_id not in CLASS_NAMES:

            print(
                f"[类别] "
                f"class {class_id} "
                f"没有定义"
            )

            continue


        # ----------------------------------------------------
        # 切换显示类别
        # ----------------------------------------------------

        DISPLAY_CLASS_ID = class_id


        # ----------------------------------------------------
        # 同时设置新建框类别
        # ----------------------------------------------------

        CURRENT_CLASS_ID = class_id


        # ----------------------------------------------------
        # 如果当前有选中的框
        # 则修改类别
        # ----------------------------------------------------

        if selected_box_index >= 0:

            change_selected_class(
                class_id
            )


        print(
            "[显示类别] "
            f"{class_id}:"
            f"{get_class_name(class_id)}"
        )

        print(
            "[当前新建类别] "
            f"{class_id}:"
            f"{get_class_name(class_id)}"
        )


# ============================================================
# 结束
# ============================================================

cv2.destroyAllWindows()


print()

print("=" * 70)

print("程序结束")

print("=" * 70)

print(
    f"修改后的标签："
    f"{NEW_LABEL_DIR}"
)

print("=" * 70)