import os
import random
import shutil
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

from PIL import Image


# ============================================================
# 配置
# ============================================================

PERSON_DIR = Path(
    r"D:\PycharmProjects\computer_controller\make_different_back\4_640_imgsz_dataset\rmbg_test\transparent"
)

LABEL_DIR = Path(
    r"D:\PycharmProjects\computer_controller\make_different_back\7_dataset_procress\labels_clean"
)

BACKGROUND_DIR = Path(
    r"D:\PycharmProjects\computer_controller\make_different_back\5_background\images"
)

EXISTING_DATASET_DIR = Path(
    r"D:\PycharmProjects\computer_controller\make_different_back\4_640_imgsz_dataset"
)

EXISTING_IMAGE_DIR = EXISTING_DATASET_DIR / "images"
EXISTING_LABEL_DIR = EXISTING_DATASET_DIR / "labels"

OUTPUT_DIR = Path(
    r"D:\PycharmProjects\computer_controller\make_different_back\10_different_back_dataset"
)


# ============================================================
# 输出目录
# ============================================================

IMAGE_TRAIN_DIR = OUTPUT_DIR / "images" / "train"
IMAGE_VAL_DIR   = OUTPUT_DIR / "images" / "val"
LABEL_TRAIN_DIR = OUTPUT_DIR / "labels" / "train"
LABEL_VAL_DIR   = OUTPUT_DIR / "labels" / "val"
YAML_PATH       = OUTPUT_DIR / "data.yaml"


# ============================================================
# 数据集配置
# ============================================================

# 每个背景生成多少张合成图（正样本总数 = 背景数 × PER_BACKGROUND）
PER_BACKGROUND = 50

# 背景负样本：True=把背景裁块作为负样本进 train
USE_BACKGROUND_NEGATIVE = True

# 每张背景裁多少块 640 作为负样本（进 train）
NEGATIVE_PER_BACKGROUND = 5

TRAIN_RATIO = 0.80
TARGET_SIZE = 640

JPEG_QUALITY = 100

# Worker 数量：8GB 内存建议 3~4，别太多（每个 Worker 临时持有背景图）
NUM_WORKERS = 16
CHUNKSIZE = 16

RANDOM_SEED = 20260815

CLASS_NAMES = [
    "body",
    "head",
]

EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# Worker 全局 —— 只存背景【路径】，不常驻图像（内存优化）
# ============================================================

WORKER_BG_PATHS = []
WORKER_BG_INDEX = 0


# ============================================================
# 获取图片
# ============================================================

def get_image_files(directory):

    if not directory.exists():
        return []

    return sorted(
        [
            p
            for p in directory.iterdir()
            if (
                p.is_file()
                and
                p.suffix.lower()
                in EXTENSIONS
            )
        ]
    )


# ============================================================
# 读取背景（按需，用完即释放）
# ============================================================

def load_background(path):

    return Image.open(path).convert("RGBA")


# ============================================================
# Worker 初始化（只收路径字符串，spawn 内存开销极小）
# ============================================================

def init_worker(bg_paths, seed):

    global WORKER_BG_PATHS, WORKER_BG_INDEX

    WORKER_BG_PATHS = bg_paths
    WORKER_BG_INDEX = 0

    pid = os.getpid()
    random.seed(seed + pid)


# ============================================================
# Worker：生成一张 RMBG + 背景图片（正样本）
# ============================================================

def generate_one(task):

    global WORKER_BG_INDEX

    (person_path, split, output_id) = task

    try:

        # ---------- RMBG 人物 ----------
        person = Image.open(person_path).convert("RGBA")

        if person.size != (TARGET_SIZE, TARGET_SIZE):

            return (
                False,
                output_id,
                person_path.name,
                f"人物尺寸错误：{person.size}"
            )

        # ---------- 标签 ----------
        label_path = LABEL_DIR / f"{person_path.stem}.txt"

        if not label_path.exists():

            return (
                False,
                output_id,
                person_path.name,
                "没有对应标签"
            )

        label_text = label_path.read_text(encoding="utf-8")

        # ---------- 背景：轮询取路径，用完释放 ----------
        bg_path = WORKER_BG_PATHS[
            WORKER_BG_INDEX % len(WORKER_BG_PATHS)
        ]

        WORKER_BG_INDEX = (
            WORKER_BG_INDEX + 1
        ) % len(WORKER_BG_PATHS)

        background = None
        background_crop = None

        try:

            background = load_background(bg_path)

            width, height = background.size

            max_x = width - TARGET_SIZE
            max_y = height - TARGET_SIZE

            if max_x < 0 or max_y < 0:

                return (
                    False,
                    output_id,
                    person_path.name,
                    f"背景过小：{background.size}"
                )

            x = random.randint(0, max_x)
            y = random.randint(0, max_y)

            background_crop = background.crop(
                (
                    x,
                    y,
                    x + TARGET_SIZE,
                    y + TARGET_SIZE
                )
            )

        finally:

            if background is not None:
                background.close()

        # ---------- Alpha 合成 ----------
        result = Image.alpha_composite(background_crop, person)

        if background_crop is not None:
            background_crop.close()
        person.close()

        # ---------- 输出 ----------
        filename = f"{output_id:06d}"

        if split == "train":
            output_image = IMAGE_TRAIN_DIR / f"{filename}.jpg"
            output_label = LABEL_TRAIN_DIR / f"{filename}.txt"
        else:
            output_image = IMAGE_VAL_DIR / f"{filename}.jpg"
            output_label = LABEL_VAL_DIR / f"{filename}.txt"

        result.convert("RGB").save(
            output_image,
            "JPEG",
            quality=JPEG_QUALITY
        )

        output_label.write_text(label_text, encoding="utf-8")

        result.close()

        return (
            True,
            output_id,
            person_path.name,
            None
        )

    except Exception as e:

        return (
            False,
            output_id,
            person_path.name,
            str(e)
        )


# ============================================================
# 根据人物数量平均分配生成数量
# ============================================================

def calculate_counts(files, total):

    if not files:
        return {}

    count = len(files)

    base = total // count
    remainder = total % count

    result = {}

    for index, path in enumerate(files):

        current = base

        if index < remainder:
            current += 1

        result[path] = current

    return result

# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 80)
    print("RMBG + 背景 YOLO 数据集生成器（内存优化 + 负样本）")
    print("=" * 80)

    print()

    print(f"RMBG 人物：{PERSON_DIR}")
    print(f"RMBG 标签：{LABEL_DIR}")
    print(f"背景：{BACKGROUND_DIR}")

    print()

    print("已有数据集：")
    print(f"  images：{EXISTING_IMAGE_DIR}")
    print(f"  labels：{EXISTING_LABEL_DIR}")

    print()

    print(f"最终输出：{OUTPUT_DIR}")

    print()

    print(f"每背景合成正样本：{PER_BACKGROUND} 张")
    print(
        f"背景负样本："
        f"{'开（每背景裁 %d 块 640 进 Train）' % NEGATIVE_PER_BACKGROUND if USE_BACKGROUND_NEGATIVE else '关'}"
    )
    print(f"新增合成 Train：{TRAIN_RATIO:.0%}")
    print(f"新增合成 Val：{1 - TRAIN_RATIO:.0%}")
    print(f"目标尺寸：{TARGET_SIZE}x{TARGET_SIZE}")
    print(f"JPEG quality：{JPEG_QUALITY}")
    print(f"Worker：{NUM_WORKERS}")

    print()

    # ========================================================
    # 检查目录
    # ========================================================

    required_dirs = [
        PERSON_DIR,
        LABEL_DIR,
        BACKGROUND_DIR,
        EXISTING_IMAGE_DIR,
        EXISTING_LABEL_DIR,
    ]

    for directory in required_dirs:

        if not directory.exists():

            raise RuntimeError(f"目录不存在：\n{directory}")


    # ========================================================
    # 获取 RMBG 人物
    # ========================================================

    person_files = get_image_files(PERSON_DIR)

    print(f"RMBG 人物图片：{len(person_files)}")


    # 检查标签
    valid_person_files = []
    missing_labels = []

    for person_path in person_files:

        label_path = LABEL_DIR / f"{person_path.stem}.txt"

        if label_path.exists():
            valid_person_files.append(person_path)
        else:
            missing_labels.append(person_path.name)

    person_files = valid_person_files

    print(f"有对应标签：{len(person_files)}")
    print(f"缺少标签：{len(missing_labels)}")

    print()

    if not person_files:
        raise RuntimeError("没有可用的 RMBG 人物。")


    # ========================================================
    # 获取背景【只拿路径，不预载全图到缓存】
    # ========================================================

    background_files = get_image_files(BACKGROUND_DIR)

    print(f"背景图片：{len(background_files)}")

    print()

    if not background_files:
        raise RuntimeError("没有背景图片。")


    # ========================================================
    # 校验背景尺寸（只读尺寸，用完释放，不常驻）
    # ========================================================

    print("=" * 80)
    print("校验背景尺寸（不加载全图常驻）")
    print("=" * 80)

    valid_bg_paths = []
    too_small = 0
    corrupt = 0

    bg_start = time.perf_counter()

    for path in background_files:

        try:

            with Image.open(path) as bg:

                width, height = bg.size

                if width < TARGET_SIZE or height < TARGET_SIZE:

                    too_small += 1
                    continue

                valid_bg_paths.append(str(path))

        except Exception:

            corrupt += 1

    bg_time = time.perf_counter() - bg_start

    print()

    print(f"可用背景：{len(valid_bg_paths)}")
    print(f"背景过小：{too_small}")
    print(f"读取失败：{corrupt}")
    print(f"耗时：{bg_time:.2f}s")

    print()

    if not valid_bg_paths:
        raise RuntimeError("没有可用背景。")


    # ========================================================
    # 获取已有数据（复制进 train）
    # ========================================================

    print("=" * 80)
    print("检查已有训练数据")
    print("=" * 80)

    existing_images = get_image_files(EXISTING_IMAGE_DIR)

    print(f"已有图片：{len(existing_images)}")

    existing_valid = []
    existing_missing_labels = []

    for image_path in existing_images:

        label_path = EXISTING_LABEL_DIR / f"{image_path.stem}.txt"

        if label_path.exists():
            existing_valid.append((image_path, label_path))
        else:
            existing_missing_labels.append(image_path.name)

    print(f"图片 + 标签完整：{len(existing_valid)}")
    print(f"缺少标签：{len(existing_missing_labels)}")

    print()


    # ========================================================
    # 随机划分 RMBG 人物（防跨 Train/Val）
    # ========================================================

    random.seed(RANDOM_SEED)

    shuffled_persons = person_files.copy()
    random.shuffle(shuffled_persons)

    person_count = len(shuffled_persons)
    train_person_count = int(person_count * TRAIN_RATIO)

    train_person_files = shuffled_persons[:train_person_count]
    val_person_files = shuffled_persons[train_person_count:]

    print("=" * 80)
    print("RMBG 人物 Train / Val 划分")
    print("=" * 80)

    print(f"人物总数：{person_count}")
    print(f"Train 人物：{len(train_person_files)}")
    print(f"Val 人物：{len(val_person_files)}")

    print()


    # ========================================================
    # 计算新增数量
    # ========================================================

    num_backgrounds = len(valid_bg_paths)

    total_generated = num_backgrounds * PER_BACKGROUND

    generated_train_target = int(total_generated * TRAIN_RATIO)
    generated_val_target = total_generated - generated_train_target

    negative_count = (
        num_backgrounds * NEGATIVE_PER_BACKGROUND
        if USE_BACKGROUND_NEGATIVE
        else 0
    )

    print("=" * 80)
    print("新增数据数量")
    print("=" * 80)

    print(f"可用背景数：{num_backgrounds}")
    print(f"每背景合成：{PER_BACKGROUND}")
    print(f"正样本合成总数：{total_generated}")
    print(f"正样本 Train：{generated_train_target}")
    print(f"正样本 Val：{generated_val_target}")
    print(f"背景负样本(Train)：{negative_count}")

    print()

    # 分配计数
    train_counts = calculate_counts(train_person_files, generated_train_target)
    val_counts = calculate_counts(val_person_files, generated_val_target)


    # ========================================================
    # 删除旧 OUTPUT
    # ========================================================

    if OUTPUT_DIR.exists():
        print("删除旧输出目录...")
        shutil.rmtree(OUTPUT_DIR)


    # ========================================================
    # 创建目录
    # ========================================================

    IMAGE_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_VAL_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_VAL_DIR.mkdir(parents=True, exist_ok=True)


    # ========================================================
    # 建立合成任务（正样本）
    # ========================================================

    tasks = []
    global_id = 1

    for person_path in train_person_files:

        count = train_counts[person_path]

        for _ in range(count):
            tasks.append((person_path, "train", global_id))
            global_id += 1

    for person_path in val_person_files:

        count = val_counts[person_path]

        for _ in range(count):
            tasks.append((person_path, "val", global_id))
            global_id += 1

    print("=" * 80)
    print("合成任务（正样本）")
    print("=" * 80)

    print(f"任务数量：{len(tasks)}")

    print()

    # ========================================================
    # Worker 数量
    # ========================================================

    if NUM_WORKERS is None:
        worker_count = max(1, cpu_count() - 2)
    else:
        worker_count = NUM_WORKERS

    print(f"CPU 核心数：{cpu_count()}")
    print(f"Worker：{worker_count}")

    print()

    # ========================================================
    # 多进程合成（正样本）
    # ========================================================

    print("=" * 80)
    print("开始生成 RMBG + 背景数据（正样本）")
    print("=" * 80)

    print()

    generation_start = time.perf_counter()

    success = 0
    failed = 0
    processed = 0


    with Pool(
        processes=worker_count,
        initializer=init_worker,
        initargs=(
            valid_bg_paths,     # ★ 只传背景路径列表（字符串），内存友好
            RANDOM_SEED
        )
    ) as pool:

        for result in pool.imap_unordered(
            generate_one,
            tasks,
            chunksize=CHUNKSIZE
        ):

            (ok, output_id, person_name, error) = result

            processed += 1

            if ok:
                success += 1
            else:
                failed += 1
                print(f"[失败] {person_name} -> {error}")

            # 进度
            if processed % 200 == 0 or processed == len(tasks):

                elapsed = time.perf_counter() - generation_start
                speed = processed / elapsed if elapsed > 0 else 0
                remaining = len(tasks) - processed
                eta = remaining / speed if speed > 0 else 0

                print(
                    f"[进度] {processed:5d}/{len(tasks)} "
                    f"| 成功 {success:5d} "
                    f"| 失败 {failed:3d} "
                    f"| {speed:6.2f} 张/s "
                    f"| ETA {eta / 60:6.1f} min"
                )


    generation_time = time.perf_counter() - generation_start

    print()
    print("=" * 80)
    print("RMBG 合成为正样本完成")
    print("=" * 80)

    print(f"成功：{success}")
    print(f"失败：{failed}")
    print(f"耗时：{generation_time / 60:.2f} 分钟")

    if generation_time > 0:
        print(f"速度：{success / generation_time:.2f} 张/s")

    print()


    # ========================================================
    # 生成背景负样本
    #
    # 每张背景随机裁 NEGATIVE_PER_BACKGROUND 块 640
    # 只进 Train；每块配 0 行空标签
    # 让模型学会"没有目标不检测"
    # ========================================================

    if USE_BACKGROUND_NEGATIVE:

        print("=" * 80)
        print(f"生成背景负样本（每背景裁 {NEGATIVE_PER_BACKGROUND} 块 640 -> Train）")
        print("=" * 80)

        print()

        neg_success = 0
        neg_failed = 0

        random.seed(RANDOM_SEED + 1)   # 负样本用独立种子，避免和正样本位置重复

        neg_start = time.perf_counter()

        neg_counter = 1   # ★ 负样本独立编号，避免冲突

        for bg_path_str in valid_bg_paths:

            bg_path = Path(bg_path_str)

            try:

                with Image.open(bg_path).convert("RGBA") as bg:

                    width, height = bg.size

                    max_x = width - TARGET_SIZE
                    max_y = height - TARGET_SIZE

                    if max_x < 0 or max_y < 0:

                        neg_failed += 1
                        continue

                    for _ in range(NEGATIVE_PER_BACKGROUND):

                        x = random.randint(0, max_x)
                        y = random.randint(0, max_y)

                        crop = bg.crop(
                            (
                                x,
                                y,
                                x + TARGET_SIZE,
                                y + TARGET_SIZE
                            )
                        )

                        neg_name = f"neg_{neg_counter:06d}"

                        neg_img = IMAGE_TRAIN_DIR / f"{neg_name}.jpg"
                        neg_lbl = LABEL_TRAIN_DIR / f"{neg_name}.txt"

                        # 保存负样本图片（无人物，转 RGB）
                        crop.convert("RGB").save(
                            neg_img,
                            "JPEG",
                            quality=JPEG_QUALITY
                        )

                        # 空标签（0 字节）：表示无目标
                        neg_lbl.write_text("", encoding="utf-8")

                        crop.close()

                        neg_counter += 1
                        neg_success += 1

            except Exception as e:

                neg_failed += 1

                print(f"[负样本失败] {bg_path.name} -> {e}")


        neg_time = time.perf_counter() - neg_start

        print()

        print(f"负样本成功：{neg_success}")
        print(f"负样本失败：{neg_failed}")
        print(f"耗时：{neg_time / 60:.2f} 分钟")

        print()


    # ========================================================
    # 复制已有数据 -> Train
    # ========================================================

    print("=" * 80)
    print("复制已有数据 -> Train")
    print("=" * 80)

    print()

    existing_success = 0
    existing_failed = 0

    existing_start = time.perf_counter()

    for index, (image_path, label_path) in enumerate(
        existing_valid,
        start=1
    ):

        try:

            new_name = f"existing_{index:06d}"

            destination_image = (
                IMAGE_TRAIN_DIR /
                f"{new_name}"
                f"{image_path.suffix.lower()}"
            )

            destination_label = (
                LABEL_TRAIN_DIR /
                f"{new_name}.txt"
            )

            shutil.copy2(image_path, destination_image)
            shutil.copy2(label_path, destination_label)

            existing_success += 1

        except Exception as e:

            existing_failed += 1

            print(f"[复制失败] {image_path.name}")
            print(f"原因：{e}")

        # 每 500 张显示进度
        if index % 500 == 0 or index == len(existing_valid):

            elapsed = time.perf_counter() - existing_start
            speed = index / elapsed if elapsed > 0 else 0

            print(
                f"[已有数据] {index}/{len(existing_valid)} "
                f"| {speed:.1f} 张/s"
            )

    print()

    print(f"已有数据成功加入 Train：{existing_success}")
    print(f"已有数据复制失败：{existing_failed}")
    print(f"已有数据缺少标签：{len(existing_missing_labels)}")

    print()

    # ========================================================
    # data.yaml
    # ========================================================

    yaml_text = f"""# YOLO Dataset

path: {OUTPUT_DIR.as_posix()}

train: images/train
val: images/val

names:
  0: {CLASS_NAMES[0]}
  1: {CLASS_NAMES[1]}
"""

    YAML_PATH.write_text(
        yaml_text,
        encoding="utf-8"
    )

    print("=" * 80)
    print("data.yaml 已生成")
    print("=" * 80)

    print(YAML_PATH)

    print()


    # ========================================================
    # 最终统计
    # ========================================================

    train_images = get_image_files(IMAGE_TRAIN_DIR)
    val_images = get_image_files(IMAGE_VAL_DIR)

    train_labels = [
        p
        for p in LABEL_TRAIN_DIR.iterdir()
        if (
            p.is_file()
            and
            p.suffix.lower() == ".txt"
        )
    ]

    val_labels = [
        p
        for p in LABEL_VAL_DIR.iterdir()
        if (
            p.is_file()
            and
            p.suffix.lower() == ".txt"
        )
    ]


    print("=" * 80)
    print("最终数据集")
    print("=" * 80)

    print()

    print(f"正样本合成 Train：{generated_train_target}")
    print(f"正样本合成 Val：{generated_val_target}")
    print(f"背景负样本 Train：{negative_count}")

    print()

    print(f"已有数据 Train：{existing_success}")

    print()

    print(f"最终 Train 图片：{len(train_images)}")
    print(f"最终 Train 标签：{len(train_labels)}")

    print()

    print(f"最终 Val 图片：{len(val_images)}")
    print(f"最终 Val 标签：{len(val_labels)}")

    print()

    print(f"最终总图片：{len(train_images) + len(val_images)}")

    print()


    # ========================================================
    # 数据集结构
    # ========================================================

    print("=" * 80)
    print("最终目录")
    print("=" * 80)

    print(
        f"""
{OUTPUT_DIR.name}/
│
├── images/
│   ├── train/
│   │   ├── 000001.jpg            (正样本合成)
│   │   ├── ...                   (正样本合成)
│   │   ├── neg_000001.jpg        (背景负样本)
│   │   ├── ...                   (背景负样本)
│   │   ├── existing_000001.jpg   (已有数据复制)
│   │   └── ...
│   │
│   └── val/
│       ├── 007201.jpg            (正样本合成 Val)
│       └── ...
│
├── labels/
│   ├── train/
│   │   ├── 000001.txt
│   │   ├── ...
│   │   ├── neg_000001.txt        (空标签 = 负样本)
│   │   ├── ...
│   │   ├── existing_000001.txt
│   │   └── ...
│   │
│   └── val/
│       ├── 007201.txt
│       └── ...
│
└── data.yaml
"""
    )


    # ========================================================
    # 数据说明
    # ========================================================

    print("=" * 80)
    print("数据集说明")
    print("=" * 80)

    print()

    print("✓ 原有数据全部进入 Train")
    print("✓ 原有数据不会进入 Val")
    print("✓ 新 RMBG 数据按照人物先划分 Train / Val")
    print("✓ 同一个 RMBG 人物不会同时出现在 Train / Val")
    print("✓ 背景不 resize，只随机裁剪 640×640")
    print("✓ RMBG 人物保持 640×640")
    print(f"✓ 每张背景生成 {PER_BACKGROUND} 张正样本合成图")
    print("✓ 每个背景都被轮询用到（背景上下左右随机位置）")
    print(f"✓ 背景负样本只进 Train：每背景随机裁 {NEGATIVE_PER_BACKGROUND} 块 640")
    print("✓ 负样本标签为空（0 行），帮助模型识别'无目标'")
    print("✓ 内存优化：背景不常驻，只传路径，Worker 按需读取释放")

    print()

    print("=" * 80)
    print("全部完成")
    print("=" * 80)