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

# ------------------------------------------------------------
# RMBG 透明人物
# ------------------------------------------------------------

PERSON_DIR = Path(
    r"D:\PycharmProjects\computer_controller\make_dataset\640_head_body_dataset\rmbg_test\transparent"
)

# ------------------------------------------------------------
# 原始 YOLO 标签
# ------------------------------------------------------------

LABEL_DIR = Path(
    r"D:\PycharmProjects\computer_controller\make_dataset\640_head_body_dataset\labels"
)

# ------------------------------------------------------------
# 背景
# ------------------------------------------------------------

BACKGROUND_DIR = Path(
    r"D:\PycharmProjects\computer_controller\make_dataset\background"
)

# ------------------------------------------------------------
# 已有训练数据
#
# 这里面的 train / test / labels
# 全部只进入新数据集的 train
#
# 不参与新的 val
# ------------------------------------------------------------

EXISTING_DATASET_DIR = Path(
    r"D:\PycharmProjects\computer_controller\make_dataset\640_head_body_dataset_train"
)

# ------------------------------------------------------------
# 最终输出
# ------------------------------------------------------------

OUTPUT_DIR = Path(
    r"D:\PycharmProjects\computer_controller\make_dataset\different_back"
)


# ============================================================
# 输出目录
# ============================================================

IMAGE_TRAIN_DIR = OUTPUT_DIR / "images" / "train"
IMAGE_VAL_DIR = OUTPUT_DIR / "images" / "val"

LABEL_TRAIN_DIR = OUTPUT_DIR / "labels" / "train"
LABEL_VAL_DIR = OUTPUT_DIR / "labels" / "val"

YAML_PATH = OUTPUT_DIR / "data.yaml"


# ============================================================
# 数据集配置
# ============================================================

MAX_GENERATED_IMAGES = 9000

TRAIN_RATIO = 0.80

TARGET_SIZE = 640


# ============================================================
# JPEG 配置
#
# 原来：
#
# quality=95
# subsampling=0
#
# JPEG 编码比较慢。
#
# 训练数据一般 quality=90 足够。
# ============================================================

JPEG_QUALITY = 90


# ============================================================
# 多进程配置
# ============================================================

# 推荐：
#
# 6
#
# 如果 CPU 还有大量余量：
#
# 8
#
# 不建议一开始就开满所有线程，
# 因为还涉及磁盘写入。
# ============================================================

NUM_WORKERS = 6


# ============================================================
# chunksize
#
# 一次给 Worker 一批任务。
#
# 太小：
#   进程调度开销大
#
# 太大：
#   任务分配不够灵活
#
# 这里 16 比较合适。
# ============================================================

CHUNKSIZE = 16


# ============================================================
# 随机种子
# ============================================================

RANDOM_SEED = 20260815


# ============================================================
# YOLO 类别
#
# 按你之前的 head/body 数据集：
#
# 0 = body
# 1 = head
#
# 如果你的原标签正好相反：
#
# 修改这里即可。
# ============================================================

CLASS_NAMES = [
    "body",
    "head",
]


# ============================================================
# 图片扩展名
# ============================================================

EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# Worker 全局变量
#
# Windows 多进程下：
# 每个 Worker 自己保存一份背景缓存。
# ============================================================

WORKER_BACKGROUNDS = []


# ============================================================
# 工具函数
# ============================================================

def get_image_files(directory):

    if not directory.exists():

        return []

    return sorted(
        [
            p
            for p in directory.iterdir()
            if p.is_file()
            and p.suffix.lower() in EXTENSIONS
        ]
    )


# ============================================================
# Worker 初始化
# ============================================================

def init_worker(
    backgrounds,
    seed
):

    global WORKER_BACKGROUNDS

    WORKER_BACKGROUNDS = backgrounds

    # 每个 Worker 使用不同随机种子
    pid = os.getpid()

    random.seed(
        seed + pid
    )


# ============================================================
# Worker：
# 生成一张图片
# ============================================================

def generate_one(
    task
):

    global WORKER_BACKGROUNDS

    (
        person_path,
        split,
        output_id
    ) = task

    try:

        # ====================================================
        # 读取人物
        # ====================================================

        person = Image.open(
            person_path
        ).convert(
            "RGBA"
        )


        # ====================================================
        # 检查人物尺寸
        # ====================================================

        if person.size != (
            TARGET_SIZE,
            TARGET_SIZE
        ):

            return (
                False,
                output_id,
                person_path.name,
                f"人物尺寸错误：{person.size}"
            )


        # ====================================================
        # 读取 YOLO 标签
        # ====================================================

        label_path = (
            LABEL_DIR /
            f"{person_path.stem}.txt"
        )

        label_text = label_path.read_text(
            encoding="utf-8"
        )


        # ====================================================
        # 随机背景
        # ====================================================

        background_name, background = random.choice(
            WORKER_BACKGROUNDS
        )


        # ====================================================
        # 随机裁剪背景
        #
        # 不 resize
        # ====================================================

        width, height = background.size

        max_x = (
            width -
            TARGET_SIZE
        )

        max_y = (
            height -
            TARGET_SIZE
        )

        x = random.randint(
            0,
            max_x
        )

        y = random.randint(
            0,
            max_y
        )

        background_crop = background.crop(
            (
                x,
                y,
                x + TARGET_SIZE,
                y + TARGET_SIZE
            )
        )


        # ====================================================
        # Alpha 合成
        # ====================================================

        result = Image.alpha_composite(
            background_crop,
            person
        )


        # ====================================================
        # 输出文件名
        # ====================================================

        filename = (
            f"{output_id:06d}"
        )


        # ====================================================
        # Train / Val
        # ====================================================

        if split == "train":

            output_image = (
                IMAGE_TRAIN_DIR /
                f"{filename}.jpg"
            )

            output_label = (
                LABEL_TRAIN_DIR /
                f"{filename}.txt"
            )

        else:

            output_image = (
                IMAGE_VAL_DIR /
                f"{filename}.jpg"
            )

            output_label = (
                LABEL_VAL_DIR /
                f"{filename}.txt"
            )


        # ====================================================
        # JPEG
        #
        # 不再：
        #
        # subsampling=0
        #
        # 让 Pillow 自己选择默认采样。
        #
        # 对 YOLO 训练通常完全够用。
        # ====================================================

        result.convert(
            "RGB"
        ).save(
            output_image,
            "JPEG",
            quality=JPEG_QUALITY
        )


        # ====================================================
        # 标签
        # ====================================================

        output_label.write_text(
            label_text,
            encoding="utf-8"
        )


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
# 根据人物数量分配生成数量
# ============================================================

def calculate_counts(
    files,
    total
):

    if not files:

        return {}


    count = len(files)

    base = (
        total //
        count
    )

    remainder = (
        total %
        count
    )

    result = {}

    for index, path in enumerate(files):

        current = base

        if index < remainder:

            current += 1

        result[path] = current

    return result


# ============================================================
# 查找已有数据集图片
# ============================================================

def find_existing_images():

    if not EXISTING_DATASET_DIR.exists():

        return []


    result = []

    checked = set()


    # --------------------------------------------------------
    # 常见结构
    # --------------------------------------------------------

    possible_dirs = [

        EXISTING_DATASET_DIR,

        EXISTING_DATASET_DIR / "images",

        EXISTING_DATASET_DIR / "train",

        EXISTING_DATASET_DIR / "test",

        EXISTING_DATASET_DIR / "images" / "train",

        EXISTING_DATASET_DIR / "images" / "test",

    ]


    for base in possible_dirs:

        if not base.exists():

            continue


        for path in base.rglob("*"):

            if not path.is_file():

                continue

            if path.suffix.lower() not in EXTENSIONS:

                continue


            resolved = path.resolve()


            if resolved in checked:

                continue


            checked.add(
                resolved
            )

            result.append(
                path
            )


    return sorted(
        result
    )


# ============================================================
# 查找已有数据对应 Label
# ============================================================

def find_existing_label(
    image_path
):

    stem = image_path.stem


    candidates = [

        EXISTING_DATASET_DIR /
        "labels" /
        f"{stem}.txt",

        EXISTING_DATASET_DIR /
        "labels" /
        "train" /
        f"{stem}.txt",

        EXISTING_DATASET_DIR /
        "labels" /
        "test" /
        f"{stem}.txt",

        image_path.parent /
        "labels" /
        f"{stem}.txt",

        image_path.parent.parent /
        "labels" /
        f"{stem}.txt",

    ]


    for candidate in candidates:

        if candidate.exists():

            return candidate


    # 最后全局搜索

    matches = list(
        EXISTING_DATASET_DIR.rglob(
            f"{stem}.txt"
        )
    )


    if matches:

        return matches[0]


    return None


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # Windows multiprocessing 必须放在 main
    # ========================================================

    print("=" * 80)
    print("RMBG + CS2 背景 YOLO 数据集生成")
    print("=" * 80)

    print()

    print(
        f"人物：{PERSON_DIR}"
    )

    print(
        f"标签：{LABEL_DIR}"
    )

    print(
        f"背景：{BACKGROUND_DIR}"
    )

    print(
        f"已有数据：{EXISTING_DATASET_DIR}"
    )

    print(
        f"输出：{OUTPUT_DIR}"
    )

    print()

    print(
        f"目标合成：{MAX_GENERATED_IMAGES}"
    )

    print(
        f"Train：{TRAIN_RATIO:.0%}"
    )

    print(
        f"Val：{1 - TRAIN_RATIO:.0%}"
    )

    print(
        f"Worker：{NUM_WORKERS}"
    )

    print(
        f"Chunksize：{CHUNKSIZE}"
    )

    print(
        f"JPEG：quality={JPEG_QUALITY}"
    )

    print()


    # ========================================================
    # 检查目录
    # ========================================================

    if not PERSON_DIR.exists():

        raise RuntimeError(
            f"人物目录不存在：\n{PERSON_DIR}"
        )


    if not LABEL_DIR.exists():

        raise RuntimeError(
            f"标签目录不存在：\n{LABEL_DIR}"
        )


    if not BACKGROUND_DIR.exists():

        raise RuntimeError(
            f"背景目录不存在：\n{BACKGROUND_DIR}"
        )


    # ========================================================
    # 获取人物
    # ========================================================

    person_files = get_image_files(
        PERSON_DIR
    )

    background_files = get_image_files(
        BACKGROUND_DIR
    )


    print(
        f"人物图片：{len(person_files)}"
    )

    print(
        f"背景图片：{len(background_files)}"
    )

    print()


    # ========================================================
    # 检查人物标签
    # ========================================================

    valid_person_files = []

    missing_labels = []


    for person_path in person_files:

        label_path = (
            LABEL_DIR /
            f"{person_path.stem}.txt"
        )


        if label_path.exists():

            valid_person_files.append(
                person_path
            )

        else:

            missing_labels.append(
                person_path.name
            )


    person_files = valid_person_files


    print(
        f"有标签人物：{len(person_files)}"
    )

    print(
        f"缺少标签：{len(missing_labels)}"
    )

    print()


    if not person_files:

        raise RuntimeError(
            "没有有效人物。"
        )


    if not background_files:

        raise RuntimeError(
            "没有背景。"
        )


    # ========================================================
    # 加载背景
    #
    # 关键优化：
    #
    # 这里直接：
    #
    # RGB -> RGBA
    #
    # 后面不再重复 convert。
    # ========================================================

    print("=" * 80)
    print("预加载背景")
    print("=" * 80)

    background_cache = []

    too_small = 0

    failed_background = 0


    background_start = time.perf_counter()


    for path in background_files:

        try:

            bg = Image.open(
                path
            ).convert(
                "RGBA"
            )


            width, height = bg.size


            if (
                width < TARGET_SIZE
                or
                height < TARGET_SIZE
            ):

                too_small += 1

                print(
                    f"[跳过] "
                    f"{path.name} "
                    f"{width}x{height}"
                )

                continue


            background_cache.append(
                (
                    path.name,
                    bg
                )
            )


        except Exception as e:

            failed_background += 1

            print(
                f"[背景失败] "
                f"{path.name}"
            )

            print(e)


    background_time = (
        time.perf_counter()
        - background_start
    )


    print()

    print(
        f"可用背景："
        f"{len(background_cache)}"
    )

    print(
        f"过小："
        f"{too_small}"
    )

    print(
        f"失败："
        f"{failed_background}"
    )

    print(
        f"加载耗时："
        f"{background_time:.2f}s"
    )

    print()


    if not background_cache:

        raise RuntimeError(
            "没有可用背景。"
        )


    # ========================================================
    # 划分人物
    #
    # 核心：
    #
    # 先划分人物
    # 再生成图片
    #
    # 防止数据泄露。
    # ========================================================

    random.seed(
        RANDOM_SEED
    )


    shuffled_persons = (
        person_files.copy()
    )


    random.shuffle(
        shuffled_persons
    )


    person_count = len(
        shuffled_persons
    )


    train_person_count = int(
        person_count *
        TRAIN_RATIO
    )


    train_person_files = (
        shuffled_persons[
            :train_person_count
        ]
    )


    val_person_files = (
        shuffled_persons[
            train_person_count:
        ]
    )


    print("=" * 80)
    print("原始人物划分")
    print("=" * 80)

    print(
        f"人物总数："
        f"{person_count}"
    )

    print(
        f"Train 人物："
        f"{len(train_person_files)}"
    )

    print(
        f"Val 人物："
        f"{len(val_person_files)}"
    )

    print()

    print(
        "✓ 同一个人物不会同时进入 Train / Val"
    )

    print(
        "✓ 不按生成后的 9000 张图片随机划分"
    )

    print()


    # ========================================================
    # 计算 Train / Val 生成数量
    # ========================================================

    generated_train_target = int(
        MAX_GENERATED_IMAGES *
        TRAIN_RATIO
    )


    generated_val_target = (
        MAX_GENERATED_IMAGES -
        generated_train_target
    )


    train_counts = calculate_counts(
        train_person_files,
        generated_train_target
    )


    val_counts = calculate_counts(
        val_person_files,
        generated_val_target
    )


    print("=" * 80)
    print("生成数量")
    print("=" * 80)

    print(
        f"Train："
        f"{generated_train_target}"
    )

    print(
        f"Val："
        f"{generated_val_target}"
    )

    print(
        f"总计："
        f"{generated_train_target + generated_val_target}"
    )

    print()


    # ========================================================
    # 删除旧数据集
    #
    # 防止旧数据残留。
    # ========================================================

    if OUTPUT_DIR.exists():

        print(
            "删除旧输出目录..."
        )

        shutil.rmtree(
            OUTPUT_DIR
        )


    # ========================================================
    # 创建目录
    # ========================================================

    IMAGE_TRAIN_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    IMAGE_VAL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    LABEL_TRAIN_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    LABEL_VAL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # 建立生成任务
    # ========================================================

    tasks = []

    global_id = 1


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    for person_path in train_person_files:

        count = train_counts[
            person_path
        ]


        for _ in range(count):

            tasks.append(
                (
                    person_path,
                    "train",
                    global_id
                )
            )

            global_id += 1


    # --------------------------------------------------------
    # Val
    # --------------------------------------------------------

    for person_path in val_person_files:

        count = val_counts[
            person_path
        ]


        for _ in range(count):

            tasks.append(
                (
                    person_path,
                    "val",
                    global_id
                )
            )

            global_id += 1


    print("=" * 80)
    print("生成任务")
    print("=" * 80)

    print(
        f"任务数："
        f"{len(tasks)}"
    )

    print()


    # ========================================================
    # 确定 Worker 数量
    # ========================================================

    if NUM_WORKERS is None:

        worker_count = max(
            1,
            cpu_count() - 2
        )

    else:

        worker_count = NUM_WORKERS


    print(
        f"CPU 核心数："
        f"{cpu_count()}"
    )

    print(
        f"实际 Worker："
        f"{worker_count}"
    )

    print()


    # ========================================================
    # 多进程生成
    # ========================================================

    print("=" * 80)
    print("开始多进程生成")
    print("=" * 80)

    print()

    generation_start = (
        time.perf_counter()
    )


    success = 0
    failed = 0

    processed = 0


    with Pool(
        processes=worker_count,
        initializer=init_worker,
        initargs=(
            background_cache,
            RANDOM_SEED
        )
    ) as pool:

        for result in pool.imap_unordered(
            generate_one,
            tasks,
            chunksize=CHUNKSIZE
        ):

            (
                ok,
                output_id,
                person_name,
                error
            ) = result


            processed += 1


            if ok:

                success += 1

            else:

                failed += 1

                print(
                    f"[失败] "
                    f"{person_name} "
                    f"-> {error}"
                )


            # ------------------------------------------------
            # 每 200 张输出一次
            # ------------------------------------------------

            if (
                processed % 200 == 0
                or
                processed == len(tasks)
            ):

                elapsed = (
                    time.perf_counter()
                    -
                    generation_start
                )


                speed = (
                    processed /
                    elapsed
                    if elapsed > 0
                    else 0
                )


                remaining = (
                    len(tasks) -
                    processed
                )


                eta = (
                    remaining /
                    speed
                    if speed > 0
                    else 0
                )


                print(
                    f"[进度] "
                    f"{processed:5d}/"
                    f"{len(tasks)} "
                    f"| 成功 "
                    f"{success:5d} "
                    f"| 失败 "
                    f"{failed:3d} "
                    f"| "
                    f"{speed:6.2f} 张/s "
                    f"| ETA "
                    f"{eta / 60:6.1f} min"
                )


    generation_time = (
        time.perf_counter()
        -
        generation_start
    )


    print()

    print("=" * 80)
    print("合成数据生成完成")
    print("=" * 80)

    print(
        f"成功："
        f"{success}"
    )

    print(
        f"失败："
        f"{failed}"
    )

    print(
        f"耗时："
        f"{generation_time / 60:.2f} 分钟"
    )


    if generation_time > 0:

        print(
            f"速度："
            f"{success / generation_time:.2f} 张/s"
        )

    print()


    # ========================================================
    # 复制已有数据
    #
    # 全部 -> Train
    #
    # 不进入 Val
    # ========================================================

    print("=" * 80)
    print("复制已有数据到 Train")
    print("=" * 80)


    existing_images = (
        find_existing_images()
    )


    print(
        f"发现已有图片："
        f"{len(existing_images)}"
    )

    print()


    existing_success = 0
    existing_missing_label = 0
    existing_failed = 0


    for index, image_path in enumerate(
        existing_images,
        start=1
    ):

        try:

            label_path = (
                find_existing_label(
                    image_path
                )
            )


            if label_path is None:

                existing_missing_label += 1

                print(
                    f"[缺标签] "
                    f"{image_path}"
                )

                continue


            # ------------------------------------------------
            # 给已有数据加前缀
            #
            # 防止和 000001.jpg 冲突
            # ------------------------------------------------

            new_name = (
                f"existing_"
                f"{index:06d}"
            )


            destination_image = (
                IMAGE_TRAIN_DIR /
                f"{new_name}"
                f"{image_path.suffix.lower()}"
            )


            destination_label = (
                LABEL_TRAIN_DIR /
                f"{new_name}.txt"
            )


            shutil.copy2(
                image_path,
                destination_image
            )


            shutil.copy2(
                label_path,
                destination_label
            )


            existing_success += 1


        except Exception as e:

            existing_failed += 1

            print(
                f"[复制失败] "
                f"{image_path}"
            )

            print(e)


    print()

    print(
        f"已有数据加入 Train："
        f"{existing_success}"
    )

    print(
        f"缺少 Label："
        f"{existing_missing_label}"
    )

    print(
        f"复制失败："
        f"{existing_failed}"
    )

    print()


    # ========================================================
    # 生成 data.yaml
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

    print()

    print(
        YAML_PATH
    )

    print()


    # ========================================================
    # 最终统计
    # ========================================================

    train_images = [
        p
        for p in IMAGE_TRAIN_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in EXTENSIONS
    ]


    val_images = [
        p
        for p in IMAGE_VAL_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in EXTENSIONS
    ]


    train_labels = [
        p
        for p in LABEL_TRAIN_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".txt"
    ]


    val_labels = [
        p
        for p in LABEL_VAL_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() == ".txt"
    ]


    # ========================================================
    # 最终结果
    # ========================================================

    print("=" * 80)
    print("最终数据集")
    print("=" * 80)

    print()

    print(
        f"Train 图片："
        f"{len(train_images)}"
    )

    print(
        f"Train 标签："
        f"{len(train_labels)}"
    )

    print()

    print(
        f"Val 图片："
        f"{len(val_images)}"
    )

    print(
        f"Val 标签："
        f"{len(val_labels)}"
    )

    print()

    print(
        f"总图片："
        f"{len(train_images) + len(val_images)}"
    )

    print()

    # ========================================================
    # 目录
    # ========================================================

    print("=" * 80)
    print("目录结构")
    print("=" * 80)

    print(
        f"""
{OUTPUT_DIR.name}/
├── images/
│   ├── train/
│   └── val/
│
├── labels/
│   ├── train/
│   └── val/
│
└── data.yaml
"""
    )


    print("=" * 80)
    print("数据泄露检查")
    print("=" * 80)

    print(
        "✓ 先按原始人物划分 Train / Val"
    )

    print(
        "✓ 同一个人物不会同时出现在 Train / Val"
    )

    print(
        "✓ 同一个人物的不同背景不会跨集合"
    )

    print(
        "✓ 新生成数据约 7200 Train + 1800 Val"
    )

    print(
        "✓ 旧数据全部进入 Train"
    )

    print(
        "✓ 旧数据不会进入 Val"
    )

    print(
        "✓ 背景不 resize"
    )

    print(
        "✓ 背景只随机裁剪 640×640"
    )

    print()

    print("=" * 80)
    print("完成")
    print("=" * 80)
