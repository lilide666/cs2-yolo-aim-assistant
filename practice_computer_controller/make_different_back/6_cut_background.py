import os
import time
from pathlib import Path

import torch
import numpy as np

from PIL import Image
from transformers import AutoModelForImageSegmentation
import torchvision.transforms as transforms


# ============================================================
# 配置
# ============================================================

IMAGE_DIR = Path(
    r"D:\PycharmProjects\computer_controller\make_different_back\4_640_imgsz_dataset\images"
)

OUTPUT_DIR = IMAGE_DIR.parent / "rmbg_test"

# 处理数量
MAX_IMAGES = 2325

# Batch 大小
# RTX 4060 8GB 建议先从 4 开始
BATCH_SIZE = 8

# RMBG 输入尺寸
INPUT_SIZE = 640

MODEL_NAME = "briaai/RMBG-2.0"


# ============================================================
# CUDA
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError("没有检测到 CUDA GPU")

device = torch.device("cuda")

print("=" * 70)
print("RMBG 2.0 高速批量抠图")
print("=" * 70)

print(f"GPU：{torch.cuda.get_device_name(0)}")

gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3

print(f"GPU显存：{gpu_memory:.2f} GB")
print(f"Batch Size：{BATCH_SIZE}")
print(f"输入尺寸：{INPUT_SIZE}x{INPUT_SIZE}")
print()


# ============================================================
# 输出目录
# ============================================================

ALPHA_DIR = OUTPUT_DIR / "transparent"
WHITE_DIR = OUTPUT_DIR / "white_background"

ALPHA_DIR.mkdir(parents=True, exist_ok=True)
WHITE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 获取图片
# ============================================================

extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}

image_files = sorted([
    p
    for p in IMAGE_DIR.iterdir()
    if p.is_file() and p.suffix.lower() in extensions
])

image_files = image_files[:MAX_IMAGES]

if not image_files:
    raise RuntimeError("没有找到图片")

print(f"图片数量：{len(image_files)}")
print()


# ============================================================
# Transform
# ============================================================

transform = transforms.Compose([
    transforms.Resize(
        (INPUT_SIZE, INPUT_SIZE),
        antialias=True
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ============================================================
# 加载 RMBG
# ============================================================

print("正在加载 RMBG 2.0 ...")

model = AutoModelForImageSegmentation.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

model = model.to(device)

# FP16
model = model.half()

model.eval()

print("RMBG 2.0 加载完成")

print()


# ============================================================
# 单个 Batch
# ============================================================

def process_batch(batch_paths):

    images = []
    original_sizes = []

    # --------------------------------------------------------
    # CPU 图片读取
    # --------------------------------------------------------

    for path in batch_paths:

        image = Image.open(path).convert("RGB")

        original_sizes.append(image.size)

        tensor = transform(image)

        images.append(tensor)

    # --------------------------------------------------------
    # Batch
    # --------------------------------------------------------

    batch = torch.stack(images)

    # FP16
    batch = batch.half()

    # GPU
    batch = batch.to(
        device,
        non_blocking=True
    )

    # --------------------------------------------------------
    # GPU 推理
    # --------------------------------------------------------

    with torch.inference_mode():

        preds = model(batch)[-1].sigmoid()

    # --------------------------------------------------------
    # CPU
    # --------------------------------------------------------

    preds = preds.float().cpu()

    results = []

    for i in range(len(batch_paths)):

        pred = preds[i].squeeze()

        alpha = (
            pred.numpy() * 255
        ).clip(
            0,
            255
        ).astype(
            np.uint8
        )

        alpha = Image.fromarray(
            alpha,
            mode="L"
        )

        alpha = alpha.resize(
            original_sizes[i],
            Image.Resampling.LANCZOS
        )

        results.append(alpha)

    return results


# ============================================================
# 处理
# ============================================================

total = len(image_files)

start_time = time.perf_counter()

processed = 0

for start in range(
    0,
    total,
    BATCH_SIZE
):

    batch_paths = image_files[
        start:start + BATCH_SIZE
    ]

    batch_start = time.perf_counter()

    try:

        # ====================================================
        # RMBG 推理
        # ====================================================

        alphas = process_batch(
            batch_paths
        )

        # ====================================================
        # 保存
        # ====================================================

        for image_path, alpha in zip(
            batch_paths,
            alphas
        ):

            image = Image.open(
                image_path
            ).convert(
                "RGB"
            )

            # ------------------------------------------------
            # RGBA
            # ------------------------------------------------

            rgba = image.convert(
                "RGBA"
            )

            rgba.putalpha(
                alpha
            )

            output_name = (
                image_path.stem + ".png"
            )

            transparent_path = (
                ALPHA_DIR / output_name
            )

            rgba.save(
                transparent_path,
                "PNG"
            )

            # ------------------------------------------------
            # 白色背景
            # ------------------------------------------------

            white = Image.new(
                "RGBA",
                image.size,
                (255, 255, 255, 255)
            )

            white.alpha_composite(
                rgba
            )

            white_path = (
                WHITE_DIR / output_name
            )

            white.convert(
                "RGB"
            ).save(
                white_path,
                "PNG"
            )

        processed += len(batch_paths)

        # ====================================================
        # 统计
        # ====================================================

        elapsed = (
            time.perf_counter()
            - start_time
        )

        speed = (
            processed / elapsed
        )

        remaining = (
            total - processed
        )

        eta = (
            remaining / speed
            if speed > 0
            else 0
        )

        batch_time = (
            time.perf_counter()
            - batch_start
        )

        print(
            f"[{processed:4d}/{total}] "
            f"Batch={len(batch_paths)} "
            f"耗时={batch_time:.2f}s "
            f"速度={speed:.2f} 张/s "
            f"ETA={eta/60:.1f} min"
        )

    except RuntimeError as e:

        # ====================================================
        # 显存不足
        # ====================================================

        if "out of memory" in str(e).lower():

            torch.cuda.empty_cache()

            print()
            print(
                "❌ CUDA 显存不足！"
            )

            print(
                "请把 BATCH_SIZE 改小。"
            )

            print(
                "例如：BATCH_SIZE = 2"
            )

            raise

        else:

            print(
                f"处理失败：{e}"
            )

    except Exception as e:

        print(
            f"处理失败：{e}"
        )


# ============================================================
# 完成
# ============================================================

total_time = (
    time.perf_counter()
    - start_time
)

print()
print("=" * 70)
print("处理完成")
print("=" * 70)

print(
    f"处理数量：{processed}"
)

print(
    f"总耗时：{total_time / 60:.2f} 分钟"
)

if total_time > 0:

    print(
        f"平均速度："
        f"{processed / total_time:.2f} 张/秒"
    )

print()
print(
    f"透明图：{ALPHA_DIR}"
)

print(
    f"白色背景：{WHITE_DIR}"
)

print()

# ============================================================
# GPU 显存
# ============================================================

allocated = (
    torch.cuda.memory_allocated()
    / 1024**3
)

reserved = (
    torch.cuda.memory_reserved()
    / 1024**3
)

print(
    f"GPU当前显存："
    f"{allocated:.2f} GB"
)

print(
    f"GPU缓存显存："
    f"{reserved:.2f} GB"
)