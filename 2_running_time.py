import time
import statistics
from pathlib import Path

import cv2
from ultralytics import YOLO


# ============================================================
# 配置
# ============================================================

MODELS = [
    (
        "PyTorch PT",
        r"D:\PycharmProjects\aim\18_model_aug\weights\best.pt"
    ),

    (
        "TensorRT FP32",
        r"D:\PycharmProjects\aim\18_model_aug\tensorrt_models\yolov26_fp32.engine"
    ),

    (
        "TensorRT FP16",
        r"D:\PycharmProjects\aim\18_model_aug\tensorrt_models\yolov26_fp16.engine"
    ),

    (
        "TensorRT INT8",
        r"D:\PycharmProjects\aim\18_model_aug\tensorrt_models\yolov26_int8.engine"
    ),
]


VIDEO = r"D:\PycharmProjects\computer_controller\video3.mp4"

IMG_SIZE = 640

TEST_SECONDS = 10

DEVICE = 0



# ============================================================
# Benchmark
# ============================================================

def benchmark(name, model_path):

    print("\n" + "=" * 90)
    print(f"测试模型：{name}")
    print(f"模型路径：{model_path}")
    print("=" * 90)


    # 加载模型
    model = YOLO(
        model_path,
        task="detect"
    )


    cap = cv2.VideoCapture(VIDEO)

    if not cap.isOpened():
        raise RuntimeError(
            f"无法打开视频：{VIDEO}"
        )


    ret, frame = cap.read()

    if not ret:
        cap.release()
        raise RuntimeError(
            "无法读取视频"
        )


    # ========================================================
    # Warmup
    # ========================================================

    print("Warmup...")

    for _ in range(30):

        model.predict(
            frame,
            imgsz=IMG_SIZE,
            device=DEVICE,
            verbose=False
        )


    print(
        f"开始测试 {TEST_SECONDS} 秒..."
    )


    # ========================================================
    # 正式测试
    # ========================================================

    times = []

    frames = 0

    start_total = time.perf_counter()


    while True:

        now = time.perf_counter()

        if now - start_total >= TEST_SECONDS:
            break


        ret, frame = cap.read()


        if not ret:

            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                0
            )

            continue



        start = time.perf_counter()


        model.predict(
            frame,
            imgsz=IMG_SIZE,
            device=DEVICE,
            verbose=False
        )


        end = time.perf_counter()


        elapsed_ms = (
            end - start
        ) * 1000


        times.append(
            elapsed_ms
        )

        frames += 1



    cap.release()



    # ========================================================
    # 统计
    # ========================================================

    times_sorted = sorted(times)


    avg = statistics.mean(
        times_sorted
    )


    p50 = times_sorted[
        int(len(times_sorted)*0.50)
    ]


    p95 = times_sorted[
        int(len(times_sorted)*0.95)
    ]


    p99 = times_sorted[
        int(len(times_sorted)*0.99)
    ]


    minimum = min(times_sorted)

    maximum = max(times_sorted)


    fps = 1000 / avg



    print()

    print(
        f"测试帧数     : {frames}"
    )

    print(
        f"平均推理时间 : {avg:.3f} ms"
    )

    print(
        f"P50          : {p50:.3f} ms"
    )

    print(
        f"P95          : {p95:.3f} ms"
    )

    print(
        f"P99          : {p99:.3f} ms"
    )

    print(
        f"最小         : {minimum:.3f} ms"
    )

    print(
        f"最大         : {maximum:.3f} ms"
    )

    print(
        f"理论 FPS     : {fps:.2f}"
    )



    return {

        "name": name,

        "avg": avg,

        "p50": p50,

        "p95": p95,

        "p99": p99,

        "min": minimum,

        "max": maximum,

        "fps": fps
    }





# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":


    print("=" * 100)

    print(
        "YOLO26 PyTorch / TensorRT Benchmark"
    )

    print("=" * 100)



    results = []



    for name, model_path in MODELS:


        if not Path(model_path).exists():

            print()

            print(
                f"跳过 {name}"
            )

            print(
                "不存在:"
            )

            print(
                model_path
            )

            continue



        result = benchmark(
            name,
            model_path
        )


        results.append(
            result
        )




    # ========================================================
    # 总表
    # ========================================================

    print("\n\n")


    print("=" * 110)


    print(
        f"{'模型':<18}"
        f"{'平均(ms)':>12}"
        f"{'P50':>12}"
        f"{'P95':>12}"
        f"{'P99':>12}"
        f"{'最小':>12}"
        f"{'最大':>12}"
        f"{'FPS':>12}"
    )


    print("-"*110)



    for r in results:


        print(

            f"{r['name']:<18}"

            f"{r['avg']:>12.3f}"

            f"{r['p50']:>12.3f}"

            f"{r['p95']:>12.3f}"

            f"{r['p99']:>12.3f}"

            f"{r['min']:>12.3f}"

            f"{r['max']:>12.3f}"

            f"{r['fps']:>12.2f}"

        )


    print("=" * 110)