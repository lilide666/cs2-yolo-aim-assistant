from ultralytics import YOLO
import shutil
from pathlib import Path

PT_MODEL = Path(
    r"D:\PycharmProjects\aim\18_model_aug\weights\best.pt"
)

DATA_YAML = r"D:\PycharmProjects\computer_controller\make_different_back\10_different_back_dataset\data.yaml"

OUT_DIR = PT_MODEL.parent.parent / "tensorrt_models"
OUT_DIR.mkdir(exist_ok=True)

model = YOLO(str(PT_MODEL))

# FP32
path = model.export(
    format="engine",
    imgsz=640,
    device=0,
    half=False,
    int8=False,
    batch=1,
    dynamic=False,
)

shutil.copy2(
    path,
    OUT_DIR / "yolov26_fp32.engine"
)

# FP16
path = model.export(
    format="engine",
    imgsz=640,
    device=0,
    half=True,
    int8=False,
    batch=1,
    dynamic=False,
)

shutil.copy2(
    path,
    OUT_DIR / "yolov26_fp16.engine"
)

# INT8
path = model.export(
    format="engine",
    imgsz=640,
    device=0,
    half=False,
    int8=True,
    data=DATA_YAML,
    batch=1,
    dynamic=False,
)

shutil.copy2(
    path,
    OUT_DIR / "yolov26_int8.engine"
)

print(f"\n模型全部保存到：{OUT_DIR}")