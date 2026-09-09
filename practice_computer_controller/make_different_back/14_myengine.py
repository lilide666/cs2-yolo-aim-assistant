from ultralytics import YOLO

model = YOLO(r"D:\PycharmProjects\computer_controller\runs\18_model_aug\weights\best.pt")

model.export(
    format="engine",
    imgsz=640,
    device=0
)