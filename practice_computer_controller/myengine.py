from ultralytics import YOLO

model = YOLO(r"D:\PycharmProjects\computer_controller\yolo26x.pt")

model.export(
    format="engine",
    imgsz=1600,
    device=0
)