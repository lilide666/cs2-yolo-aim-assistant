from ultralytics import YOLO

model = YOLO(r"D:\PycharmProjects\computer_controller\runs_1216\head_body_1216_yolo26x\weights\best.pt")

model.export(
    format="engine",
    imgsz=640,
    device=0
)