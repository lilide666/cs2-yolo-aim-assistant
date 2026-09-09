from ultralytics import YOLO

model = YOLO(r"yolo26x-seg.pt")
model.predict(
    source=r"D:\PycharmProjects\computer_controller\video3.mp4",
    save=False,
    show=True,
)

# model = YOLO(r"yolo11n-pose.pt")
# model.predict(
#     source=0,
#     save=False,
#     show=True,
# )