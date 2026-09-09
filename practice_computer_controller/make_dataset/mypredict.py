from ultralytics import YOLO

model = YOLO(r"D:\PycharmProjects\computer_controller\runs\18_model_aug\weights\best.pt")
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