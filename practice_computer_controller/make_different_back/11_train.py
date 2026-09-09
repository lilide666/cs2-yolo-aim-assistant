from ultralytics import YOLO
from win32process import ResumeThread

if __name__ == '__main__':
    model = YOLO(r"D:\PycharmProjects\computer_controller\yolo26n.pt")
    model.train(
        data=r"D:\PycharmProjects\computer_controller\make_different_back\10_different_back_dataset\data.yaml",
        epochs=300,
        patience=20,
        imgsz=640,
        batch=32,
        cache="ram",
        workers=0,


        project=r"D:\PycharmProjects\computer_controller\runs",
        name="13_model",
    )


# if __name__ == '__main__':
#     model = YOLO(r"D:\PycharmProjects\computer_controller\yolo26x.pt")
#     model.train(
#         data=r"D:\PycharmProjects\computer_controller\make_dataset\body_head_dataset\data.yaml",
#         epochs=80,
#         patience=15,
#         imgsz=1216,
#         batch=1,
#         cache="ram",
#         workers=1,
#
#         project=r"D:\PycharmProjects\computer_controller\runs_1216",
#         name="head_body_1216_yolo26x",
#     )
