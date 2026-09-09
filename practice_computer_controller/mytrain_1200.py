from ultralytics import YOLO
from win32process import ResumeThread

if __name__ == '__main__':
    model = YOLO(r"D:\PycharmProjects\computer_controller\runs_1216\yolo26x_exp1\weights\last.pt")
    model.train(
        data=r"D:\PycharmProjects\computer_controller\dataset_predict\data.yaml",
        epochs=80,
        patience=15,
        imgsz=1216,
        batch=10,
        cache="ram",
        workers=0,

        resume=True,

        project=r"D:\PycharmProjects\computer_controller\runs_1216",
        name="yolo26x_exp1",
    )
