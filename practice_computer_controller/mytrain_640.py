from ultralytics import YOLO

if __name__ == '__main__':

    model = YOLO(r"yolo26n.pt")
    model.train(
        data=r"D:\PycharmProjects\computer_controller\dataset_predict\data.yaml",
        epochs=80,
        patience=20,
        imgsz=800,

        batch=24,
        cache="ram",
        workers=2,

        project=r"D:\PycharmProjects\computer_controller\runs_640",
        name="yolo26x_exp1",

        resume=True
    )


