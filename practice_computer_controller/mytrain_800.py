from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"yolo26x.pt")
    model.train(
        data=r"D:\PycharmProjects\computer_controller\dataset_predict\data.yaml",
        epochs=80,
        patience=10,
        imgsz=800,
        batch=3,
        cache="ram",
        workers=4,

        project=r"D:\PycharmProjects\work_yolo\runs_800",
        name="yolo26x_exp1",
    )
