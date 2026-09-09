from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r"D:\PycharmProjects\work_yolo\26x_640_1\train\weights\best.pt")
    model.val(
        data=r"D:\PycharmProjects\work_yolo\data_yolo_5\data.yaml",
        split="val",
        imgsz=640,
        batch=4,
    )
