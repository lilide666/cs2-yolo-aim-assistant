from ultralytics import YOLO
from pathlib import Path
import cv2
import shutil
import random
import torch
import gc
from tqdm import tqdm


# ==========================
# 路径
# ==========================

MODEL = r"D:\PycharmProjects\computer_controller\runs_640\yolo26x_exp1-2\weights\best.pt"


SRC_IMAGES = Path(
    r"D:\PycharmProjects\computer_controller\dataset_all_frames\images"
)


OUT = Path(
    r"D:\PycharmProjects\computer_controller\dataset_predict"
)


# ==========================
# 参数
# ==========================

IMG_SIZE = 640

CONF = 0.3

DEVICE = 0

BATCH = 40

VAL_RATIO = 0.2


NAMES = {
    0:"head"
}


# ==========================
# 创建目录
# ==========================

for p in [

    OUT/"images/train",
    OUT/"images/val",

    OUT/"labels/train",
    OUT/"labels/val"

]:

    p.mkdir(
        parents=True,
        exist_ok=True
    )


# ==========================
# classes.txt
# ==========================

with open(
    OUT/"classes.txt.txt",
    "w",
    encoding="utf-8"
) as f:

    for k in NAMES:
        f.write(
            NAMES[k]+"\n"
        )


# ==========================
# 模型
# ==========================

model = YOLO(MODEL)

print("模型加载完成")


# ==========================
# 图片
# ==========================

images = sorted(
    list(SRC_IMAGES.glob("*.jpg"))
)


print("图片数量:",len(images))


train_count=0
val_count=0
empty=0



# ==========================
# split
# ==========================

def split():

    global train_count,val_count


    if random.random()<VAL_RATIO:

        val_count+=1
        return "val"

    else:

        train_count+=1
        return "train"



# ==========================
# 保存
# ==========================

def save_result(result,img_path):

    global empty


    img=cv2.imread(
        str(img_path)
    )


    if img is None:

        print(
            "读取失败:",
            img_path
        )

        return



    h,w=img.shape[:2]


    labels=[]


    for box in result.boxes:


        conf=float(
            box.conf[0]
        )


        if conf<CONF:
            continue



        cls=int(
            box.cls[0]
        )


        x1,y1,x2,y2=(
            box.xyxy[0]
            .cpu()
            .numpy()
        )


        xc=((x1+x2)/2)/w
        yc=((y1+y2)/2)/h

        bw=(x2-x1)/w
        bh=(y2-y1)/h



        labels.append(
            f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}"
        )



    if len(labels)==0:

        empty+=1
        return



    s=split()



    shutil.copy(
        img_path,
        OUT/"images"/s/img_path.name
    )


    with open(
        OUT/"labels"/s/(img_path.stem+".txt"),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "\n".join(labels)
        )



# ==========================
# 开始预测
# ==========================


print("开始预测")


for i in range(
    0,
    len(images),
    BATCH
):


    batch_images=images[i:i+BATCH]


    print(
        f"\n预测 {i}-{i+len(batch_images)}"
    )



    results=model.predict(

        source=batch_images,

        imgsz=IMG_SIZE,

        conf=CONF,

        batch=BATCH,

        device=DEVICE,

        verbose=False

    )



    for img,r in zip(
        batch_images,
        results
    ):

        save_result(
            r,
            img
        )



    del results

    gc.collect()

    torch.cuda.empty_cache()



# ==========================
# yaml
# ==========================


yaml=f"""

path: {OUT}

train: images/train

val: images/val


names:

"""


for k,v in NAMES.items():

    yaml+=f"  {k}: {v}\n"



with open(
    OUT/"data.yaml",
    "w",
    encoding="utf-8"
) as f:

    f.write(yaml)



print("====================")

print("train:",train_count)

print("val:",val_count)

print("无目标:",empty)

print("完成")