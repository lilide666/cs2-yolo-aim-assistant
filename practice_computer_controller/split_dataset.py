from pathlib import Path
import random
import shutil


# ============================
# 参数
# ============================

DATASET = Path(
    r"D:\PycharmProjects\computer_controller\dataset"
)


# 训练集比例

TRAIN_RATIO = 0.8


# 固定随机种子

SEED = 42



# ============================
# 划分函数
# ============================

def split_dataset():


    random.seed(SEED)


    image_source = DATASET / "images"

    label_source = DATASET / "labels"



    images = list(
        image_source.glob("*.jpg")
    )


    if len(images) == 0:

        raise Exception(
            "没有找到jpg图片"
        )



    random.shuffle(images)



    train_num = int(
        len(images) * TRAIN_RATIO
    )


    train_images = images[:train_num]

    val_images = images[train_num:]



    print("===================")

    print(
        f"总图片数量: {len(images)}"
    )

    print(
        f"训练集: {len(train_images)}"
    )

    print(
        f"验证集: {len(val_images)}"
    )

    print("===================")



    # ============================
    # 创建YOLO标准目录
    # ============================


    for split in [
        "train",
        "val"
    ]:


        (DATASET /
         "images" /
         split).mkdir(
            parents=True,
            exist_ok=True
        )


        (DATASET /
         "labels" /
         split).mkdir(
            parents=True,
            exist_ok=True
        )



    # ============================
    # 清空旧划分
    # ============================


    for split in [
        "train",
        "val"
    ]:


        for f in (
            DATASET /
            "images" /
            split
        ).glob("*"):

            f.unlink()



        for f in (
            DATASET /
            "labels" /
            split
        ).glob("*"):

            f.unlink()



    # ============================
    # 复制文件
    # ============================


    for split, files in [

        ("train", train_images),

        ("val", val_images)

    ]:


        for img in files:


            # 图片

            shutil.copy(

                img,

                DATASET /
                "images" /
                split /
                img.name

            )



            # 标签

            label = (

                label_source /
                (img.stem + ".txt")

            )


            if label.exists():

                shutil.copy(

                    label,

                    DATASET /
                    "labels" /
                    split /
                    label.name

                )



    print("数据集划分完成")



if __name__ == "__main__":

    split_dataset()