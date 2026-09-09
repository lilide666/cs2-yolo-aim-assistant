from ultralytics import YOLO
from win32process import ResumeThread

if __name__ == '__main__':
    model = YOLO(r"yolo26n.pt")

    model.train(
        data=r"D:\PycharmProjects\computer_controller\make_different_back\10_different_back_dataset\data.yaml",
        # 数据集配置文件
        # 里面定义 train、val 图片路径，以及类别名称和数量

        epochs=150,
        # 总训练轮数
        # 1 epoch = 整个训练集完整训练一遍

        patience=30,
        # Early Stopping（提前停止）
        # 如果验证集指标连续 20 个 epoch 没有明显提升，就提前结束训练

        imgsz=640,
        # 训练输入图片尺寸
        # 图片训练时会被处理成约 640×640
        # 越大通常越容易识别小目标，但显存占用和训练时间也会增加

        batch=16,
        # Batch Size（批次大小）
        # 每次同时拿 32 张图片进行一次参数更新
        # 越大越吃显存

        cache="disk",
        # 把数据集缓存到内存 RAM
        # 可以减少硬盘读取，提高数据加载速度
        # 但是会占用大量内存

        workers=4,
        # DataLoader 使用的 CPU 数据加载进程数量
        # 0 = 不创建额外子进程
        # Windows 下比较稳定，但数据加载速度可能稍慢


        # ---- 明暗增强（核心，针对漏检）----

        hsv_h=0.02,
        # Hue：色相增强
        # 随机改变图片的颜色
        # 例如红色稍微偏橙、蓝色稍微偏紫
        # 主要增加模型对不同颜色的适应能力

        hsv_s=0.9,
        # Saturation：饱和度增强
        # 让图片随机变得更鲜艳或者更灰
        # 0.9 属于比较强的增强

        hsv_v=0.8,
        # Value：明度/亮度增强
        # 随机改变图片亮度
        # 可以模拟明亮、阴暗、不同游戏画面亮度
        # 对 CS2 场景比较有意义


        # ---- 几何增强（游戏里人物会动/近大远小）----

        scale=0.6,
        # 随机缩放
        # 模拟人物距离变化
        # 例如：
        #   人物离得近 → 目标比较大
        #   人物离得远 → 目标比较小
        # 0.6 属于比较明显的缩放增强

        translate=0.2,
        # 随机平移
        # 把图片中的目标随机向上下左右移动
        # 0.2 ≈ 最多进行约 20% 的位置偏移
        # 可以避免模型只习惯画面中央的人物

        degrees=5.0,
        # 随机旋转角度
        # 最大大约 ±5°
        # CS2正常画面人物不会整体旋转，所以这里设置比较小

        shear=5.0,
        # Shear：错切/剪切变换
        # 对图片进行轻微的倾斜、拉伸
        # 用于增加模型对几何变化的适应能力

        perspective=0.0005,
        # Perspective：透视变换
        # 对图片进行轻微的透视扭曲
        # 可以模拟一定程度的视角变化
        # 0.0005 非常轻微


        # ---- 马赛克/混合 ----

        mosaic=1.0,
        # Mosaic：马赛克增强
        # 把多张图片拼到一张训练图片中
        #
        # 例如：
        #
        # ┌────────┬────────┐
        # │ 图片1  │ 图片2  │
        # ├────────┼────────┤
        # │ 图片3  │ 图片4  │
        # └────────┴────────┘
        #
        # 可以增加：
        #   多目标场景
        #   小目标
        #   不同目标位置
        #   不同目标大小
        #
        # 1.0 = 高概率使用 Mosaic

        mixup=0.2,
        # MixUp：图片混合增强
        # 把两张图片按照一定比例混合成一张新图片
        #
        # 例如：
        # 图片A + 图片B → 新图片
        #
        # 可以增加训练数据的变化
        # 但过高可能让图片变得不真实
        # 你这里 0.2 属于比较明显但不算极端

        copy_paste=0.3,
        # Copy-Paste：复制粘贴增强
        # 把一张图片中的目标复制出来
        # 再粘贴到另一张图片中
        #
        # 对人物数据集可以增加：
        #   人物数量
        #   人物位置
        #   人物组合
        #
        # 0.3 = 有一定强度的 Copy-Paste 增强

        project=r"D:\PycharmProjects\computer_controller\runs",

        name="18_model_aug",
    )


# ============================================================
# 下面是你之前的 YOLO26x 训练代码，目前被注释掉了
# ============================================================

# if __name__ == '__main__':
#
#     model = YOLO(
#         r"D:\PycharmProjects\computer_controller\yolo26x.pt"
#     )
#
#     model.train(
#
#         data=r"D:\PycharmProjects\computer_controller\make_dataset\body_head_dataset\data.yaml",
#         # 数据集 YAML 配置文件
#
#         epochs=80,
#         # 最大训练 80 个 epoch
#
#         patience=15,
#         # 连续 15 个 epoch 没有明显提升就提前停止
#
#         imgsz=1216,
#         # 输入尺寸 1216×1216
#         # 比 640 能保留更多小目标细节
#         # 但显存占用明显增加
#
#         batch=1,
#         # 每次只训练 1 张图片
#         # 因为 YOLO26x + 1216 非常吃显存
#
#         cache="ram",
#         # 将图片缓存到 RAM
#
#         workers=1,
#         # 使用 1 个 CPU 数据加载进程
#
#         project=r"D:\PycharmProjects\computer_controller\runs_1216",
#         # 训练结果保存目录
#
#         name="head_body_1216_yolo26x",
#         # 本次实验名称
#     )