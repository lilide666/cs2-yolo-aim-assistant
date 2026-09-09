# 早期思考与尝试实践过程（存档）

本目录保存的是 `aim_PID/`、`aim_trace/` 主程序成型**之前**的探索过程代码，属于思考、尝试、实践的记录，不是可直接使用的成品。文件名中的数字编号大致反映迭代顺序，其中保留了当时的各种试错版本，包括重复、废弃与一次性脚本。

## 内容

- 根目录 `1_` ~ `11_` 编号脚本、`cs2*.py`、`main.py` 等：从最初的 PID 寻的、最短路径移动、锁头、耗时分析到双模型推理的各个版本。
- `learn_mouse_move/`：鼠标移动量学习 / 标定的早期实验，含采集数据 `motion_data.csv`、`aim_grid_8x8.json`。
- `make_dataset/`：数据集制作流水线——从录屏抽帧、合成 640 输入图、AI 辅助标注（`ai_labelimg.py`）到切分 YOLO 数据集。
- `make_different_back/`：不同地图背景、不同输入尺寸下的数据集制作与训练实验脚本；`aim_reports/` 中保留了当时的运行统计记录。
- 其余 `*.py`：训练（`mytrain*.py`）、验证（`myval.py`）、数据集转换（`video_to_yolo_dataset_*.py`、`split_dataset.py`）等工具脚本。

## 未上传的文件

以下文件体积过大，均未上传：

- 模型权重与引擎：`*.pt`、`*.onnx`、`*.engine`（最大的 yolo26x 引擎约 1.6GB）。
- 数据集图片与标注：`make_dataset/`、`make_different_back/` 下的图片集（合计数十 GB）。
- 录屏视频：`*.mp4`（含 `video3.mp4`、`back.mp4` 等，单个最大约 1.27GB）。
