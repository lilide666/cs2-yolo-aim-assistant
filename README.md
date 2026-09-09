# CS2 YOLO 瞄准助手

基于 YOLO 目标检测的 CS2（Counter-Strike 2）自动瞄准练习项目：截取屏幕中心画面，用 YOLO 检测游戏内人员与头部（类别 0=person_t、1=head_t、2=person_c、3=head_c），经目标选择与跟踪、PID + 前馈控制计算鼠标位移，通过 Windows SendInput 移动鼠标，并在对准目标时自动点击。

## 目录结构

| 路径 | 说明 |
| --- | --- |
| `aim_PID/` | 主程序（PID 版本）：抓屏 → YOLO 推理（后台线程并行）→ 目标跟踪 → PID + 前馈 → 鼠标移动 / 点击 |
| `aim_trace/` | 轨迹预测版本：在 PID 版基础上加入卡尔曼滤波（`kalman.py`）预测目标运动 |
| `0_aim_check_screen_move.py` ~ `7_aim_PID.py` | 迭代过程脚本，编号即演进顺序（截屏检查、模型学习、耗时测量、无等待、最短耗时、双模型等） |
| `18_model_aug/` | 增强训练的模型输出：权重（`best.pt` / `last.pt` / ONNX）、训练曲线、混淆矩阵与验证结果图 |
| `head_body_640_yolo26n_screen_640/` | 640 输入模型的训练输出与权重 |
| `myengine.py`、`mypredict.py` | TensorRT 引擎导出与推理测试脚本 |
| `practice_computer_controller/` | 本项目早期的思考、尝试与实践过程存档，详见该目录下的 README |

## 运行方式

- Python 3.11，主要依赖：`ultralytics`、`opencv-python`、`bettercam`（屏幕截取）、`numpy`；使用 TensorRT 引擎推理时另需安装 `tensorrt`。
- 入口：`python -m aim_PID.main`（PID 版）或 `python -m aim_trace.main`（轨迹预测版）。
- 模型路径在对应包的 `config.py` 中通过 `model_path` 配置，默认指向 TensorRT 引擎；也可直接改为 `weights/best.pt` 或 `weights/best.onnx` 运行。
- 运行中可通过方向键调整 PID 参数，`K` 键切换暂停 / 退出（见 `config.py` 键盘配置）。

## 未上传的文件

- `video3.mp4`（约 142MB 测试录屏）：超过 GitHub 单文件 100MB 限制，未上传。
- `*.engine`（TensorRT 引擎，23~57MB / 个）：与本机 GPU、TensorRT 版本绑定，更换机器需重新导出，未上传；可使用各 `weights/` 目录下的 `.pt` / `.onnx` 自行导出。
