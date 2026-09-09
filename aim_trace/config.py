from dataclasses import dataclass


@dataclass
class Config:
    # ===== 鼠标按下（单击）—— 请在此自行设置 =====
    align_threshold: float = 10.0          # 对准判定阈值(px)
    click_frames: int = 1                  # 连续对齐多少帧后触发一次单击
    click_down_interval: float = 0.01      # 按下与抬起之间的间隔(秒)
    click_repeat_interval: float = 0.25    # 两次单击之间的最小间隔(秒)

    # ===== 抓屏 =====
    capture_size: int = 640                # 只截取屏幕中心的方块边长(px)
    screen_origin: tuple = (0, 0)          # 抓屏区域左上角(自动计算)，用于坐标换算

    # ===== YOLO 并行 =====
    pipeline: bool = True                  # 真=后台线程跑 YOLO 与瞄准并行；假=串行等待推理

    # ===== 时间报告 =====
    report_dir: str = "aim_reports"
    timing_precision: int = 4

    # ===== 模型 =====
    model_path: str = r"D:\PycharmProjects\aim\18_model_aug\weights\best.engine"

    # ===== 目标最低可信度 =====
    conf_threshold: float = 0.5

    # ===== 类别（0=person_t, 1=head_t, 2=person_c, 3=head_c）=====
    enabled_classes: tuple = (1,)

    # ===== 目标跟踪 =====
    max_track_time: float = 1.0
    max_distance: float = 50.0
    lost_reset_frames: int = 2

    # ===== PID =====
    pid_kp: float = 0.6
    pid_ki: float = 0.3
    pid_kd: float = 0.1
    pid_y_scale: float = 0.7

    # ===== 前馈（目标速度）=====
    feedforward_x: float = 1.0
    feedforward_y: float = 0.5
    velocity_smooth: float = 0.7

    # ===== 轨迹预测（卡尔曼滤波）=====
    predict_enabled: bool = True          # 是否启用卡尔曼轨迹预测
    predict_lead_time: float = 0.12       # 预测提前量(秒)，即"截图→鼠标生效"总延迟
    kf_process_noise: float = 1e-2        # 过程噪声 Q（越大=越相信测量、越跟手，但也越抖）
    kf_measure_noise: float = 2e1         # 测量噪声 R（越大=越平滑、越滞后）

    # ===== 键盘 =====
    esc_vk: int = 0x4B
    vk_up: int = 0x26
    vk_down: int = 0x28
    vk_left: int = 0x25
    vk_right: int = 0x27
    esc_exit_hold: float = 0.5            # 长按 ESC 0.5 秒 = 干净退出
    esc_toggle_hold: float = 0.1


CFG = Config()