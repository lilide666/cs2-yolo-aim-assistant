# 类别名称表
CLASS_NAMES = {
    0: "person_t",
    1: "head_t",
    2: "person_c",
    3: "head_c",
}


def get_class_name(class_id):
    return CLASS_NAMES.get(class_id, f"class_{class_id}")


# ===== 鼠标事件标志 =====
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

# ===== 键盘虚拟键码 =====
VK_W = 0x57
VK_A = 0x41
VK_S = 0x53
VK_D = 0x44
KEYEVENTF_KEYUP = 0x0002

# ===== 时间统计：每步名称 / 显示名 / CSV 列名 / 直方图分箱 =====
STEP_NAMES = ["截屏", "YOLO", "目标处理", "鼠标位置", "PID", "对准+点击", "移动"]
DISP_NAMES = {
    1: "锁定→按下(ms)", 3: "帧率(fps)", 4: "截屏(ms)", 5: "YOLO(ms)",
    6: "目标处理(ms)", 7: "鼠标位置(ms)", 8: "PID(ms)",
    9: "对准+点击(ms)", 10: "移动(ms)",
}
COL_NAMES = {
    1: "lock_to_click_ms", 3: "fps", 4: "grab_ms", 5: "yolo_ms",
    6: "select_ms", 7: "mouse_ms", 8: "pid_ms",
    9: "align_click_ms", 10: "move_ms",
}
HIST_BINS = [
    (0, 10, "<10"), (10, 25, "10~25"), (25, 50, "25~50"),
    (50, 100, "50~100"), (100, 250, "100~250"),
    (250, 500, "250~500"), (500, 1000, "500~1000"),
    (1000, float("inf"), ">1000"),
]