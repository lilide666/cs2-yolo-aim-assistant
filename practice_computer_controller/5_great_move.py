"""
Auto-Aim 鼠标自动瞄准程序
=========================
基于 YOLO 目标检测 + 鼠标自动控制，将检测到的目标锁定并移动鼠标到屏幕中心，
对准后周期性点击鼠标左键。

控制策略:
    - X 方向: 大位移用解析公式一步到位, 精修阶段用 PID 逼近
    - Y 方向: 始终用 PID 逼近

快捷键:
    - ESC 单击: 暂停 / 继续
    - ESC 长按 1 秒: 退出程序
"""

import ctypes
import math
import time
from dataclasses import dataclass

import bettercam
from ultralytics import YOLO


# =========================================================
# 配置区 (修改参数只需要看这里)
# =========================================================

@dataclass
class Config:
    # ----- YOLO 模型 -----
    model_path: str = (
        r"D:\PycharmProjects\computer_controller"
        r"\runs_1200\yolo26x_exp1\weights\best.pt"
    )

    # ----- 屏幕中心 / 容差 -----
    center_x: int = 1280
    center_y: int = 800
    tolerance: int = 20          # 判定已对准中心的最大误差(px)

    # ----- 水平大位移 (解析公式) -----
    input_dx: int = 50           # 每次大位移的鼠标输入量(px)
    formula_a: float = 53.4      # 对数公式增益
    formula_emax: float = 952.0  # 对数公式分子常数
    formula_c: float = 932.0     # 对数公式分母偏移

    # ----- PID 参数 -----
    pid_kp: float = 10
    pid_ki: float = 1
    pid_kd: float = 0.2
    pid_max_x: float = 200.0     # X 方向 PID 单次输出上限
    pid_max_y: float = 200.0     # Y 方向 PID 单次输出上限
    pid_x_range: float = 100.0   # X 大位移 → PID 精修的切换阈值(px)

    # ----- 周期点击 -----
    click_frames: int = 3        # 对准后需稳定帧数才开始点击
    click_interval: float = 0.5  # 点击间隔(秒)

    # ----- 目标跟踪 -----
    max_distance: float = 50.0   # 连续帧判定为同一目标的最大距离(px)

    # ----- 平滑移动 -----
    smooth_steps: int = 100
    smooth_delay: float = 0.0001

    # ----- 键盘 -----
    esc_vk: int = 0x1B           # ESC 虚拟键码
    exit_hold_seconds: float = 1.0  # 长按 ESC 退出所需秒数


CFG = Config()


# =========================================================
# Windows 键盘 / 鼠标底层
# =========================================================

user32 = ctypes.windll.user32


class POINT(ctypes.Structure):
    """Windows POINT 结构：鼠标坐标。"""
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class MOUSEINPUT(ctypes.Structure):
    """Windows MOUSEINPUT 结构。"""
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class INPUT(ctypes.Structure):
    """Windows INPUT 结构。"""
    _fields_ = [("type", ctypes.c_ulong), ("mi", MOUSEINPUT)]


# Windows 鼠标事件标志
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004


def _get_mouse_pos() -> tuple[int, int]:
    """返回当前鼠标屏幕坐标 (x, y)。"""
    point = POINT()
    user32.GetCursorPos(ctypes.byref(point))
    return point.x, point.y


def _send_move(dx: int, dy: int) -> None:
    """通过 SendInput 相对移动鼠标 dx, dy 像素。"""
    extra = ctypes.c_ulong(0)
    mi = MOUSEINPUT(int(dx), int(dy), 0, MOUSEEVENTF_MOVE, 0, ctypes.pointer(extra))
    inp = INPUT(0, mi)
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def _mouse_left_down() -> None:
    """鼠标左键按下。"""
    extra = ctypes.c_ulong(0)
    mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, ctypes.pointer(extra))
    inp = INPUT(0, mi)
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def _mouse_left_up() -> None:
    """鼠标左键抬起。"""
    extra = ctypes.c_ulong(0)
    mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, ctypes.pointer(extra))
    inp = INPUT(0, mi)
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def mouse_left_click() -> None:
    """鼠标左键单击（按下 + 短暂间隔 + 抬起）。"""
    _mouse_left_down()
    time.sleep(0.01)
    _mouse_left_up()


# =========================================================
# ESC 处理 (暂停/继续 + 长按退出)
# =========================================================

class EscController:
    """管理 ESC 键: 单击切换暂停, 长按退出程序。"""

    def __init__(self, exit_hold_seconds: float):
        self.exit_hold_seconds = exit_hold_seconds
        self.paused = False
        self._last_state = False
        self._press_start: float | None = None

    def _is_pressed(self) -> bool:
        return bool(user32.GetAsyncKeyState(CFG.esc_vk) & 0x8000)

    def poll(self) -> bool:
        """
        处理一帧的 ESC 状态。
        返回 True 表示应退出程序。
        """
        current = self._is_pressed()

        # 按下沿: 切换暂停状态
        if current and not self._last_state:
            self._press_start = time.time()
            self.paused = not self.paused
            print("ESC: 已暂停" if self.paused else "ESC: 已继续")

        # 保持按住: 检查长按退出
        if current and self._last_state and self._press_start is not None:
            if time.time() - self._press_start >= self.exit_hold_seconds:
                print("ESC 长按, 程序退出")
                return True

        # 松开沿: 清除按下时间
        if not current and self._last_state:
            self._press_start = None

        self._last_state = current
        return False


# =========================================================
# 平滑移动
# =========================================================

def smooth_move(dx: float, dy: float, steps: int, delay: float) -> bool:
    """
    将 dx, dy 拆分成 steps 步平滑移动鼠标（带取整累加）。
    返回 True 表示移动完整执行完成。
    """
    step_x = dx / steps
    step_y = dy / steps
    acc_x = 0.0
    acc_y = 0.0

    for _ in range(steps):
        acc_x += step_x
        acc_y += step_y

        move_x = int(acc_x)
        move_y = int(acc_y)

        acc_x -= move_x
        acc_y -= move_y

        if move_x != 0 or move_y != 0:
            _send_move(move_x, move_y)

        time.sleep(delay)

    return True


# =========================================================
# PID 控制器
# =========================================================

class PID:
    """带积分衰减与微分项的 PID 控制器。"""

    def __init__(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.last_error = 0.0
        self.integral = 0.0

    def update(self, error: float) -> float:
        """根据当前误差返回控制输出。"""
        self.integral += error
        if abs(error) < 100:
            self.integral *= 0.3

        derivative = error - self.last_error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative

        self.last_error = error
        return output

    def reset(self) -> None:
        """清空历史状态。"""
        self.last_error = 0.0
        self.integral = 0.0


# =========================================================
# 水平移动计算 (解析公式)
# =========================================================

def parse_move_steps(target_x: float) -> int:
    """
    根据目标 X 与屏幕中心的距离, 计算大位移所需的移动次数。
    返回 0 表示已在容差内。
    """
    e = abs(target_x - CFG.center_x)
    if e <= CFG.tolerance:
        return 0
    n = CFG.formula_a * math.log(CFG.formula_emax / (e + CFG.formula_c))
    return math.ceil(n)


# =========================================================
# YOLO 检测
# =========================================================

def load_model():
    return YOLO(CFG.model_path)


def get_box_info(box) -> tuple[float, float, float, float, float]:
    """返回检测框的 (中心x, 中心y, 宽, 高, 置信度)。"""
    x1, y1, x2, y2 = box.xyxy[0]
    x1 = x1.item()
    y1 = y1.item()
    x2 = x2.item()
    y2 = y2.item()

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    conf = box.conf.item()

    return cx, cy, w, h, conf


# =========================================================
# 目标选择算法
# =========================================================

def find_target_from_mouse(result):
    """初次寻找 / 重新锁定: 选择距离鼠标当前位置最近的目标。"""
    mouse_x, mouse_y = _get_mouse_pos()

    min_distance = float("inf")
    min_index = None

    for i, box in enumerate(result.boxes):
        cx, cy, _, _, _ = get_box_info(box)
        distance = math.hypot(cx - mouse_x, cy - mouse_y)
        if distance < min_distance:
            min_distance = distance
            min_index = i

    if min_index is None:
        return None
    return result.boxes[min_index]


def find_target_from_locked(result, locked_x: float, locked_y: float):
    """
    已锁定期间: 选择距上一帧锁定位置最近且在 MAX_DISTANCE 内的目标。
    若附近无目标则返回 None。
    """
    best = None
    best_distance = float("inf")

    for box in result.boxes:
        cx, cy, _, _, _ = get_box_info(box)
        distance = math.hypot(cx - locked_x, cy - locked_y)
        if distance <= CFG.max_distance and distance < best_distance:
            best_distance = distance
            best = box

    return best


# =========================================================
# 目标追踪状态机
# =========================================================

class TargetTracker:
    """维护锁定目标及其连续性跟踪。"""

    def __init__(self):
        self.locked_x: float | None = None
        self.locked_y: float | None = None
        self.tracking = False

    def select(self, result) -> bool:
        """
        当前帧选择目标。返回 True 表示选到了目标。
        优先保持原有锁定; 丢失后在鼠标附近重新寻找。
        """
        if self.tracking and self.locked_x is not None:
            target = find_target_from_locked(result, self.locked_x, self.locked_y)
            if target is not None:
                self.locked_x, self.locked_y, _, _, _ = get_box_info(target)
                return True
            # 跟踪丢失: 重新寻找
            self.tracking = False

        if not self.tracking:
            target = find_target_from_mouse(result)
            if target is not None:
                self.locked_x, self.locked_y, _, _, _ = get_box_info(target)
                self.tracking = True
                return True

        return False

    def reset(self):
        """清空锁定状态。"""
        self.locked_x = None
        self.locked_y = None
        self.tracking = False


# =========================================================
# 点击器 (周期点击)
# =========================================================

class Clicker:
    """目标对准后按固定间隔周期点击鼠标左键。"""

    def __init__(self, frames: int, interval: float):
        self.frames = frames
        self.interval = interval
        self._aligned_frames = 0
        self._last_click_time = 0.0

    def update(self, centered: bool):
        """centered=True 时累计对准帧并周期性点击。"""
        if centered:
            self._aligned_frames += 1
            now = time.time()
            if (self._aligned_frames >= self.frames
                    and now - self._last_click_time >= self.interval):
                mouse_left_click()
                self._last_click_time = now
        else:
            self._aligned_frames = 0

    def reset(self):
        self._aligned_frames = 0


# =========================================================
# 主控制逻辑
# =========================================================

def compute_move(
    target_x: float,
    target_y: float,
    pid_x: PID,
    pid_y: PID,
) -> tuple[float, float]:
    """
    根据目标位置计算 X / Y 方向本次应移动的量 (像素)。
    X: 大位移用解析公式, 精修用 PID;  Y: 始终 PID。
    """
    error_x = target_x - CFG.center_x
    error_y = target_y - CFG.center_y

    # ----- X 方向 -----
    if abs(error_x) <= CFG.pid_x_range:
        # 精修阶段: PID 逼近
        if abs(error_x) <= CFG.tolerance:
            move_x = 0.0
        else:
            move_x = pid_x.update(error_x)
            move_x = max(-CFG.pid_max_x, min(CFG.pid_max_x, move_x))
    else:
        # 大位移阶段: 解析公式一步到位 (不使用 PID)
        pid_x.reset()
        steps = parse_move_steps(target_x)
        move_x = -steps * CFG.input_dx if error_x > 0 else steps * CFG.input_dx

    # ----- Y 方向 -----
    if abs(error_y) <= CFG.tolerance:
        move_y = 0.0
    else:
        move_y = pid_y.update(error_y)
        move_y = max(-CFG.pid_max_y, min(CFG.pid_max_y, move_y))

    return move_x, move_y


def main() -> None:
    """程序主入口。"""
    # 初始化资源
    camera = bettercam.create(output_color="BGR")
    model = load_model()

    esc = EscController(CFG.exit_hold_seconds)
    tracker = TargetTracker()
    clicker = Clicker(CFG.click_frames, CFG.click_interval)
    pid_x = PID(CFG.pid_kp, CFG.pid_ki, CFG.pid_kd)
    pid_y = PID(CFG.pid_kp, CFG.pid_ki, CFG.pid_kd)

    moving = False  # 移动锁

    print("=" * 60)
    print("Auto-Aim 启动. ESC 单击暂停/继续, 长按退出.")
    print("=" * 60)

    try:
        while True:
            # ----- 键盘控制 -----
            if esc.poll():
                break

            if esc.paused:
                time.sleep(0.05)
                continue

            if moving:
                continue

            # ----- 采集与检测 -----
            frame = camera.grab()
            if frame is None:
                continue

            result = model(frame, verbose=False)[0]

            # 无目标
            if len(result.boxes) == 0:
                tracker.reset()
                pid_x.reset()
                pid_y.reset()
                clicker.reset()
                continue

            # ----- 目标选择 -----
            if not tracker.select(result):
                continue

            # ----- 计算移动量 -----
            move_x, move_y = compute_move(
                tracker.locked_x, tracker.locked_y, pid_x, pid_y
            )

            # 是否已对准中心
            centered = (abs(tracker.locked_x - CFG.center_x) <= CFG.tolerance
                        and abs(tracker.locked_y - CFG.center_y) <= CFG.tolerance)

            clicker.update(centered)

            if centered:
                continue

            if move_x == 0 and move_y == 0:
                continue

            # ----- 执行移动 -----
            moving = True
            smooth_move(move_x, move_y, CFG.smooth_steps, CFG.smooth_delay)
            moving = False

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        camera.release()
        print("程序结束")


if __name__ == "__main__":
    main()