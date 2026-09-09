"""
Auto-Aim 鼠标自动瞄准程序 (最终版)
==================================
基于 YOLO 目标检测 + PID 鼠标控制。
将对准鼠标最近的目标锁定并持续跟踪, 用 PID 将鼠标移到目标上,
对准后周期点击鼠标左键。

控制流程(每帧):
    1. 截屏 -> YOLO 检测
    2. 无目标 -> 重置状态
    3. 有目标 -> 锁定离鼠标最近的目标, 并连续性跟踪
    4. 用 PID 把鼠标移向锁定目标 (误差 = 目标 - 鼠标位置)
    5. 鼠标对准目标 -> 周期点击左键

快捷键:
    - ESC 单击: 暂停 / 继续鼠标移动
    - ESC 长按 1 秒: 退出程序
"""

import ctypes
import math
import time
from dataclasses import dataclass

import bettercam
import cv2
from ultralytics import YOLO


# =========================================================
# 配置区 (所有可调参数集中在这里)
# =========================================================

@dataclass
class Config:
    # ----- 模型 / 相机 -----
    model_path: str = r"D:\PycharmProjects\computer_controller\head_body_640_yolo26n_screen_full\weights\best.engine"

    # ----- 目标跟踪 -----
    max_distance: float = 50.0       # 连续帧判定为同一目标的最大距离(px)
    lost_reset_frames: int = 1       # 丢失目标多少帧后重新搜索

    # ----- PID 参数 -----
    pid_kp: float = 1
    pid_ki: float = 0.01
    pid_kd: float = 0.05

    # ----- 点击 -----
    click_frames: int = 2            # 鼠标对准目标连续多少帧后点击一次
    align_threshold: float = 10.0     # 误差小于该值视为"对准"(px)

    # ----- 平滑移动 -----
    smooth_steps: int = 1          # 平滑步数
    smooth_delay: float = 0.0001     # 每步延迟(秒)

    # ----- 键盘 -----
    esc_vk: int = 0x1B               # ESC 虚拟键码
    esc_exit_hold: float = 1.0       # 长按 ESC 多少秒后退出


CFG = Config()


# =========================================================
# Windows 底层 (鼠标)
# =========================================================

user32 = ctypes.windll.user32


class POINT(ctypes.Structure):
    """Windows POINT 结构: 鼠标坐标。"""
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


def get_mouse_pos() -> tuple[int, int]:
    """返回当前鼠标屏幕坐标 (x, y)。"""
    point = POINT()
    user32.GetCursorPos(ctypes.byref(point))
    return point.x, point.y


def send_move(dx: float, dy: float) -> None:
    """相对移动鼠标 dx, dy 像素。"""
    extra = ctypes.c_ulong(0)
    mi = MOUSEINPUT(
        int(dx), int(dy), 0, MOUSEEVENTF_MOVE, 0, ctypes.pointer(extra)
    )
    inp = INPUT(0, mi)
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def mouse_left_down() -> None:
    """鼠标左键按下。"""
    extra = ctypes.c_ulong(0)
    mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, ctypes.pointer(extra))
    inp = INPUT(0, mi)
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def mouse_left_up() -> None:
    """鼠标左键抬起。"""
    extra = ctypes.c_ulong(0)
    mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, ctypes.pointer(extra))
    inp = INPUT(0, mi)
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def mouse_left_click() -> None:
    """鼠标左键单击 (按下 + 抬起)。"""
    mouse_left_down()
    time.sleep(0.01)
    mouse_left_up()


# =========================================================
# 平滑移动
# =========================================================

def smooth_move(dx: float, dy: float, steps: int, delay: float) -> None:
    """
    将 dx, dy 拆分成 steps 步平滑移动鼠标 (带取整累加)。
    每步 sleep delay 秒以产生平滑轨迹。
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
            send_move(move_x, move_y)

        time.sleep(delay)


# =========================================================
# ESC 开关 (暂停/继续 + 长按退出)
# =========================================================

class EscController:
    """
    管理 ESC 键:
      - 单击: 暂停 <-> 继续 (停止/恢复鼠标移动)
      - 长按: 退出程序
    """

    def __init__(self, exit_hold_seconds: float):
        self.exit_hold_seconds = exit_hold_seconds
        self.paused = False
        self._last_state = False
        self._press_start: float | None = None

    def _is_pressed(self) -> bool:
        return bool(user32.GetAsyncKeyState(CFG.esc_vk) & 0x8000)

    def poll(self) -> bool:
        """
        处理一帧 ESC 状态。
        返回 True 表示应退出程序。
        """
        current = self._is_pressed()

        # 按下沿: 切换暂停状态
        if current and not self._last_state:
            self._press_start = time.time()
            self.paused = not self.paused
            print("ESC: 已暂停, 鼠标移动停止" if self.paused else "ESC: 已继续")

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
# PID 控制器
# =========================================================

class PID:
    def __init__(self, kp=1.2, ki=0.005, kd=0.05, integral_max=500.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_max = integral_max   # 积分硬限幅
        self.last_error = 0.0
        self.integral = 0.0

    def update(self, error):
        self.integral += error
        if abs(error) < 100:
            self.integral *= 0.1
        # 积分限幅: 防止长时间大误差下积分无限增长(积分饱和)
        self.integral = max(-self.integral_max,
                            min(self.integral_max, self.integral))
        derivative = error - self.last_error
        output = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )
        self.last_error = error
        return output

    def reset(self) -> None:
        self.last_error = 0.0
        self.integral = 0.0


# =========================================================
# 目标检测与锁定
# =========================================================

def load_model():
    """加载 YOLO 模型。"""
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


def _box_center(box):
    """返回检测框中心 (cx, cy)。"""
    cx, cy, _, _, _ = get_box_info(box)
    return cx, cy


def find_target_from_mouse(result):
    """选择距离鼠标当前位置最近的目标; 无则返回 None。"""
    mouse_x, mouse_y = get_mouse_pos()

    min_distance = float("inf")
    min_index = None

    for i, box in enumerate(result.boxes):
        cx, cy = _box_center(box)
        distance = math.hypot(cx - mouse_x, cy - mouse_y)
        if distance < min_distance:
            min_distance = distance
            min_index = i

    if min_index is None:
        return None
    return result.boxes[min_index]


def find_target_near_locked(result, locked_x: float, locked_y: float):
    """在距上一帧锁定位置 MAX_DISTANCE 内找最近目标; 无则 None。"""
    best = None
    best_distance = float("inf")

    for box in result.boxes:
        cx, cy = _box_center(box)
        distance = math.hypot(cx - locked_x, cy - locked_y)
        if distance <= CFG.max_distance and distance < best_distance:
            best_distance = distance
            best = box

    return best


# =========================================================
# 目标跟踪状态机
# =========================================================

class TargetTracker:
    """维护锁定目标与连续性跟踪, 处理目标丢失与重锁。"""

    def __init__(self):
        self.locked_x: float | None = None
        self.locked_y: float | None = None
        self.locked_w: float | None = None
        self.locked_h: float | None = None
        self.locked_conf: float | None = None
        self.tracking = False
        self.lost_frames = 0

    def select(self, result, frame_count: int) -> bool:
        """当前帧更新/选择目标; True=成功锁定可用, False=无目标。"""
        if not self.tracking:
            return self._lock_from_mouse(result, frame_count)
        return self._track(result, frame_count)

    def _lock_from_mouse(self, result, frame_count: int) -> bool:
        """初次锁定: 选离鼠标最近的目标。"""
        target = find_target_from_mouse(result)
        if target is None:
            return False
        self._set_locked(target)
        self.tracking = True
        self.lost_frames = 0
        print("=" * 70)
        print(f"NEW TARGET LOCKED  frame={frame_count}")
        print(
            f"  position=({self.locked_x:.2f},{self.locked_y:.2f}) "
            f"size={self.locked_w:.2f}x{self.locked_h:.2f} "
            f"conf={self.locked_conf:.4f}"
        )
        print("=" * 70)
        return True

    def _track(self, result, frame_count: int) -> bool:
        """锁定中: 在附近找最近目标; 丢失则重新搜索。"""
        target = find_target_near_locked(result, self.locked_x, self.locked_y)
        if target is not None:
            self._set_locked(target)
            self.lost_frames = 0
            return True

        # 丢失
        self.lost_frames += 1
        if self.lost_frames >= CFG.lost_reset_frames:
            self.tracking = False
            return self._lock_from_mouse(result, frame_count)
        return False

    def _set_locked(self, box) -> None:
        """用检测框更新锁定目标信息。"""
        (self.locked_x, self.locked_y,
         self.locked_w, self.locked_h, self.locked_conf) = get_box_info(box)

    def reset(self) -> None:
        """清空锁定状态。"""
        self.locked_x = None
        self.locked_y = None
        self.tracking = False
        self.lost_frames = 0


# =========================================================
# 鼠标操作 (对准点击)
# =========================================================

class Clicker:
    """鼠标对准目标后, 连续几帧一致就点击一次。"""

    def __init__(self, frames: int):
        self.frames = frames
        self._aligned_frames = 0

    def update(self, aligned: bool) -> None:
        """aligned=True 时累计, 达到阈值点击; 否则清零。"""
        if aligned:
            self._aligned_frames += 1
            if self._aligned_frames >= self.frames:
                mouse_left_click()
                self._aligned_frames = 0
        else:
            self._aligned_frames = 0


# =========================================================
# 主程序
# =========================================================

def main() -> None:
    """程序主入口。"""
    camera = bettercam.create(output_color="BGR")
    model = load_model()

    esc = EscController(CFG.esc_exit_hold)
    tracker = TargetTracker()
    clicker = Clicker(CFG.click_frames)
    pid_x = PID(CFG.pid_kp, CFG.pid_ki, CFG.pid_kd)
    pid_y = PID(CFG.pid_kp, CFG.pid_ki, CFG.pid_kd)

    frame_count = 0

    print("=" * 70)
    print("Auto-Aim 启动. 鼠标对准目标后自动点击.")
    print("ESC 单击暂停/继续, 长按 1 秒退出.")
    print("=" * 70)

    try:
        while True:
            frame_count += 1

            # ----- ESC 控制 (暂停/继续 + 长按退出) -----
            if esc.poll():
                break

            if esc.paused:
                # 暂停: 不做任何检测/移动/点击
                time.sleep(0.05)
                continue

            # ----- 截屏 -----
            frame = camera.grab()
            if frame is None:
                continue

            # ----- YOLO 检测 -----
            result = model(frame, verbose=False)[0]

            # ----- 无目标: 重置并继续 -----
            if len(result.boxes) == 0:
                tracker.reset()
                pid_x.reset()
                pid_y.reset()
                continue

            # ----- 目标选择/跟踪 -----
            if not tracker.select(result, frame_count):
                continue

            # ----- 鼠标位置 & 误差 -----
            mouse_x, mouse_y = get_mouse_pos()
            error_x = tracker.locked_x - mouse_x
            error_y = tracker.locked_y - mouse_y

            # ----- PID 输出 -----
            move_x = pid_x.update(error_x)
            move_y = pid_y.update(error_y)

            # ----- 对准判断 & 点击 -----
            aligned = (abs(error_x) < CFG.align_threshold
                       and abs(error_y) < CFG.align_threshold)
            clicker.update(aligned)

            # ----- 平滑移动鼠标 -----
            smooth_move(move_x, move_y, CFG.smooth_steps, CFG.smooth_delay)

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        cv2.destroyAllWindows()
        camera.release()
        print("程序结束")


if __name__ == "__main__":
    main()