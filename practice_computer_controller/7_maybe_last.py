"""
Auto-Aim 鼠标自动瞄准程序 (最终版 + 前馈追运动目标)
==================================================
基于 YOLO 目标检测 + PID + 目标速度前馈。

核心改进(解决"追不上匀速目标/慢半拍"):
    - 估计目标速度(velocity_x/y), 作为前馈加入移动输出
    - move = PID(误差) + 目标速度 × feedforward
    - 这样鼠标速度 ≈ 目标速度 + 修正量, 匀速目标不再滞后

运行时调节:
    - ↑↓: 调整当前PID项(无冷却), ←→: 切换 kp/ki/kd
ESC:
    - 单击: 暂停/继续(2s防连按), 长按1秒: 退出
"""

import ctypes
import math
import time
from dataclasses import dataclass

import bettercam
import cv2
from ultralytics import YOLO


# =========================================================
# 配置区
# =========================================================

@dataclass
class Config:
    # ----- 模型 / 相机 -----
    model_path: str = r"D:\PycharmProjects\computer_controller\best.pt"

    # ----- 目标跟踪 -----
    max_distance: float = 50.0
    lost_reset_frames: int = 1

    # ----- PID 参数 -----
    pid_kp: float = 0.8
    pid_ki: float = 0.05
    pid_kd: float = 0.1

    # ----- 前馈(追运动目标核心) -----
    feedforward_x: float = 2.0    # X 前馈系数
    feedforward_y: float = 1.0    # Y 前馈系数
    velocity_smooth: float = 0.5  # 速度估计平滑系数(0~1, 越大越跟手)

    # ----- 点击 -----
    click_frames: int = 1
    align_threshold: float = 10.0

    # ----- 平滑移动 -----
    smooth_steps: int = 1
    smooth_delay: float = 0.0001

    # ----- 键盘 -----
    esc_vk: int = 0x1B
    vk_up: int = 0x26
    vk_down: int = 0x28
    vk_left: int = 0x25
    vk_right: int = 0x27
    esc_exit_hold: float = 1.0
    esc_toggle_hold: float = 2.0


CFG = Config()


# =========================================================
# Windows 底层 (鼠标)
# =========================================================

user32 = ctypes.windll.user32


class POINT(ctypes.Structure):
    _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", ctypes.c_long),
        ("dy", ctypes.c_long),
        ("mouseData", ctypes.c_ulong),
        ("dwFlags", ctypes.c_ulong),
        ("time", ctypes.c_ulong),
        ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
    ]


class INPUT(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("mi", MOUSEINPUT)]


MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004


def get_mouse_pos() -> tuple[int, int]:
    point = POINT()
    user32.GetCursorPos(ctypes.byref(point))
    return point.x, point.y


def send_move(dx: float, dy: float) -> None:
    extra = ctypes.c_ulong(0)
    mi = MOUSEINPUT(
        int(dx), int(dy), 0, MOUSEEVENTF_MOVE, 0, ctypes.pointer(extra)
    )
    inp = INPUT(0, mi)
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def mouse_left_down() -> None:
    extra = ctypes.c_ulong(0)
    mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTDOWN, 0, ctypes.pointer(extra))
    inp = INPUT(0, mi)
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def mouse_left_up() -> None:
    extra = ctypes.c_ulong(0)
    mi = MOUSEINPUT(0, 0, 0, MOUSEEVENTF_LEFTUP, 0, ctypes.pointer(extra))
    inp = INPUT(0, mi)
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def mouse_left_click() -> None:
    mouse_left_down()
    time.sleep(0.01)
    mouse_left_up()


# =========================================================
# 平滑移动
# =========================================================

def smooth_move(dx: float, dy: float, steps: int, delay: float) -> None:
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
# 键盘边沿检测
# =========================================================

class Key:
    """单个按键的按下沿检测 (按下瞬间触发一次)。"""

    def __init__(self, vk: int):
        self.vk = vk
        self._last = False

    def just_pressed(self) -> bool:
        cur = bool(user32.GetAsyncKeyState(self.vk) & 0x8000)
        edge = cur and not self._last
        self._last = cur
        return edge


# =========================================================
# ESC 开关 (暂停/继续带冷却 + 长按退出)
# =========================================================

class EscController:
    def __init__(self, exit_hold_seconds, cooldown):
        self.exit_hold_seconds = exit_hold_seconds
        self.cooldown = cooldown
        self.paused = False
        self._last_state = False
        self._press_start = None
        self._last_toggle_time = 0.0

    def poll(self):
        cur = bool(user32.GetAsyncKeyState(CFG.esc_vk) & 0x8000)

        if cur and not self._last_state:
            self._press_start = time.time()
            now = time.time()
            if now - self._last_toggle_time >= self.cooldown:
                self._last_toggle_time = now
                self.paused = not self.paused
                print("ESC: 已暂停, 鼠标移动停止" if self.paused else "ESC: 已继续")

        if cur and self._last_state and self._press_start is not None:
            if time.time() - self._press_start >= self.exit_hold_seconds:
                print("ESC 长按, 程序退出")
                return True

        if not cur and self._last_state:
            self._press_start = None

        self._last_state = cur
        return False


# =========================================================
# PID 控制器 (带积分限幅 + reset)
# =========================================================

class PID:
    def __init__(self, kp, ki, kd, integral_max=500.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.integral_max = integral_max
        self.last_error = 0.0
        self.integral = 0.0

    def update(self, error):
        self.integral += error
        if abs(error) < 100:
            self.integral *= 0.1
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

    def reset(self):
        self.last_error = 0.0
        self.integral = 0.0


# =========================================================
# PID 运行时调节器 (无冷却, 暂停时也可调)
# =========================================================

class PIDTuner:
    def __init__(self, pid_x, pid_y):
        self.pid_x = pid_x
        self.pid_y = pid_y
        self._keys = {
            "up": Key(CFG.vk_up),
            "down": Key(CFG.vk_down),
            "left": Key(CFG.vk_left),
            "right": Key(CFG.vk_right),
        }
        self.current = "kp"
        self.step = 0.01

    def poll(self):
        if self._keys["up"].just_pressed():
            self._adjust(+self.step)
        elif self._keys["down"].just_pressed():
            self._adjust(-self.step)
        elif self._keys["left"].just_pressed():
            self._cycle(-1)
        elif self._keys["right"].just_pressed():
            self._cycle(1)

    def _cycle(self, direction):
        order = ["kp", "ki", "kd"]
        idx = order.index(self.current)
        self.current = order[(idx + direction) % len(order)]
        print(f">> 切换调整项: {self.current}  (↑↓调大小)")
        self._print_all()

    def _adjust(self, delta):
        for pid in (self.pid_x, self.pid_y):
            if self.current == "kp":
                pid.kp = max(0.0, pid.kp + delta)
            elif self.current == "ki":
                pid.ki = max(0.0, pid.ki + delta)
            elif self.current == "kd":
                pid.kd = max(0.0, pid.kd + delta)
        print(f">> {self.current} = {self._get_value():.3f}")
        self._print_all()

    def _get_value(self):
        pid = self.pid_x
        return {"kp": pid.kp, "ki": pid.ki, "kd": pid.kd}[self.current]

    def _print_all(self):
        print(f"    kp={self.pid_x.kp:.3f}  ki={self.pid_x.ki:.3f}  "
              f"kd={self.pid_x.kd:.3f}")


# =========================================================
# 目标检测与锁定
# =========================================================

def load_model():
    return YOLO(CFG.model_path)


def get_box_info(box):
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
    cx, cy, _, _, _ = get_box_info(box)
    return cx, cy


def find_target_from_mouse(result):
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


def find_target_near_locked(result, locked_x, locked_y):
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
# 目标跟踪状态机 (含目标速度估计, 用于前馈)
# =========================================================

class TargetTracker:
    def __init__(self):
        self.locked_x = None
        self.locked_y = None
        self.locked_w = None
        self.locked_h = None
        self.locked_conf = None
        self.tracking = False
        self.lost_frames = 0
        # 目标速度估计 (像素/帧), 用于前馈
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self._last_x = None
        self._last_y = None

    def select(self, result, frame_count):
        if not self.tracking:
            return self._lock_from_mouse(result, frame_count)
        return self._track(result, frame_count)

    def _lock_from_mouse(self, result, frame_count):
        target = find_target_from_mouse(result)
        if target is None:
            return False
        self._set_locked(target)
        # 首次锁定: 没有历史, 速度归零
        self._last_x = self.locked_x
        self._last_y = self.locked_y
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.tracking = True
        self.lost_frames = 0
        return True

    def _track(self, result, frame_count):
        target = find_target_near_locked(result, self.locked_x, self.locked_y)
        if target is not None:
            # 记录更新前位置, 用于算速度
            prev_x, prev_y = self.locked_x, self.locked_y
            self._set_locked(target)
            self._update_velocity(prev_x, prev_y)
            self.lost_frames = 0
            return True

        self.lost_frames += 1
        if self.lost_frames >= CFG.lost_reset_frames:
            self.tracking = False
            self.velocity_x = 0.0
            self.velocity_y = 0.0
            return self._lock_from_mouse(result, frame_count)
        return False

    def _set_locked(self, box):
        (self.locked_x, self.locked_y,
         self.locked_w, self.locked_h, self.locked_conf) = get_box_info(box)

    def _update_velocity(self, prev_x, prev_y):
        """用上/本帧位置差估计目标速度(带平滑)。"""
        if prev_x is not None:
            inst_vx = self.locked_x - prev_x
            inst_vy = self.locked_y - prev_y
            s = CFG.velocity_smooth
            self.velocity_x = inst_vx * s + self.velocity_x * (1 - s)
            self.velocity_y = inst_vy * s + self.velocity_y * (1 - s)

    def reset(self):
        self.locked_x = None
        self.locked_y = None
        self.tracking = False
        self.lost_frames = 0
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self._last_x = None
        self._last_y = None


# =========================================================
# 鼠标操作 (对准点击)
# =========================================================

class Clicker:
    def __init__(self, frames):
        self.frames = frames
        self._aligned_frames = 0

    def update(self, aligned):
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

def main():
    camera = bettercam.create(output_color="BGR")
    model = load_model()

    esc = EscController(CFG.esc_exit_hold, CFG.esc_toggle_hold)
    tracker = TargetTracker()
    clicker = Clicker(CFG.click_frames)
    pid_x = PID(CFG.pid_kp, CFG.pid_ki, CFG.pid_kd)
    pid_y = PID(CFG.pid_kp, CFG.pid_ki, CFG.pid_kd)

    tuner = PIDTuner(pid_x, pid_y)

    frame_count = 0

    print("=" * 70)
    print("Auto-Aim 启动 (PID + 前馈追目标).")
    print("  ↑↓ 调整当前PID项, ←→ 切换 kp/ki/kd")
    print("  ESC 单击暂停/继续(2s防连按), 长按 1 秒退出")
    print("-" * 70)
    tuner._print_all()
    print("=" * 70)

    try:
        while True:
            frame_count += 1

            # ----- ESC 控制 -----
            if esc.poll():
                break

            if esc.paused:
                tuner.poll()
                time.sleep(0.05)
                continue

            # ----- PID 运行时调节 -----
            tuner.poll()

            # ----- 截屏 -----
            frame = camera.grab()
            if frame is None:
                continue

            # ----- YOLO 检测 -----
            result = model(frame, verbose=False)[0]

            # ----- 无目标 -----
            if len(result.boxes) == 0:
                tracker.reset()
                pid_x.reset()
                pid_y.reset()
                continue

            # ----- 目标选择/跟踪 (含速度估计) -----
            if not tracker.select(result, frame_count):
                continue

            # ----- 鼠标位置 & 误差 -----
            mouse_x, mouse_y = get_mouse_pos()
            error_x = tracker.locked_x - mouse_x
            error_y = tracker.locked_y - mouse_y

            # ----- 控制输出 = PID + 目标速度前馈 -----
            move_x = pid_x.update(error_x) + tracker.velocity_x * CFG.feedforward_x
            move_y = pid_y.update(error_y) + tracker.velocity_y * CFG.feedforward_y

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