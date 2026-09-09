"""

==================================================

YOLO 检测 + PID + 目标速度前馈

当前模型：class 0 = body，class 1 = head
本程序只追踪 class 1（head），完全忽略 class 0（body）

功能：
    - YOLO 检测
    - 只选择 head
    - PID 控制 + 目标速度前馈
    - 目标跟踪（单目标最多追踪固定时长，超时改选可信度最高目标）
    - 时间报告（明细 + 汇总统计，含 平均/中位/P95/P99/最大/最小，存 CSV 文件夹）
    - PID 运行时调节（↑↓ 调整，←→ 切换 kp/ki/kd）
    - ESC 单击：暂停/继续；ESC 长按：退出

"""

import csv
import ctypes
import math
import os
import time
from dataclasses import dataclass

import bettercam
import cv2
from ultralytics import YOLO


@dataclass
class Config:
    # ===== 鼠标按下（单击）—— 请在此自行设置 =====
    align_threshold: float = 10.0          # 对准判定阈值(px)
    click_frames: int = 1                  # 连续对齐多少帧后触发一次单击
    click_down_interval: float = 0.01      # 按下与抬起之间的间隔(秒)
    click_repeat_interval: float = 0.2     # 两次单击之间的最小间隔(秒)

    # ===== 时间报告 =====
    report_dir: str = "aim_reports"        # 时间报告存放文件夹
    timing_precision: int = 4              # 时间数值保留的小数位数（调大可看更细耗时）

    # ===== 模型 =====
    model_path: str = r"D:\PycharmProjects\computer_controller\runs_640\head_body_640_yolo26n\weights\best.engine"

    # ===== 目标类别（0=body, 1=head，这里只追踪 1）=====
    target_class: int = 1

    # ===== 目标跟踪 =====
    max_track_time: float = 2.0   # 单个目标最多追踪时长(秒)，超时改选可信度最高的目标
    max_distance: float = 50.0    # 目标附近搜索范围(px)
    lost_reset_frames: int = 1    # 丢失多少帧后重新选择

    # ===== PID =====
    pid_kp: float = 0.8
    pid_ki: float = 0.1
    pid_kd: float = 0.3

    # ===== 前馈 =====
    feedforward_x: float = 2.0
    feedforward_y: float = 1.0
    velocity_smooth: float = 0.5

    # ===== 平滑移动 =====
    smooth_steps: int = 1
    smooth_delay: float = 0.0001

    # ===== 键盘 =====
    esc_vk: int = 0x1B
    vk_up: int = 0x26
    vk_down: int = 0x28
    vk_left: int = 0x25
    vk_right: int = 0x27
    esc_exit_hold: float = 1.0
    esc_toggle_hold: float = 2.0


CFG = Config()

# ===== Windows 底层鼠标 =====
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
    mi = MOUSEINPUT(int(dx), int(dy), 0, MOUSEEVENTF_MOVE, 0, ctypes.pointer(extra))
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
    time.sleep(CFG.click_down_interval)
    mouse_left_up()


# ===== 时间统计（记录每次“找目标→按下”的各个动作耗时 + 汇总分析）=====
COL_NAMES = {
    1: "lock_to_click_ms", 3: "fps", 4: "grab_ms", 5: "yolo_ms",
    6: "select_ms", 7: "mouse_ms", 8: "pid_ms",
    9: "align_click_ms", 10: "move_ms",
}
DISP_NAMES = {
    1: "锁定→按下(ms)", 3: "帧率(fps)", 4: "截屏(ms)", 5: "YOLO(ms)",
    6: "目标处理(ms)", 7: "鼠标位置(ms)", 8: "PID(ms)",
    9: "对准+点击(ms)", 10: "移动(ms)",
}


def _percentile(sorted_vals, p):
    """线性插值求第 p 百分位（p 取 0~1）。"""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    k = (n - 1) * p
    lo = int(k)
    hi = min(lo + 1, n - 1)
    if hi == lo:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


class Stats:
    def __init__(self):
        self._lock_time = None    # 锁定目标的时间
        self._n = 0               # 锁定后经过的帧数
        self._report_count = 0
        self._last_times = {}     # 触发按下一帧时各动作的耗时(ms)
        self.click_rows = []      # 内存表：每次命中一条

    def mark_lock(self):
        self._lock_time = time.perf_counter()
        self._last_times = {}
        self._n = 0

    def add(self, name, seconds):
        self._last_times[name] = seconds * 1000.0

    def track_frame(self):
        self._n += 1

    def mark_click(self):
        if self._lock_time is None or self._n == 0:
            return
        self._report_count += 1
        p = CFG.timing_precision
        lock_to_click_ms = (time.perf_counter() - self._lock_time) * 1000.0
        fps = (self._n / lock_to_click_ms) * 1000.0 if lock_to_click_ms > 0 else 0.0
        lt = self._last_times
        self.click_rows.append([
            self._report_count,
            round(lock_to_click_ms, p),
            self._n,
            round(fps, p),
            round(lt.get("截屏", 0.0), p),
            round(lt.get("YOLO", 0.0), p),
            round(lt.get("目标处理", 0.0), p),
            round(lt.get("鼠标位置", 0.0), p),
            round(lt.get("PID", 0.0), p),
            round(lt.get("对准+点击", 0.0), p),
            round(lt.get("移动", 0.0), p),
        ])
        print(f"[命中 #{self._report_count}] 锁定→按下 {lock_to_click_ms:.4f} ms  "
              f"({self._n} 帧, {fps:.1f} fps)")

    def summarize(self, col):
        """对某一列取 平均/中位/P95/P99/最大/最小。"""
        vals = sorted(r[col] for r in self.click_rows)
        if not vals:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        mean = sum(vals) / len(vals)
        return (mean, _percentile(vals, .5), _percentile(vals, .95),
                _percentile(vals, .99), vals[-1], vals[0])

    def print_summary(self):
        n = len(self.click_rows)
        if n == 0:
            print("本次运行无命中记录")
            return
        print()
        print("═" * 72)
        print(f"  命中时间汇总（共 {n} 次）—— 主指标为“锁定→按下”端到端耗时")
        print("═" * 72)
        print(f" {'指标':<14}{'平均':>10}{'中位':>10}{'P95':>10}{'P99':>10}{'最大':>10}{'最小':>10}")
        for col, name in DISP_NAMES.items():
            mean, med, p95, p99, mx, mn = self.summarize(col)
            print(f" {name:<14}{mean:>10.2f}{med:>10.2f}{p95:>10.2f}{p99:>10.2f}{mx:>10.2f}{mn:>10.2f}")
        fvals = sorted(r[2] for r in self.click_rows)
        fmed = fvals[len(fvals)//2] if len(fvals) % 2 else \
            (fvals[len(fvals)//2-1] + fvals[len(fvals)//2]) / 2
        print(f" {'命中帧数(帧)':<14}{sum(fvals)/len(fvals):>10.1f}{fmed:>10.1f}"
              f"{_percentile(fvals, .95):>10.1f}{_percentile(fvals, .99):>10.1f}"
              f"{fvals[-1]:>10d}{fvals[0]:>10d}")
        print("═" * 72)

    def save_csv(self):
        """退出时写入 明细 + 汇总 两份报告到文件夹（不实时写盘，不影响速度）"""
        try:
            os.makedirs(CFG.report_dir, exist_ok=True)
            stamp = time.strftime('%Y%m%d_%H%M%S')
            f_detail = os.path.join(CFG.report_dir, f"hit_timing_{stamp}.csv")
            with open(f_detail, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["hit_no", "lock_to_click_ms", "frames", "fps",
                            "grab_ms", "yolo_ms", "select_ms", "mouse_ms",
                            "pid_ms", "align_click_ms", "move_ms"])
                w.writerows(self.click_rows)
            print(f"已保存命中明细 -> {f_detail} ({len(self.click_rows)} 条)")

            f_summary = os.path.join(CFG.report_dir, f"summary_{stamp}.csv")
            with open(f_summary, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["metric", "mean", "median", "p95", "p99", "max", "min"])
                for col, name in COL_NAMES.items():
                    mean, med, p95, p99, mx, mn = self.summarize(col)
                    w.writerow([name, round(mean, 4), round(med, 4),
                                round(p95, 4), round(p99, 4), round(mx, 4), round(mn, 4)])
                fvals = sorted(r[2] for r in self.click_rows)
                w.writerow(["frames", round(sum(fvals)/len(fvals), 2),
                            round(_percentile(fvals, .5), 2),
                            round(_percentile(fvals, .95), 2),
                            round(_percentile(fvals, .99), 2), fvals[-1], fvals[0]])
            print(f"已保存命中汇总 -> {f_summary} ({len(self.click_rows)} 条命中)")
        except Exception as e:
            print("保存时间报告失败:", e)


STATS = Stats()


# ===== 平滑移动 =====
def smooth_move(dx: float, dy: float, steps: int, delay: float) -> None:
    if steps <= 1:                       # 单步：省掉循环和累加器
        send_move(int(dx), int(dy))
        return
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


# ===== 键盘边沿检测 =====
class Key:
    def __init__(self, vk: int):
        self.vk = vk
        self._last = False

    def just_pressed(self) -> bool:
        cur = bool(user32.GetAsyncKeyState(self.vk) & 0x8000)
        edge = cur and not self._last
        self._last = cur
        return edge


# ===== ESC =====
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
                print("ESC: 已暂停，鼠标移动停止" if self.paused else "ESC: 已继续")
        if cur and self._last_state and self._press_start is not None:
            if time.time() - self._press_start >= self.exit_hold_seconds:
                print("ESC 长按，程序退出")
                return True
        if not cur and self._last_state:
            self._press_start = None
        self._last_state = cur
        return False


# ===== PID =====
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
        self.integral = max(-self.integral_max, min(self.integral_max, self.integral))
        derivative = error - self.last_error
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.last_error = error
        return output

    def reset(self):
        self.last_error = 0.0
        self.integral = 0.0


# ===== PID 调节器 =====
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
        print(f">> 切换调整项: {self.current} (↑↓调大小)")
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
        return {
            "kp": self.pid_x.kp,
            "ki": self.pid_x.ki,
            "kd": self.pid_x.kd,
        }[self.current]

    def _print_all(self):
        print(f"    kp={self.pid_x.kp:.3f}  ki={self.pid_x.ki:.3f}  kd={self.pid_x.kd:.3f}")


# ===== YOLO 加载 =====
def load_model():
    print(f"加载模型: {CFG.model_path}")
    model = YOLO(CFG.model_path)
    print(f"目标类别: class {CFG.target_class}")
    return model


# ===== Box 信息 =====
def get_box_info(box):
    # 仅在对最终选中的单个目标锁定时调用一次
    x1, y1, x2, y2 = box.xyxy[0]
    x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    conf = box.conf.item()
    cls = int(box.cls.item())
    return cx, cy, w, h, conf, cls


# ===== 一次遍历，过滤目标类别 + 预提取中心/置信度（纯 Python 浮点）=====
def get_candidates(result):
    """返回 [(box, cx, cy, conf), ...]，只含目标类别，每帧只调用一次。"""
    cands = []
    for box in result.boxes:
        if int(box.cls.item()) != CFG.target_class:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()   # 一次取 4 个坐标，减少边界调用
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        conf = box.conf.item()
        cands.append((box, cx, cy, conf))
    return cands


# ===== 目标选择（全部用预提取值 + 平方距离，避免重复遍历/重复 item/开方）=====
def find_target_from_mouse(cands):
    mouse_x, mouse_y = get_mouse_pos()
    min_d = float("inf")
    best = None
    for box, cx, cy, _ in cands:
        dx = cx - mouse_x
        dy = cy - mouse_y
        d = dx * dx + dy * dy
        if d < min_d:
            min_d = d
            best = box
    return best


def find_best_confidence(cands):
    best = None
    best_conf = -1.0
    for box, _, _, conf in cands:
        if conf > best_conf:
            best_conf = conf
            best = box
    return best


def find_target_near_locked(cands, locked_x, locked_y):
    best = None
    best_d = float("inf")
    max_d = CFG.max_distance
    max_d_sq = max_d * max_d
    for box, cx, cy, _ in cands:
        dx = cx - locked_x
        dy = cy - locked_y
        d = dx * dx + dy * dy
        if d <= max_d_sq and d < best_d:
            best_d = d
            best = box
    return best


# ===== 目标跟踪状态机 =====
class TargetTracker:
    def __init__(self):
        self.locked_x = None
        self.locked_y = None
        self.locked_w = None
        self.locked_h = None
        self.locked_conf = None
        self.locked_cls = None
        self.tracking = False
        self.lost_frames = 0
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self._last_x = None
        self._last_y = None
        self._lock_start = None  # 当前目标开始追踪的时间

    def select(self, cands):
        if not self.tracking:
            return self._lock_from_mouse(cands)
        # 单目标追踪时间达上限，改选可信度最高的目标
        if self._lock_start is not None and time.time() - self._lock_start >= CFG.max_track_time:
            return self._lock_best(cands)
        return self._track(cands)

    def _lock_from_mouse(self, cands):
        target = find_target_from_mouse(cands)
        if target is None:
            return False
        self._apply_lock(target)
        STATS.mark_lock()
        return True

    def _lock_best(self, cands):
        # 重新锁定可信度最高的目标
        target = find_best_confidence(cands)
        if target is None:
            return False
        self._apply_lock(target)
        STATS.mark_lock()
        return True

    def _apply_lock(self, box):
        self._set_locked(box)
        self._last_x = self.locked_x
        self._last_y = self.locked_y
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self.tracking = True
        self.lost_frames = 0
        self._lock_start = time.time()

    def _track(self, cands):
        target = find_target_near_locked(cands, self.locked_x, self.locked_y)
        if target is not None:
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
            return self._lock_from_mouse(cands)
        return False

    def _set_locked(self, box):
        (self.locked_x, self.locked_y, self.locked_w, self.locked_h,
         self.locked_conf, self.locked_cls) = get_box_info(box)

    def _update_velocity(self, prev_x, prev_y):
        if prev_x is None:
            return
        s = CFG.velocity_smooth
        self.velocity_x = (self.locked_x - prev_x) * s + self.velocity_x * (1 - s)
        self.velocity_y = (self.locked_y - prev_y) * s + self.velocity_y * (1 - s)

    def reset(self):
        self.locked_x = None
        self.locked_y = None
        self.locked_w = None
        self.locked_h = None
        self.locked_conf = None
        self.locked_cls = None
        self.tracking = False
        self.lost_frames = 0
        self.velocity_x = 0.0
        self.velocity_y = 0.0
        self._last_x = None
        self._last_y = None
        self._lock_start = None


# ===== 点击 =====
class Clicker:
    def __init__(self, frames, repeat_interval):
        self.frames = frames
        self.repeat_interval = repeat_interval
        self._aligned_frames = 0
        self._last_click_time = 0.0

    def update(self, aligned):
        if aligned:
            self._aligned_frames += 1
            now = time.time()
            if self._aligned_frames >= self.frames and now - self._last_click_time >= self.repeat_interval:
                mouse_left_click()
                STATS.mark_click()
                self._aligned_frames = 0
                self._last_click_time = now
        else:
            self._aligned_frames = 0


# ===== 主程序 =====
def main():
    camera = bettercam.create(output_color="BGR")
    model = load_model()

    esc = EscController(CFG.esc_exit_hold, CFG.esc_toggle_hold)
    tracker = TargetTracker()
    clicker = Clicker(CFG.click_frames, CFG.click_repeat_interval)
    pid_x = PID(CFG.pid_kp, CFG.pid_ki, CFG.pid_kd)
    pid_y = PID(CFG.pid_kp, CFG.pid_ki, CFG.pid_kd)
    tuner = PIDTuner(pid_x, pid_y)
    frame_count = 0

    print()
    print("=" * 70)
    print("Auto-Aim 启动")
    print("YOLO + PID + 前馈 + Head 追踪")
    print("-" * 70)
    print("模型:", CFG.model_path)
    print("追踪类别:", CFG.target_class, "(head)")
    print("忽略类别: 0 (body)")
    print("单目标最长追踪:", CFG.max_track_time, "秒")
    print("时间报告文件夹:", CFG.report_dir)
    print("-" * 70)
    print("↑↓ 调整 PID")
    print("←→ 切换 kp / ki / kd")
    print("ESC 单击：暂停/继续")
    print("ESC 长按：退出")
    print("-" * 70)
    tuner._print_all()
    print("=" * 70)

    try:
        while True:
            frame_count += 1

            # ESC
            if esc.poll():
                break

            # 暂停
            if esc.paused:
                tuner.poll()
                time.sleep(0.05)
                continue
            tuner.poll()

            # 截屏
            t = time.perf_counter()
            frame = camera.grab()
            if frame is None:
                continue
            STATS.add("截屏", time.perf_counter() - t)

            # YOLO
            t = time.perf_counter()
            result = model(frame, verbose=False)[0]
            STATS.add("YOLO", time.perf_counter() - t)

            # 一次遍历：只保留 Head 并预提取候选信息
            cands = get_candidates(result)
            if len(cands) == 0:
                tracker.reset()
                pid_x.reset()
                pid_y.reset()
                continue

            # 目标选择 / 跟踪
            t = time.perf_counter()
            if not tracker.select(cands):
                continue
            STATS.add("目标处理", time.perf_counter() - t)

            # 鼠标位置
            t = time.perf_counter()
            mouse_x, mouse_y = get_mouse_pos()
            error_x = tracker.locked_x - mouse_x
            error_y = tracker.locked_y - mouse_y
            STATS.add("鼠标位置", time.perf_counter() - t)

            # 误差
            dist = math.hypot(error_x, error_y)

            # 目标速度
            vel = math.hypot(tracker.velocity_x, tracker.velocity_y)
            STATS.track_frame()

            # PID + 前馈
            t = time.perf_counter()
            move_x = pid_x.update(error_x) + tracker.velocity_x * CFG.feedforward_x
            move_y = pid_y.update(error_y) + tracker.velocity_y * CFG.feedforward_y
            STATS.add("PID", time.perf_counter() - t)

            # 对准判断 + 点击触发
            t = time.perf_counter()
            aligned = abs(error_x) < CFG.align_threshold and abs(error_y) < CFG.align_threshold
            clicker.update(aligned)
            STATS.add("对准+点击", time.perf_counter() - t)

            # 鼠标移动
            t = time.perf_counter()
            smooth_move(move_x, move_y, CFG.smooth_steps, CFG.smooth_delay)
            STATS.add("移动", time.perf_counter() - t)

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        cv2.destroyAllWindows()
        camera.release()
        STATS.save_csv()
        STATS.print_summary()
        print("程序结束")


if __name__ == "__main__":
    main()