"""

==================================================

YOLO 检测 + PID + 目标速度前馈 + 多类别过滤 + 置信度 + 退出
（已移除 WASD 持续按键功能）

模型类别：
    0 = person_t
    1 = head_t          ← 默认只启用这个
    2 = person_c
    3 = head_c

功能：
    - YOLO 检测（可后台线程并行，隐藏推理延迟）
    - 多类别多选追踪（可随时在控制台开关类别，默认只启用 class 1）
    - 目标最低可信度阈值调节
    - 每个菜单面板末尾都有"返回"选项（回到顶层重新选择，不退出程序）
    - PID 控制 + 目标速度前馈（跟紧移动目标）
    - 目标跟踪（单目标最多追踪固定时长，超时改选可信度最高目标）
    - 只抓取屏幕中心的 capture_size 方块并交给模型
    - 时间报告：每个生命周期的每步累计总耗时 + 分布直方图 + 汇总统计
    - ESC 单击：返回菜单顶层/暂停继续；ESC 长按：退出

"""

import csv
import ctypes
import math
import os
import sys
import threading
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
    click_repeat_interval: float = 0.25     # 两次单击之间的最小间隔(秒)

    # ===== 抓屏 =====
    capture_size: int = 640                # 只截取屏幕中心的方块边长(px)
    screen_origin: tuple = (0, 0)          # 抓屏区域左上角(自动计算)，用于坐标换算

    # ===== YOLO 并行 =====
    pipeline: bool = True                  # 真=后台线程跑 YOLO 与瞄准并行；假=串行等待推理

    # ===== 时间报告 =====
    report_dir: str = "aim_reports"        # 时间报告存放文件夹
    timing_precision: int = 4              # 时间数值保留的小数位数

    # ===== 模型 =====
    model_path: str = r"D:\PycharmProjects\aim\18_model_aug\weights\best.engine"

    # ===== 目标最低可信度（可在菜单里实时调节）=====
    conf_threshold: float = 0.5

    # ===== 类别（0=person_t, 1=head_t, 2=person_c, 3=head_c）=====
    # 默认只启用 class 1 (head_t)
    enabled_classes: tuple = (1,)

    # ===== 目标跟踪 =====
    max_track_time: float = 1.0   # 单个目标最多追踪时长(秒)，超时改选可信度最高的目标
    max_distance: float = 50.0    # 目标附近搜索范围(px)
    lost_reset_frames: int = 2    # 丢失多少帧后重新选择

    # ===== PID =====
    pid_kp: float = 0.6
    pid_ki: float = 0.3
    pid_kd: float = 0.1
    pid_y_scale: float = 0.7      # y 轴 PID 输出倍数 (= 1600/2560)

    # ===== 前馈（目标速度）=====
    feedforward_x: float = 1.0
    feedforward_y: float = 0.5
    velocity_smooth: float = 0.7

    # ===== 键盘 =====
    esc_vk: int = 0x4B
    vk_up: int = 0x26
    vk_down: int = 0x28
    vk_left: int = 0x25
    vk_right: int = 0x27
    esc_exit_hold: float = 10
    esc_toggle_hold: float = 0.1


CFG = Config()


# ============================================================
# 类别名称表
# ============================================================
CLASS_NAMES = {
    0: "person_t",
    1: "head_t",
    2: "person_c",
    3: "head_c",
}


def get_class_name(class_id):
    return CLASS_NAMES.get(class_id, f"class_{class_id}")


# ============================================================
# 多类别过滤器（记录每个类别的开/关；默认只开 class1）
# ============================================================
class ClassFilter:
    def __init__(self, enabled=None):
        self.state = {cid: False for cid in (0, 1, 2, 3)}
        if enabled is None:
            enabled = (1,)
        for cid in (0, 1, 2, 3):
            self.state[cid] = (cid in enabled)

    def ids(self):
        return sorted(self.state.keys())

    def is_enabled(self, class_id):
        return bool(self.state.get(class_id, False))

    def toggle(self, class_id):
        if class_id in self.state:
            self.state[class_id] = not self.state[class_id]
            return self.state[class_id]
        return False

    def enabled_text(self):
        parts = [
            f"{cid}:{get_class_name(cid)}=开" if self.state[cid]
            else f"{cid}:{get_class_name(cid)}=关"
            for cid in self.ids()
        ]
        return "  ".join(parts)


CLASS_FILTER = ClassFilter()


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


# ===== 时间统计：生命周期每步累计 + 分布直方图 + 汇总 =====
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


def _percentile(sorted_vals, p):
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    k = (n - 1) * p
    lo = int(k)
    hi = min(lo + 1, n - 1)
    if hi == lo:
        return sorted_vals[lo]
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo)


def _time_histogram(vals):
    n = len(vals)
    out = []
    for lo, hi, label in HIST_BINS:
        if hi == float("inf"):
            cnt = sum(1 for v in vals if v >= lo)
        else:
            cnt = sum(1 for v in vals if lo <= v < hi)
        out.append((label, cnt, (cnt / n * 100.0) if n else 0.0))
    return out


class Stats:
    def __init__(self):
        self._lock_time = None
        self._n = 0
        self._life_clicks = 0
        self._life_times = {}
        self._report_count = 0
        self._last_times = {}
        self.click_rows = []
        self.life_rows = []

    def _settle_life(self, closing=False):
        if self._lock_time is None or not self._life_times:
            return
        total_ms = (time.perf_counter() - self._lock_time) * 1000.0
        self.life_rows.append([
            len(self.life_rows) + 1,
            round(total_ms, 4),
            self._n,
            self._life_clicks,
            round(self._life_times.get("截屏", 0.0), 4),
            round(self._life_times.get("YOLO", 0.0), 4),
            round(self._life_times.get("目标处理", 0.0), 4),
            round(self._life_times.get("鼠标位置", 0.0), 4),
            round(self._life_times.get("PID", 0.0), 4),
            round(self._life_times.get("对准+点击", 0.0), 4),
            round(self._life_times.get("移动", 0.0), 4),
        ])
        tag = "关闭前结算" if closing else f"生命周期结束 #{len(self.life_rows)}"
        print()
        print("─" * 58)
        print(f"  {tag}: 总时长 {total_ms:.1f} ms  |  {self._n} 帧  |  命中 {self._life_clicks} 次")
        print("─" * 58)
        for step in STEP_NAMES:
            v = self._life_times.get(step, 0.0)
            pct = (v / total_ms * 100.0) if total_ms > 0 else 0.0
            print(f"  {step:<8}: 总共 {v:10.2f} ms   ({pct:5.1f}%)")
        print("─" * 58)

    def mark_lock(self):
        self._settle_life()
        self._lock_time = time.perf_counter()
        self._last_times = {}
        self._life_times = {}
        self._n = 0
        self._life_clicks = 0

    def add(self, name, seconds):
        self._last_times[name] = seconds * 1000.0
        self._life_times[name] = self._life_times.get(name, 0.0) + seconds * 1000.0

    def track_frame(self):
        self._n += 1

    def mark_click(self):
        if self._lock_time is None or self._n == 0:
            return
        self._report_count += 1
        self._life_clicks += 1
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

    def print_summary(self):
        if not self.click_rows:
            print("本次运行无命中记录")
            return
        n = len(self.click_rows)
        print()
        print("═" * 72)
        print(f"  命中时间汇总（共 {n} 次）—— 主指标为“锁定→按下”端到端耗时")
        print("═" * 72)
        print(f" {'指标':<14}{'平均':>10}{'中位':>10}{'P95':>10}{'P99':>10}{'最大':>10}{'最小':>10}")
        for col, name in DISP_NAMES.items():
            vals = sorted(r[col] for r in self.click_rows)
            mean = sum(vals) / len(vals)
            print(f" {name:<14}{mean:>10.2f}{_percentile(vals,.5):>10.2f}"
                  f"{_percentile(vals,.95):>10.2f}{_percentile(vals,.99):>10.2f}"
                  f"{vals[-1]:>10.2f}{vals[0]:>10.2f}")
        print("═" * 72)
        per = [r[1] for r in self.click_rows]
        print("  每次行动总耗时(ms): " + ", ".join(f"{v:.1f}" for v in per))
        bars = _time_histogram(per)
        width = 30
        max_cnt = max((c for _, c, _ in bars), default=1) or 1
        print()
        print("  — 锁定→按下 总时间分布 —")
        for label, cnt, pct in bars:
            bar = "█" * round(cnt / max_cnt * width)
            print(f"   {label:>7}ms : {cnt:>5} 次 ({pct:5.1f}%)  {bar}")
        print("═" * 72)

    def save_csv(self):
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

            f_life = os.path.join(CFG.report_dir, f"life_accum_{stamp}.csv")
            with open(f_life, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["life_no", "total_ms", "frames", "hits",
                            "grab_ms", "yolo_ms", "select_ms", "mouse_ms",
                            "pid_ms", "align_click_ms", "move_ms"])
                w.writerows(self.life_rows)
            print(f"已保存生命周期累计 -> {f_life} ({len(self.life_rows)} 个生命周期)")

            f_summary = os.path.join(CFG.report_dir, f"summary_{stamp}.csv")
            with open(f_summary, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.writer(f)
                w.writerow(["metric", "mean", "median", "p95", "p99", "max", "min"])
                for col, name in COL_NAMES.items():
                    vals = sorted(r[col] for r in self.click_rows)
                    if vals:
                        w.writerow([name, round(sum(vals)/len(vals), 4),
                                    round(_percentile(vals,.5), 4), round(_percentile(vals,.95), 4),
                                    round(_percentile(vals,.99), 4), vals[-1], vals[0]])
                w.writerow([])
                w.writerow(["lock_to_click_ms_dist_bin", "count", "percent"])
                for label, cnt, pct in _time_histogram([r[1] for r in self.click_rows]):
                    w.writerow([label, cnt, round(pct, 2)])
            print(f"已保存命中汇总 -> {f_summary} ({len(self.click_rows)} 条命中)")
        except Exception as e:
            print("保存时间报告失败:", e)


STATS = Stats()


# ===== 后台 YOLO 推理线程（与瞄准并行）=====
class InferencePipeline:
    def __init__(self, model):
        self.model = model
        self._lock = threading.Lock()
        self._latest = None
        self._pending = None
        self._have = threading.Event()
        self._yolo_ms = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, frame):
        with self._lock:
            self._pending = frame

    def get_result(self):
        if not self._have.is_set():
            return None
        with self._lock:
            return self._latest

    def poll_yolo_elapsed_ms(self):
        with self._lock:
            return self._yolo_ms

    def _run(self):
        while not self._stop.is_set():
            with self._lock:
                frame = self._pending
                self._pending = None
            if frame is None:
                time.sleep(0.0005)
                continue
            t = time.perf_counter()
            result = self.model(frame, verbose=False)[0]
            elapsed_ms = (time.perf_counter() - t) * 1000.0
            with self._lock:
                self._latest = result
                self._yolo_ms = elapsed_ms
            self._have.set()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)


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


# ===== 控制台配置菜单 =====
class ControlMenu:
    PANELS = ["pid", "cls", "conf", "exit"]
    PANEL_NAMES = {
        "pid": "PID",
        "cls": "类别",
        "conf": "置信度",
        "exit": "退出",
    }

    def __init__(self, pid_x, pid_y, class_filter, conf=0.25):
        self.pid_x = pid_x
        self.pid_y = pid_y
        self.cls = class_filter
        self.conf = conf
        self.conf_step = 0.05
        self.request_exit = False
        self._keys = {
            "up": Key(CFG.vk_up),
            "down": Key(CFG.vk_down),
            "left": Key(CFG.vk_left),
            "right": Key(CFG.vk_right),
        }
        self.layer = "top"
        self.panel = "pid"
        self.pid_item = "kp"
        self.pid_step = 0.01
        self.cls_cursor = 0
        self.conf_item = "conf"

    def poll(self):
        up = self._keys["up"].just_pressed()
        down = self._keys["down"].just_pressed()
        left = self._keys["left"].just_pressed()
        right = self._keys["right"].just_pressed()

        if self.layer == "top":
            self._top(left, right, down)
        elif self.layer == "pid":
            self._pid_panel(left, right, up, down)
        elif self.layer == "cls":
            self._cls_panel(left, right, up, down)
        elif self.layer == "conf":
            self._conf_panel(left, right, up, down)
        elif self.layer == "exit":
            self._exit_panel(left, right, up, down)

    # ---------- 顶层 ----------
    def _top(self, left, right, down):
        idx = self.PANELS.index(self.panel)
        if left:
            idx = (idx - 1) % len(self.PANELS)
            self.panel = self.PANELS[idx]
            print(">> 选择: " + self.PANEL_NAMES[self.panel])
            self._print_top()
        elif right:
            idx = (idx + 1) % len(self.PANELS)
            self.panel = self.PANELS[idx]
            print(">> 选择: " + self.PANEL_NAMES[self.panel])
            self._print_top()
        elif down:
            self.layer = self.panel
            self._enter_panel()

    def _print_top(self):
        parts = []
        for p in self.PANELS:
            mark = "▶" if p == self.panel else " "
            parts.append(f"{mark}[{self.PANEL_NAMES[p]}]")
        print("    " + "  ".join(parts) + "   (←→选择, ↓进入)")

    def _enter_panel(self):
        if self.layer == "pid":
            self.pid_item = "kp"
            print(">> 进入 PID 面板 (←→ 切 kp/ki/kd/返回, ↑↓ 调大小, 返回处=回顶层, ESC 返回)")
            self._print_pid_items()
        elif self.layer == "cls":
            self.cls_cursor = 0
            print(">> 进入 类表面板 (←→ 移动 类别/返回, ↑↓ 开/关, 返回处=回顶层, ESC 返回)")
            self.print_classes()
        elif self.layer == "conf":
            self.conf_item = "conf"
            self.conf_step = 0.05
            print(f">> 进入 置信度面板 当前 conf={self.conf:.2f} (←→ 切 置信度/返回, ↑↓ 调, 返回处=回顶层, ESC 返回)")
            self._print_conf_items()
        elif self.layer == "exit":
            print(">> 退出程序？ 按 ↓ 确认退出, 按 ←/→ 返回顶层")

    def _go_back_top(self):
        self.layer = "top"
        print(">> 返回顶层")
        self._print_top()

    # ---------- PID 面板 ----------
    def _pid_panel(self, left, right, up, down):
        order = ["kp", "ki", "kd", "back"]
        if left:
            idx = order.index(self.pid_item)
            self.pid_item = order[(idx - 1) % len(order)]
            self._print_pid_items()
        elif right:
            idx = order.index(self.pid_item)
            self.pid_item = order[(idx + 1) % len(order)]
            self._print_pid_items()
        elif up or down:
            if self.pid_item == "back":
                self._go_back_top()
            else:
                delta = self.pid_step if up else -self.pid_step
                self._pid_adjust(delta)

    def _print_pid_items(self):
        parts = []
        for item in ["kp", "ki", "kd", "back"]:
            mark = "▶" if item == self.pid_item else " "
            if item == "back":
                parts.append(f"{mark}[返回]")
            else:
                parts.append(f"{mark}[{item}={self._pid_value(item):.3f}]")
        print("    " + "  ".join(parts) + "   (←→移动, ↑↓调大小)")

    def _pid_value(self, item):
        return {
            "kp": self.pid_x.kp,
            "ki": self.pid_x.ki,
            "kd": self.pid_x.kd,
        }[item]

    def _pid_adjust(self, delta):
        for pid in (self.pid_x, self.pid_y):
            if self.pid_item == "kp":
                pid.kp = max(0.0, pid.kp + delta)
            elif self.pid_item == "ki":
                pid.ki = max(0.0, pid.ki + delta)
            elif self.pid_item == "kd":
                pid.kd = max(0.0, pid.kd + delta)
        print(f">> {self.pid_item} = {self._pid_value(self.pid_item):.3f}")
        self._print_pid_items()

    # ---------- 类表面板 ----------
    def _cls_panel(self, left, right, up, down):
        ids = self.cls.ids()
        n = len(ids)
        total = n + 1
        if left:
            self.cls_cursor = (self.cls_cursor - 1) % total
            self._print_cls_cursor()
        elif right:
            self.cls_cursor = (self.cls_cursor + 1) % total
            self._print_cls_cursor()
        elif up or down:
            if self.cls_cursor == n:
                self._go_back_top()
            else:
                cid = ids[self.cls_cursor]
                self.cls.toggle(cid)
                print(f">> 类别 {cid}:{get_class_name(cid)} -> {'开' if self.cls.is_enabled(cid) else '关'}")
                self.print_classes()

    def _print_cls_cursor(self):
        self.print_classes()

    def print_classes(self):
        ids = self.cls.ids()
        parts = []
        for i, cid in enumerate(ids):
            mark = "▶" if i == self.cls_cursor else " "
            state = "开" if self.cls.is_enabled(cid) else "关"
            parts.append(f"{mark}[{cid}:{get_class_name(cid)}={state}]")
        mark = "▶" if self.cls_cursor == len(ids) else " "
        parts.append(f"{mark}[返回]")
        print("    " + "  ".join(parts) + "   (←→移动, ↑↓开/关)")

    # ---------- 置信度面板 ----------
    def _conf_panel(self, left, right, up, down):
        if left or right:
            self.conf_item = "back" if self.conf_item == "conf" else "conf"
            self._print_conf_items()
        elif up or down:
            if self.conf_item == "back":
                self._go_back_top()
            else:
                if up:
                    self.conf = min(1.0, self.conf + self.conf_step)
                else:
                    self.conf = max(0.0, self.conf - self.conf_step)
                print(f">> 最低可信度 = {self.conf:.2f}")

    def _print_conf_items(self):
        parts = []
        for item in ["conf", "back"]:
            mark = "▶" if item == self.conf_item else " "
            if item == "conf":
                parts.append(f"{mark}[最低可信度={self.conf:.2f}]")
            else:
                parts.append(f"{mark}[返回]")
        print("    " + "  ".join(parts) + "   (←→移动, ↑↓调阈值)")

    # ---------- 退出面板（顶层真正退出程序）----------
    def _exit_panel(self, left, right, up, down):
        if down or up:
            self.request_exit = True
            print(">> 已请求退出程序...")
        elif left or right:
            self.layer = "top"
            print(">> 取消退出，返回顶层")
            self._print_top()


MENU = None

# ===== YOLO 加载 =====
def load_model():
    print(f"加载模型: {CFG.model_path}")
    model = YOLO(CFG.model_path)
    print("模型类别: 0=person_t, 1=head_t, 2=person_c, 3=head_c")
    return model


# ===== Box 信息（坐标已 +区域偏移，统一为屏幕绝对坐标）=====
def get_box_info(box):
    x1, y1, x2, y2 = box.xyxy[0]
    x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
    ox, oy = CFG.screen_origin
    cx = (x1 + x2) / 2 + ox
    cy = (y1 + y2) / 2 + oy
    w = x2 - x1
    h = y2 - y1
    conf = box.conf.item()
    cls = int(box.cls.item())
    return cx, cy, w, h, conf, cls


# ===== 一次遍历：类别过滤 + 置信度过滤 + 预提取中心 =====
def get_candidates(result):
    cands = []
    ox, oy = CFG.screen_origin
    min_conf = MENU.conf if MENU is not None else CFG.conf_threshold
    for box in result.boxes:
        cls = int(box.cls.item())
        if not CLASS_FILTER.is_enabled(cls):
            continue
        conf = box.conf.item()
        if conf < min_conf:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = (x1 + x2) * 0.5 + ox
        cy = (y1 + y2) * 0.5 + oy
        cands.append((box, cx, cy, conf))
    return cands


# ===== 目标选择（预提取值 + 平方距离）=====
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
        self._lock_start = None

    def select(self, cands):
        if not self.tracking:
            return self._lock_from_mouse(cands)
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
    sm_w = user32.GetSystemMetrics(0)
    sm_h = user32.GetSystemMetrics(1)
    cs = min(CFG.capture_size, sm_w, sm_h)
    left = (sm_w - cs) // 2
    top = (sm_h - cs) // 2
    region = (left, top, left + cs, top + cs)
    CFG.screen_origin = (left, top)

    camera = bettercam.create(output_color="BGR", region=region)
    model = load_model()
    pipe = InferencePipeline(model) if CFG.pipeline else None

    esc = EscController(CFG.esc_exit_hold, CFG.esc_toggle_hold)
    tracker = TargetTracker()
    clicker = Clicker(CFG.click_frames, CFG.click_repeat_interval)
    pid_x = PID(CFG.pid_kp, CFG.pid_ki, CFG.pid_kd)
    pid_y = PID(CFG.pid_kp, CFG.pid_ki, CFG.pid_kd)
    global MENU
    MENU = ControlMenu(pid_x, pid_y, CLASS_FILTER, conf=CFG.conf_threshold)
    frame_count = 0

    print()
    print("=" * 70)
    print("Auto-Aim 启动")
    print("YOLO + PID + 前馈 + 多类别 + 置信度 + 退出")
    print("-" * 70)
    print("模型:", CFG.model_path)
    print("类别开关:" + "  " + CLASS_FILTER.enabled_text())
    print("最低可信度:", MENU.conf)
    print("单目标最长追踪:", CFG.max_track_time, "秒")
    print("抓屏区域(屏幕中心):", region, f"({cs}x{cs})")
    print("YOLO 并行流水线:", "开" if CFG.pipeline else "关(串行)")
    print("y 轴 PID 倍数:", CFG.pid_y_scale)
    print("时间报告文件夹:", CFG.report_dir)
    print("-" * 70)
    print("【配置菜单】")
    print("  顶层:  ←→ 选择 PID/类别/置信度/退出,   ↓ 进入")
    print("  每个面板内:  ←→ 移动选项(含末尾[返回])")
    print("              ↑↓ 调节/开关, 停在[返回]按 ↑↓ 回到顶层")
    print("  顶层 [退出]: ↓ 确认退出程序")
    print("  ESC:   顶层单击=暂停/继续,  长按=退出")
    print("-" * 70)
    MENU._print_pid_items()
    MENU._print_top()
    print("=" * 70)

    try:
        while True:
            frame_count += 1

            if MENU.request_exit:
                print("退出请求，程序结束")
                break

            if esc.poll():
                break

            if esc.paused:
                MENU.poll()
                time.sleep(0.01)
                continue

            MENU.poll()

            # ======= 瞄准 =======
            t = time.perf_counter()
            frame = camera.grab()
            if frame is None:
                continue
            STATS.add("截屏", time.perf_counter() - t)

            if CFG.pipeline:
                pipe.submit(frame)
                STATS.add("YOLO", pipe.poll_yolo_elapsed_ms() / 1000.0)
                result = pipe.get_result()
                if result is None:
                    continue
            else:
                t = time.perf_counter()
                result = model(frame, verbose=False)[0]
                STATS.add("YOLO", time.perf_counter() - t)

            cands = get_candidates(result)

            if len(cands) == 0:
                # 无目标：重置跟踪与 PID
                tracker.reset()
                pid_x.reset()
                pid_y.reset()
                continue

            t = time.perf_counter()
            if not tracker.select(cands):
                continue
            STATS.add("目标处理", time.perf_counter() - t)

            t = time.perf_counter()
            mouse_x, mouse_y = get_mouse_pos()
            error_x = tracker.locked_x - mouse_x
            error_y = tracker.locked_y - mouse_y
            STATS.add("鼠标位置", time.perf_counter() - t)

            dist = math.hypot(error_x, error_y)

            vel = math.hypot(tracker.velocity_x, tracker.velocity_y)
            STATS.track_frame()

            t = time.perf_counter()
            move_x = pid_x.update(error_x) + tracker.velocity_x * CFG.feedforward_x
            move_y = pid_y.update(error_y) * CFG.pid_y_scale + tracker.velocity_y * CFG.feedforward_y
            STATS.add("PID", time.perf_counter() - t)

            t = time.perf_counter()
            aligned = abs(error_x) < CFG.align_threshold and abs(error_y) < CFG.align_threshold
            clicker.update(aligned)
            STATS.add("对准+点击", time.perf_counter() - t)

            t = time.perf_counter()
            send_move(move_x, move_y)
            STATS.add("移动", time.perf_counter() - t)

    except KeyboardInterrupt:
        print("\n用户中断")
    finally:
        cv2.destroyAllWindows()
        camera.release()
        if pipe is not None:
            pipe.stop()
        STATS._settle_life(closing=True)
        STATS.save_csv()
        STATS.print_summary()
        print()
        print("═" * 58)
        print("  本次设置的 PID / 类别 / 置信度：")
        print(f"    pid_kp        = {pid_x.kp:.3f}")
        print(f"    pid_ki        = {pid_x.ki:.3f}")
        print(f"    pid_kd        = {pid_x.kd:.3f}")
        print(f"    feedforward_x = {CFG.feedforward_x}")
        print(f"    feedforward_y = {CFG.feedforward_y}")
        print(f"    pid_y_scale   = {CFG.pid_y_scale}")
        print(f"    最低可信度    = {MENU.conf:.2f}")
        print("    类别开关: " + CLASS_FILTER.enabled_text())
        print("  请把以上值写回脚本开头的 Config 后再运行。")
        print("═" * 58)
        print("程序结束")


if __name__ == "__main__":
    main()