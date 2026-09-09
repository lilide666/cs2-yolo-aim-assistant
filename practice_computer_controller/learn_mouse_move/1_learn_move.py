"""
视觉伺服 - 8x8 径向标定（限幅±120 + 边缘保护）

改动 vs 上一版：
  - 单步限幅 clamp_move(±120)：保留对目标出画的保护，但比之前±80更大，更接近完整PID位移
  - 保留边缘保护 EDGE_GUARD=30
  - 精确版：每点先从中心出发，再拖到网格点（纯径向数据）
  - 使用目标验证可用的 PID：kp=0.6 ki=0.3 kd=0.1 y_scale=0.7
"""
import ctypes
import time
import json

import numpy as np
import bettercam
import cv2
from ultralytics import YOLO

# ============================================================
# 配置
# ============================================================
MODEL_PATH = r"D:\PycharmProjects\computer_controller\runs\18_model_aug\weights\last.engine"
CAPTURE_SIZE = 640
CONF = 0.35
TARGET_CLASS = 1

PID_KP = 0.6
PID_KI = 0.3
PID_KD = 0.1
PID_Y_SCALE = 0.7

GRID_N = 8
MARGIN = 60
CONVERGE_EPS = 3.0
MAX_ITER = 300
AVG_FRAMES = 5

MAX_STEP = 120          # 单步限幅：放大到 ±120
EDGE_GUARD = 30         # 边缘保护

SAVE_PATH = "aim_grid_8x8.json"

# ============================================================
# 鼠标 / PID / 检测
# ============================================================
user32 = ctypes.windll.user32
MOUSEEVENTF_MOVE = 0x0001

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


def send_move(dx, dy):
    extra = ctypes.c_ulong(0)
    mi = MOUSEINPUT(int(dx), int(dy), 0, MOUSEEVENTF_MOVE, 0, ctypes.pointer(extra))
    inp = INPUT(0, mi)
    user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))


def clamp_move(fx, fy, limit=MAX_STEP):
    """限制单步位移幅值，防一帧冲过头。"""
    mag = (fx * fx + fy * fy) ** 0.5
    if mag > limit:
        scale = limit / mag
        fx = int(fx * scale)
        fy = int(fy * scale)
    return fx, fy


class PID:
    def __init__(self, kp, ki, kd, imax=500.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.imax = imax
        self.reset()

    def reset(self):
        self.last_error = 0.0
        self.integral = 0.0

    def update(self, error):
        self.integral += error
        if abs(error) < 100:
            self.integral *= 0.1
        self.integral = max(-self.imax, min(self.imax, self.integral))
        derivative = error - self.last_error
        out = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.last_error = error
        return out


def detect_target(model, frame):
    result = model(frame, verbose=False)[0]
    best = None
    best_conf = 0.0
    for box in result.boxes:
        if int(box.cls.item()) != TARGET_CLASS:
            continue
        conf = box.conf.item()
        if conf < CONF:
            continue
        if conf > best_conf:
            best_conf = conf
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            best = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
    return best


def detect_target_avg(model, camera, n=AVG_FRAMES):
    xs, ys = [], []
    for _ in range(n):
        frame = camera.grab()
        if frame is None:
            continue
        p = detect_target(model, frame)
        if p is not None:
            xs.append(p[0])
            ys.append(p[1])
    if not xs:
        return None
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def drag_to(model, camera, tx, ty):
    """把目标拖到 (tx,ty)，返回消耗位移。限幅±120 + 边缘保护。"""
    pid_x = PID(PID_KP, PID_KI, PID_KD)
    pid_y = PID(PID_KP, PID_KI, PID_KD)
    sum_fx = 0.0
    sum_fy = 0.0
    stable = 0
    it = 0
    while it < MAX_ITER:
        it += 1
        p = detect_target_avg(model, camera)
        if p is None:
            continue
        cx, cy = p
        err_x = cx - tx
        err_y = cy - ty

        # 边缘保护
        if cx < EDGE_GUARD and err_x < 0:
            err_x = 0
        if cx > CAPTURE_SIZE - EDGE_GUARD and err_x > 0:
            err_x = 0
        if cy < EDGE_GUARD and err_y < 0:
            err_y = 0
        if cy > CAPTURE_SIZE - EDGE_GUARD and err_y > 0:
            err_y = 0

        if abs(err_x) < CONVERGE_EPS and abs(err_y) < CONVERGE_EPS:
            stable += 1
            if stable >= 3:
                break
        else:
            stable = 0

        fx = int(pid_x.update(err_x))
        fy = int(pid_y.update(err_y) * PID_Y_SCALE)
        fx, fy = clamp_move(fx, fy)     # 单步限幅 ±120
        send_move(fx, fy)
        sum_fx += fx
        sum_fy += fy
        time.sleep(0.01)
    return sum_fx, sum_fy


# ============================================================
# 主流程
# ============================================================
def main():
    model = YOLO(MODEL_PATH)

    sm_w = user32.GetSystemMetrics(0)
    sm_h = user32.GetSystemMetrics(1)
    cs = min(CAPTURE_SIZE, sm_w, sm_h)
    left = (sm_w - cs) // 2
    top = (sm_h - cs) // 2
    camera = bettercam.create(output_color="BGR",
                              region=(left, top, left + cs, top + cs))

    center_x = CAPTURE_SIZE / 2.0
    center_y = CAPTURE_SIZE / 2.0

    xs = np.linspace(MARGIN, CAPTURE_SIZE - MARGIN, GRID_N)
    ys = np.linspace(MARGIN, CAPTURE_SIZE - MARGIN, GRID_N)

    print("=" * 70)
    print("8x8 径向标定（限幅±120 + 边缘保护）")
    print("=" * 70)
    print(f"网格 {GRID_N}x{GRID_N}={GRID_N*GRID_N} 点, 边距 {MARGIN}px")
    print(f"PID: kp={PID_KP} ki={PID_KI} kd={PID_KD} y_scale={PID_Y_SCALE}")
    print(f"单步限幅: ±{MAX_STEP} | 边缘保护: {EDGE_GUARD}px")

    print("把目标拖到画面中心...")
    drag_to(model, camera, center_x, center_y)

    samples = []
    t0 = time.time()
    idx = 0

    for gy in ys:
        for gx in xs:
            idx += 1
            du = gx - center_x
            dv = gy - center_y
            r = (du * du + dv * dv) ** 0.5

            drag_to(model, camera, center_x, center_y)   # 回中心
            dx, dy = drag_to(model, camera, gx, gy)       # 拖到该点

            samples.append({
                "gx": gx, "gy": gy,
                "du": du, "dv": dv, "r": r,
                "dx": dx, "dy": dy,
            })

            if idx % 16 == 0 or idx == GRID_N * GRID_N:
                el = time.time() - t0
                print(f"  进度 {idx}/{GRID_N*GRID_N}  耗时 {el:.1f}s")

    print()
    print("=" * 70)
    print("标定数据 (中心->每点)")
    print("=" * 70)
    for s in samples:
        print(f"  偏移({s['du']:+.0f},{s['dv']:+.0f}) r={s['r']:.0f} "
              f"-> 位移({s['dx']:+.0f},{s['dy']:+.0f})")

    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "grid_n": GRID_N,
            "margin": MARGIN,
            "capture": CAPTURE_SIZE,
            "samples": samples,
        }, f, indent=2, ensure_ascii=False)
    print(f"已保存: {SAVE_PATH}")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()