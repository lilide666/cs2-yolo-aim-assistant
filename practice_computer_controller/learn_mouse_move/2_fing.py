"""
视觉伺服 - 45° 对角线径向扫描（重复平均版）

沿 du=dv 的对角线，在多个半径 r 处，每档重复 REPEAT 次从中心出发测位移，取平均。
目的：压噪声，得到干净可信的 K_x(r)、K_y(r)，判断真实径向畸变。

配置：45°对角线, 每档5次, r从80到360步长40
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

CONVERGE_EPS = 3.0
MAX_ITER = 250
AVG_FRAMES = 4

MAX_STEP = 120          # 单步限幅
EDGE_GUARD = 30

REPEAT = 5              # 每档重复次数
R_STEPS = [80, 120, 160, 200, 240, 280, 320, 360]   # 45°对角线半径档位

SAVE_PATH = "aim_radial_scan.json"

# ============================================================
# 鼠标 / PID / 检测（同前）
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
        fx, fy = clamp_move(fx, fy)
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

    print("=" * 70)
    print("45° 对角线径向扫描（每档重复 %d 次取平均）" % REPEAT)
    print("=" * 70)
    print(f"半径档位: {R_STEPS}")

    # 把目标拖到中心（起点）
    print("把目标拖到画面中心...")
    drag_to(model, camera, center_x, center_y)

    results = []

    for r in R_STEPS:
        # 45°对角线: du = dv = r*sin(45) = r/sqrt(2)
        d = r / (2 ** 0.5)
        tx = center_x + d
        ty = center_y + d   # 对角线第一象限方向

        # 收集该半径档 REPEAT 次测量
        dx_sum = 0.0
        dy_sum = 0.0
        ok_count = 0

        for rep in range(REPEAT):
            drag_to(model, camera, center_x, center_y)   # 回中心
            dx, dy = drag_to(model, camera, tx, ty)      # 从中心拖到该点
            dx_sum += dx
            dy_sum += dy
            ok_count += 1

        avg_dx = dx_sum / ok_count if ok_count else 0.0
        avg_dy = dy_sum / ok_count if ok_count else 0.0

        # 系数（45°对角线上一个点同时给 X 和 Y）
        kx = abs(avg_dx / d) if d != 0 else 0.0
        ky = abs(avg_dy / d) if d != 0 else 0.0

        results.append({
            "r": r,
            "d": d,
            "dx_avg": avg_dx,
            "dy_avg": avg_dy,
            "kx": kx,
            "ky": ky,
        })

        print(f"  r={r:3d} (d={d:6.1f}px) "
              f"dx_avg={avg_dx:7.1f} dy_avg={avg_dy:7.1f} "
              f"Kx={kx:.4f} Ky={ky:.4f}")

        # 回中心，准备下一档
        drag_to(model, camera, center_x, center_y)

    print()
    print("=" * 70)
    print("径向扫描结果")
    print("=" * 70)
    rs = [res["r"] for res in results]
    kxs = [res["kx"] for res in results]
    kys = [res["ky"] for res in results]

    print("  r : " + "  ".join(f"{v:5d}" for v in rs))
    print("  Kx: " + "  ".join(f"{v:5.3f}" for v in kxs))
    print("  Ky: " + "  ".join(f"{v:5.3f}" for v in kys))

    # 线性拟合 Kx(r), Ky(r)
    def fit(rs_, ks_):
        A = np.vstack([rs_, np.ones_like(rs_)]).T
        coef, *_ = np.linalg.lstsq(A, ks_, rcond=None)
        b, a = coef
        pred = A @ coef
        ss_res = np.sum((ks_ - pred) ** 2)
        ss_tot = np.sum((ks_ - np.mean(ks_)) ** 2)
        r2 = 1 - ss_res / ss_tot
        return a, b, r2

    a_x, b_x, r2_x = fit(rs, kxs)
    a_y, b_y, r2_y = fit(rs, kys)

    print()
    print("线性拟合 K = a + b·r")
    print(f"  X: Kx(r) = {a_x:.4f} {b_x:+.6f}·r   R²={r2_x:.4f}")
    print(f"    中心Kx={a_x+b_x*80:.3f} 边缘Kx={a_x+b_x*360:.3f} "
          f"变化{(b_x*280)/(a_x+b_x*80)*100:+.1f}%")
    print(f"  Y: Ky(r) = {a_y:.4f} {b_y:+.6f}·r   R²={r2_y:.4f}")
    print(f"    中心Ky={a_y+b_y*80:.3f} 边缘Ky={a_y+b_y*360:.3f} "
          f"变化{(b_y*280)/(a_y+b_y*80)*100:+.1f}%")

    # 判断
    print()
    if abs(b_x * 280 / (a_x + b_x * 80)) < 0.03 and abs(b_y * 280 / (a_y + b_y * 80)) < 0.03:
        print("✓ 径向变化 < 3%，且重复平均后 R² 低 → 基本是常数，可用 K≈常数")
    else:
        print("✗ 径向变化显著 → 用 K(r)=a+b·r 径向自适应")

    with open(SAVE_PATH, "w", encoding="utf-8") as f:
        json.dump({"r_steps": R_STEPS, "repeat": REPEAT, "results": results}, f,
                  indent=2, ensure_ascii=False)
    print()
    print(f"已保存: {SAVE_PATH}")

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()