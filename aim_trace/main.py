import ctypes
import math
import time

import bettercam
import cv2

from .config import CFG
from .class_filter import CLASS_FILTER
from .windows_input import get_mouse_pos, send_move
from .stats import STATS
from .inference import InferencePipeline, load_model
from .keys import EscController
from .pid import PID
from .menu import ControlMenu
from .tracker import TargetTracker, get_candidates
from .clicker import Clicker

user32 = ctypes.windll.user32


def main():
    sm_w = user32.GetSystemMetrics(0)
    sm_h = user32.GetSystemMetrics(1)
    cs = min(CFG.capture_size, sm_w, sm_h)
    left = (sm_w - cs) // 2
    top = (sm_h - cs) // 2
    region = (left, top, left + cs, top + cs)
    CFG.screen_origin = (left, top)

    camera = bettercam.create(output_color="BGR", region=region)
    model = load_model(CFG.model_path)
    pipe = InferencePipeline(model) if CFG.pipeline else None

    esc = EscController(CFG.esc_exit_hold, CFG.esc_toggle_hold)
    tracker = TargetTracker()
    clicker = Clicker(CFG.click_frames, CFG.click_repeat_interval)
    pid_x = PID(CFG.pid_kp, CFG.pid_ki, CFG.pid_kd)
    pid_y = PID(CFG.pid_kp, CFG.pid_ki, CFG.pid_kd)
    menu = ControlMenu(pid_x, pid_y, CLASS_FILTER, conf=CFG.conf_threshold)

    print()
    print("=" * 70)
    print("Auto-Aim 启动")
    print("YOLO + PID + 前馈 + 轨迹预测 + 多类别 + 置信度 + 退出")
    print("-" * 70)
    print("模型:", CFG.model_path)
    print("类别开关:" + "  " + CLASS_FILTER.enabled_text())
    print("最低可信度:", menu.conf)
    print("单目标最长追踪:", CFG.max_track_time, "秒")
    print("抓屏区域(屏幕中心):", region, f"({cs}x{cs})")
    print("YOLO 并行流水线:", "开" if CFG.pipeline else "关(串行)")
    print("轨迹预测:", "开" if CFG.predict_enabled else "关",
          f"(提前量 {CFG.predict_lead_time}s)")
    print("y 轴 PID 倍数:", CFG.pid_y_scale)
    print("-" * 70)
    print("【配置菜单】")
    print("  顶层:  ←→ 选择 PID/类别/置信度/退出,   ↓ 进入")
    print("  每个面板内:  ←→ 移动选项(含末尾[返回])")
    print("              ↑↓ 调节/开关, 停在[返回]按 ↑↓ 回到顶层")
    print("  顶层 [退出]: ↓ 确认退出程序")
    print("  ESC:   顶层单击=暂停/继续,  长按=退出")
    print("-" * 70)
    menu._print_pid_items()
    menu._print_top()
    print("=" * 70)

    try:
        while True:
            if menu.request_exit:
                print("退出请求，程序结束")
                break

            if esc.poll():
                break

            if esc.paused:
                menu.poll()
                time.sleep(0.01)
                continue

            menu.poll()

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

            cands = get_candidates(result, CLASS_FILTER, menu.conf)

            if len(cands) == 0:
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
            # 用轨迹预测的目标位置计算误差，补偿移动延迟
            target_x, target_y = tracker.predicted_position()
            error_x = target_x - mouse_x
            error_y = target_y - mouse_y
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
        try:
            camera.release()
        except Exception:
            pass
        if pipe is not None:
            try:
                pipe.stop()
            except Exception:
                pass
        STATS.settle()
        print()
        print("═" * 58)
        print("  本次设置的 PID / 类别 / 置信度：")
        print(f"    pid_kp        = {pid_x.kp:.3f}")
        print(f"    pid_ki        = {pid_x.ki:.3f}")
        print(f"    pid_kd        = {pid_x.kd:.3f}")
        print(f"    feedforward_x = {CFG.feedforward_x}")
        print(f"    feedforward_y = {CFG.feedforward_y}")
        print(f"    predict_enabled = {CFG.predict_enabled}")
        print(f"    predict_lead_time = {CFG.predict_lead_time}s")
        print(f"    pid_y_scale   = {CFG.pid_y_scale}")
        print(f"    最低可信度    = {menu.conf:.2f}")
        print("    类别开关: " + CLASS_FILTER.enabled_text())
        print("═" * 58)
        print("程序结束")


if __name__ == "__main__":
    main()