import time
from .config import CFG


class Stats:
    def __init__(self):
        # 当前生命周期累计
        self._life_times = {}
        self._life_clicks = 0
        self._n = 0
        self._last_times = {}
        self._lock_time = None
        # 全局累计（便于末尾汇总）
        self.total_frames = 0
        self.total_clicks = 0

    def _reset_life(self):
        self._lock_time = time.perf_counter()
        self._last_times = {}
        self._life_times = {}
        self._n = 0
        self._life_clicks = 0

    def mark_lock(self):
        # 新锁定一个目标 -> 重置当前生命周期累计
        self._print_life()
        self._reset_life()

    def add(self, name, seconds):
        ms = seconds * 1000.0
        self._last_times[name] = ms
        self._life_times[name] = self._life_times.get(name, 0.0) + ms

    def track_frame(self):
        self._n += 1
        self.total_frames += 1

    def mark_click(self):
        if self._lock_time is None:
            return
        self._life_clicks += 1
        self.total_clicks += 1

    def _print_life(self):
        """打印上一个生命周期的耗时分布（各步骤合计=100%）。"""
        if self._lock_time is None:
            return
        total_ms = sum(self._life_times.values())
        if total_ms <= 0:
            return
        print()
        print("─" * 58)
        print(f"  生命周期: {self._n} 帧  |  命中 {self._life_clicks} 次  |"
              f" 合计 {total_ms:.1f} ms")
        print("─" * 58)
        for step, ms in self._life_times.items():
            pct = (ms / total_ms * 100.0)
            print(f"  {step:<8}: {ms:10.2f} ms  ({pct:5.1f}%)")
        print("─" * 58)

    def settle(self):
        """程序退出时打印最后一段 + 汇总。"""
        self._print_life()
        print()
        print("═" * 58)
        print(f"  总帧数: {self.total_frames}  |  总命中: {self.total_clicks}")
        print("═" * 58)


STATS = Stats()