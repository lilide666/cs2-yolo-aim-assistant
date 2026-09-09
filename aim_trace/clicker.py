import time
from .config import CFG
from .stats import STATS
from .windows_input import mouse_left_down, mouse_left_up


def mouse_left_click() -> None:
    mouse_left_down()
    time.sleep(CFG.click_down_interval)
    mouse_left_up()


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