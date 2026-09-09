import ctypes
import time
from .config import CFG

user32 = ctypes.windll.user32


class Key:
    def __init__(self, vk: int):
        self.vk = vk
        self._last = False

    def just_pressed(self) -> bool:
        cur = bool(user32.GetAsyncKeyState(self.vk) & 0x8000)
        edge = cur and not self._last
        self._last = cur
        return edge


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