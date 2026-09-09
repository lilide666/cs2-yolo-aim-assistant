from .config import CFG
from .keys import Key
from .consts import get_class_name


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

    # ---------- 退出面板 ----------
    def _exit_panel(self, left, right, up, down):
        if down or up:
            self.request_exit = True
            print(">> 已请求退出程序...")
        elif left or right:
            self.layer = "top"
            print(">> 取消退出，返回顶层")
            self._print_top()