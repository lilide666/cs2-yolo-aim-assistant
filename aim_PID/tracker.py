import math
import time
from .config import CFG
from .stats import STATS
from .windows_input import get_mouse_pos


def get_box_info(box, screen_origin):
    x1, y1, x2, y2 = box.xyxy[0]
    x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
    ox, oy = screen_origin
    cx = (x1 + x2) / 2 + ox
    cy = (y1 + y2) / 2 + oy
    w = x2 - x1
    h = y2 - y1
    conf = box.conf.item()
    cls = int(box.cls.item())
    return cx, cy, w, h, conf, cls


def get_candidates(result, class_filter, min_conf):
    cands = []
    ox, oy = CFG.screen_origin
    for box in result.boxes:
        cls = int(box.cls.item())
        if not class_filter.is_enabled(cls):
            continue
        conf = box.conf.item()
        if conf < min_conf:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = (x1 + x2) * 0.5 + ox
        cy = (y1 + y2) * 0.5 + oy
        cands.append((box, cx, cy, conf))
    return cands


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
         self.locked_conf, self.locked_cls) = get_box_info(box, CFG.screen_origin)

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