from .consts import get_class_name


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