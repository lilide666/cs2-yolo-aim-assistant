import threading
import time
from ultralytics import YOLO


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


def load_model(model_path: str):
    print(f"加载模型: {model_path}")
    model = YOLO(model_path)
    print("模型类别: 0=person_t, 1=head_t, 2=person_c, 3=head_c")
    return model