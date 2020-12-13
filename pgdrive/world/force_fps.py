import time


class ForceFPS:
    def __init__(self, fps):
        self.last = time.time()
        self.interval = 1 / fps if fps is not None else None

    def __enter__(self):
        # print("Force fps, now: ", self.last)
        now = time.time()
        if self.interval and now - self.last < self.interval:
            time.sleep(self.interval - (now - self.last))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.last = time.time()