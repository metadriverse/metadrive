import time


class ForceFPS:
    UNLIMITED = 0
    # FPS60 = 1
    FORCED = 2

    def __init__(self, fps, start=False):
        self.last = time.time()
        self.init_fps = fps
        if start:
            self.state = self.FORCED
            self.fps = fps
        else:
            self.state = self.UNLIMITED
            self.fps = None

    @property
    def interval(self):
        return 1 / self.fps if self.fps is not None else None

    def __enter__(self):
        # print("Force fps, now: ", self.last)
        now = time.time()
        if self.interval and now - self.last < self.interval:
            time.sleep(self.interval - (now - self.last))

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.last = time.time()

    def toggle(self):
        if self.state == self.UNLIMITED:
            self.state = self.FORCED
            self.fps = self.init_fps

        elif self.state == self.FORCED:
            self.state = self.UNLIMITED
            self.fps = None
