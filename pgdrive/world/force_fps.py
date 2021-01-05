import time
from pgdrive.world.constants import RENDER_MODE_ONSCREEN


class ForceFPS:
    UNLIMITED = "UnlimitedFPS"
    FORCED = "ForceFPS"

    def __init__(self, pg_world, start=False):
        fps = 1 / pg_world.pg_config["physics_world_step_size"]
        self.pg_world = pg_world
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

    def tick(self):
        # print("Force fps, now: ", self.last)
        sim_interval = self.pg_world.taskMgr.globalClock.getDt()
        if self.interval and sim_interval < self.interval:
            time.sleep(self.interval - sim_interval)

    def toggle(self):
        if self.state == self.UNLIMITED:
            self.pg_world.taskMgr.add(self.force_fps_task, "force_fps")
            self.state = self.FORCED
            self.fps = self.init_fps
        elif self.state == self.FORCED:
            self.pg_world.taskMgr.remove("force_fps")
            self.state = self.UNLIMITED
            self.fps = None

    def force_fps_task(self, task):
        self.tick()
        return task.cont

    @property
    def real_time_simulation(self):
        return self.state == self.FORCED and self.pg_world.mode == RENDER_MODE_ONSCREEN
