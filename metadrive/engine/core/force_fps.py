import time

from metadrive.constants import RENDER_MODE_ONSCREEN, RENDER_MODE_NONE


class ForceFPS:
    UNLIMITED = "UnlimitedFPS"
    FORCED = "ForceFPS"

    def __init__(self, engine):
        self.engine = engine
        interval = engine.global_config["physics_world_step_size"] \
            if engine.global_config["force_render_fps"] is None else 1 / engine.global_config["force_render_fps"]
        fps = 1 / interval
        self.init_fps = fps
        if engine.mode == RENDER_MODE_ONSCREEN:
            self.state = self.FORCED
            self.engine.taskMgr.add(self.force_fps_task, "force_fps")
            self.fps = fps
        else:
            self.state = self.UNLIMITED
            self.fps = None

    @property
    def interval(self):
        return 1 / self.fps if self.fps is not None else None

    def tick(self):
        # print("Force fps, now: ", self.last)
        sim_interval = self.engine.task_manager.globalClock.getDt()
        if self.interval and sim_interval < self.interval:
            time.sleep(self.interval - sim_interval)

    def toggle(self):
        if self.state == self.UNLIMITED:
            self.engine.task_manager.add(self.force_fps_task, "force_fps")
            self.state = self.FORCED
            self.fps = self.init_fps
        elif self.state == self.FORCED:
            self.engine.task_manager.remove("force_fps")
            self.state = self.UNLIMITED
            self.fps = None

    def force_fps_task(self, task):
        self.tick()
        return task.cont

    @property
    def real_time_simulation(self):
        return self.state == self.FORCED and self.engine.mode == RENDER_MODE_ONSCREEN
