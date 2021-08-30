import time

from metadrive.envs.top_down_env import TopDownSingleFrameMetaDriveEnv
from metadrive.utils import setup_logger


def vis_top_down_render_with_panda_render():
    setup_logger(True)

    env = TopDownSingleFrameMetaDriveEnv(
        {
            "environment_num": 1,
            "manual_control": True,
            "use_render": True,
            "offscreen_render": False,
            "traffic_mode": "respawn"
        }
    )
    o = env.reset()
    s = time.time()
    for i in range(1, 100000):
        o, r, d, info = env.step(env.action_space.sample())
        env.render(
            text={
                "vehicle_num": len(env.engine.traffic_manager.vehicles),
                "traffic_vehicle": len(env.engine.traffic_manager.traffic_vehicles)
            }
        )
        # if d:
        #     env.reset()
        if i % 1000 == 0:
            print("Steps: {}, Time: {}".format(i, time.time() - s))
    env.close()


if __name__ == '__main__':
    vis_top_down_render_with_panda_render()
