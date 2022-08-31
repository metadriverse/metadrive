from metadrive.envs.marl_envs.marl_parking_lot import MultiAgentParkingLotEnv
import cv2


def vis_ma_parking_lot_env():
    env = MultiAgentParkingLotEnv(
        {
            "use_render": False,
            "offscreen_render": True,
            # it is a switch telling metadrive to use rgb as observation
            "rgb_clip": True,  # clip rgb to range(0,1) instead of (0, 255)
            "delay_done": 0,
            "num_agents": 4,
            "vehicle_config": {
                "stack_size": 5,
                "rgb_camera": (800, 600),
                "lidar": {
                    "num_others": 8
                }
            }
        }
    )
    env.reset()
    o, r, d, i = env.step(env.action_space.sample())
    for i in range(4):
        cv2.imshow('img', o["agent{}".format(i)]["image"][..., -1])
        cv2.waitKey(0)


if __name__ == '__main__':
    vis_ma_parking_lot_env()
