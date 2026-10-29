from metadrive.envs.marl_envs.marl_parking_lot import MultiAgentParkingLotEnv


def vis_ma_parking_lot_env():
    import cv2
    env = MultiAgentParkingLotEnv(
        {
            "use_render": False,
            "image_observation": True,
            # it is a switch telling metadrive to use rgb as observation
            "norm_pixel": True,  # clip rgb to range(0,1) instead of (0, 255)
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
    o, r, tm, tc, i = env.step(env.action_space.sample())
    for i in range(4):
        cv2.imshow('img', o["agent{}".format(i)]["image"][..., -1])
        cv2.waitKey(0)


if __name__ == '__main__':
    vis_ma_parking_lot_env()
