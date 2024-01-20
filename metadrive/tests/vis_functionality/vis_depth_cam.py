from metadrive.component.sensors.depth_camera import DepthCamera
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv

if __name__ == "__main__":
    env = SafeMetaDriveEnv(
        {
            "num_scenarios": 1,
            "traffic_density": 0.,
            "accident_prob": 1.,
            "start_seed": 4,
            "map": "SSSSS",
            "manual_control": False,
            # "use_render": True,
            "image_observation": True,
            "norm_pixel": True,
            "debug": True,
            "interface_panel": ["depth_camera"],
            "sensors": dict(depth_camera=(DepthCamera, 800, 600)),
            "vehicle_config": dict(image_source="depth_camera"),
            # "map_config": {
            #     BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
            #     BaseMap.GENERATE_CONFIG: 12,
            #     BaseMap.LANE_WIDTH: 3.5,
            #     BaseMap.LANE_NUM: 3,
            # }
        }
    )
    env.reset()

    def get_image(env):
        depth_cam = env.agent.get_camera(env.agent.config["image_source"])
        rgb_cam = env.agent.get_camera("rgb_camera")
        for h in range(-180, 180, 20):
            env.engine.graphicsEngine.renderFrame()
            depth_cam.get_cam().setH(h)
            rgb_cam.get_cam().setH(h)
            depth_cam.save_image(env.agent, "depth_{}.jpg".format(h))
            rgb_cam.save_image(env.agent, "rgb_{}.jpg".format(h))
        env.engine.screenshot()

    env.engine.accept("m", get_image, extraArgs=[env])
    import cv2

    for i in range(1, 100000):
        o, r, tm, tc, info = env.step([0, 1])
        assert env.observation_space.contains(o)
        cv2.imshow('img', o["image"][..., -1])
        cv2.waitKey(1)
        if env.config["use_render"]:
            # for i in range(ImageObservation.STACK_SIZE):
            #     ObservationType.show_gray_scale_array(o["image"][:, :, i])
            env.render()
        # if tm or tc:
        #     # print("Reset")
        #     env.reset()
    env.close()
