from metadrive.component.sensors.semantic_camera import SemanticCamera
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv

if __name__ == "__main__":

    def get_image(env):
        semantic_cam = env.engine.get_sensor(env.vehicle.config["image_source"])
        # for h in range(-180, 180, 20):
        #     env.engine.graphicsEngine.renderFrame()
        #     semantic_cam.get_cam().setH(h)
        #     # rgb_cam.get_cam().setH(h)
        semantic_cam.save_image(env.vehicle, "semantic.jpg".format())
        # rgb_cam.save_image(env.vehicle, "rgb_{}.jpg".format(h))
        # env.engine.screenshot()


    env = SafeMetaDriveEnv(
        {
            "num_scenarios": 1,
            "traffic_density": 0.,
            "accident_prob": 1.,
            "start_seed": 4,
            "map": "SSSSS",
            "manual_control": True,
            "use_render": True,
            "image_observation": True,
            "rgb_clip": True,
            "interface_panel": ["semantic_camera"],
            "sensors": dict(semantic_camera=(SemanticCamera, 800, 600)),
            "vehicle_config": dict(image_source="semantic_camera"),
            # "map_config": {
            #     BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
            #     BaseMap.GENERATE_CONFIG: 12,
            #     BaseMap.LANE_WIDTH: 3.5,
            #     BaseMap.LANE_NUM: 3,
            # }
        }
    )
    env.reset()
    env.engine.accept("m", get_image, extraArgs=[env])

    for i in range(1, 100000):
        o, r, tm, tc, info = env.step([0, 1])
        assert env.observation_space.contains(o)
        if env.config["use_render"]:
            # for i in range(ImageObservation.STACK_SIZE):
            #     ObservationType.show_gray_scale_array(o["image"][:, :, i])
            env.render()
        # if tm or tc:
        #     # print("Reset")
        #     env.reset()
    env.close()
