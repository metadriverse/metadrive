from metadrive.component.sensors.semantic_camera import SemanticCamera
from metadrive.engine.asset_loader import AssetLoader
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.envs.real_data_envs.nuscenes_env import NuScenesEnv

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

    env = NuScenesEnv(
        {
            "use_render": True,
            "image_observation": True,
            "rgb_clip": True,
            "show_interface": True,
            "agent_policy": ReplayEgoCarPolicy,
            "interface_panel": ["semantic_camera"],
            "sensors": dict(semantic_camera=(SemanticCamera, 800, 600)),
            "vehicle_config": dict(image_source="semantic_camera"),
            "data_directory": AssetLoader.file_path("waymo", return_raw_style=False),
        }
    )
    env.reset(seed=1)
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
