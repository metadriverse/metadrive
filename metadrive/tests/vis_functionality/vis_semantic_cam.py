from metadrive.component.sensors.semantic_camera import SemanticCamera
from metadrive.engine.asset_loader import AssetLoader
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.envs.scenario_env import ScenarioEnv

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

    env = ScenarioEnv(
        {
            "use_render": True,
            "image_observation": False,
            "norm_pixel": True,
            "show_interface": True,
            "agent_policy": ReplayEgoCarPolicy,
            "interface_panel": ["semantic_camera"],
            "sensors": dict(semantic_camera=(SemanticCamera, 800, 600)),
            "vehicle_config": dict(image_source="semantic_camera"),
            "data_directory": AssetLoader.file_path("nuscenes", unix_style=False),
        }
    )
    env.reset(seed=0)
    env.engine.accept("m", get_image, extraArgs=[env])
    import cv2
    for i in range(1, 100000):
        o, r, tm, tc, info = env.step([0, 1])
        assert env.observation_space.contains(o)
        # save
        # rgb_cam = env.engine.get_sensor(env.vehicle.config["image_source"])
        # # rgb_cam.save_image(env.vehicle, name="{}.png".format(i))
        # cv2.imshow('img', o["image"][..., -1])
        # cv2.waitKey(1)

        # if env.config["use_render"]:
        # for i in range(ImageObservation.STACK_SIZE):
        #     ObservationType.show_gray_scale_array(o["image"][:, :, i])
        # env.render()
        # if tm or tc:
        #     # print("Reset")
        #     env.reset()
    env.close()
