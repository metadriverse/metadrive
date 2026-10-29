import time

from metadrive.component.sensors.semantic_camera import SemanticCamera
from metadrive.engine.asset_loader import AssetLoader
from metadrive.policy.replay_policy import ReplayEgoCarPolicy
from metadrive.envs.scenario_env import ScenarioEnv

if __name__ == "__main__":

    def get_image(env):
        semantic_cam = env.engine.get_sensor(env.agent.config["image_source"])
        # for h in range(-180, 180, 20):
        #     env.engine.graphicsEngine.renderFrame()
        #     semantic_cam.get_cam().setH(h)
        #     # rgb_cam.get_cam().setH(h)
        semantic_cam.save_image(env.agent, "semantic.jpg".format())
        # rgb_cam.save_image(env.agent, "rgb_{}.jpg".format(h))
        # env.engine.screenshot()

    env = ScenarioEnv(
        {
            "use_render": True,
            "image_observation": False,
            "num_scenarios": 10,
            "debug": True,
            "debug_static_world": True,
            "norm_pixel": True,
            "show_interface": True,
            "show_sidewalk": True,
            "show_crosswalk": True,
            "agent_policy": ReplayEgoCarPolicy,
            "interface_panel": ["semantic_camera"],
            "sensors": dict(semantic_camera=(SemanticCamera, 800, 600)),
            "vehicle_config": dict(image_source="semantic_camera"),
            "data_directory": AssetLoader.file_path("nuscenes", unix_style=False),
        }
    )
    start = time.time()
    env.reset(seed=0)
    print(time.time() - start)
    env.engine.accept("m", get_image, extraArgs=[env])

    # env.engine.current_map.show_bounding_box()
    import cv2
    for i in range(1, 100000):
        o, r, tm, tc, info = env.step([0, 1])
        assert env.observation_space.contains(o)
        # save
        # rgb_cam = env.engine.get_sensor(env.agent.config["image_source"])
        # # rgb_cam.save_image(env.agent, name="{}.png".format(i))
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
