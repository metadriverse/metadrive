from metadrive.component.sensors.rgb_camera import RGBCamera
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy

if __name__ == "__main__":
    env = MetaDriveEnv(
        {
            "num_scenarios": 1,
            "traffic_density": 0.1,
            "start_seed": 4,
            "stack_size": 5,
            "agent_policy": IDMPolicy,
            "debug": True,
            "use_mesh_terrain": True,
            # "debug_panda3d": True,
            "vehicle_config": dict(image_source="rgb_camera"),
            "sensors": {
                "rgb_camera": (RGBCamera, 521, 512)
            },
            "interface_panel": ["dashboard", "rgb_camera"],
            "manual_control": True,
            "use_render": False,
            "image_observation": True,  # it is a switch telling metadrive to use rgb as observation
            "norm_pixel": True,  # clip rgb to range(0,1) instead of (0, 255)
            # "pstats": True,
        }
    )
    env.reset()
    # # print m to capture rgb observation
    env.engine.accept("m", env.engine.terrain.reload_terrain_shader)
    import cv2

    for i in range(1, 100000):
        o, r, tm, tc, info = env.step([0, 1])
        assert env.observation_space.contains(o)
        # save
        rgb_cam = env.engine.get_sensor(env.agent.config["image_source"])
        # rgb_cam.save_image(env.agent, name="{}.png".format(i))
        cv2.imshow('img', o["image"][..., -1])
        cv2.waitKey(1)

        # if env.config["use_render"]:
        # for i in range(ImageObservation.STACK_SIZE):
        #      ObservationType.show_gray_scale_array(o["image"][:, :, i])
        # image = env.render(mode="any str except human", text={"can you see me": i})
        # ObservationType.show_gray_scale_array(image)
        # if tm or tc:
        #     # print("Reset")
        #     env.reset()
    env.close()
