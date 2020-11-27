from pg_drive.envs.generalization_racing import GeneralizationRacing


class ResetEnv(GeneralizationRacing):
    def __init__(self):
        super(ResetEnv, self).__init__(
            {
                "environment_num": 1,
                "traffic_density": 0.0,
                "start_seed": 4,
                "image_buffer_name": "front_cam",
                "manual_control": True,
                "use_render": True,
                "use_rgb": True,
                "rgb_clip": True,
                "vehicle_config":dict(mini_map=(512, 512, 250))
            }
        )


if __name__ == "__main__":
    env = ResetEnv()
    env.reset()
    env.bullet_world.accept("m", env.vehicle.mini_map.save_image)
    env.bullet_world.accept("c", env.vehicle.front_cam.save_image)
    import time
    from pg_drive.envs.observation_type import ObservationType, ImageObservation
    for i in range(1, 100000):
        # start = time.time()
        # print("Step: ", i)
        o, r, d, info = env.step([0, 1])
        assert env.observation_space.contains(o)
        for i in range(ImageObservation.STACK_SIZE):
            ObservationType.show_gray_scale_array(o["image"][:,:,i])
        # print(r)
        # print(o)
        # print(time.time() - start)
        # print(len(o), "Vs.", env.observation_space.shape[0])
        # print(info)
        env.render(text={"can you see me": i})
        if d:
            print("Reset")
            env.reset()
    env.close()
