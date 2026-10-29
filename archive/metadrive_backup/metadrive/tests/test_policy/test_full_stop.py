from metadrive.envs import MetaDriveEnv


def test_full_stop():
    config = {
        "use_render": False,
        "map": "y",
        "vehicle_config": {
            "spawn_position_heading": [[10, 10], 0],
        },
    }

    env = MetaDriveEnv(config)
    env.reset()

    try:
        # Driving forward phase
        for step in range(30):  # Drive forward for 30 steps
            env.step([0, 1])  # Full throttle, no steering
            # print("Speed: ", env.agent.speed)
            # env.render(mode="topdown")

        success = False
        print("Starting Brake Phase")
        for i in range(20):  # Continue braking until the car stops
            env.step([0.0, -1.0])  # No throttle, apply brake
            # print(i, "Speed: ", env.agent.speed)
            # env.render(mode="topdown")

            if env.agent.speed <= 0.01:  # Stop if speed is effectively zero
                print("Car has stopped.")
                success = True
                break

        if not success:
            raise ValueError("Car did not stop after 20 steps")

    finally:
        env.close()


if __name__ == '__main__':
    test_full_stop()
