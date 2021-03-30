from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.utils import PGConfig


class SafePGDriveEnv(PGDriveEnv):
    def default_config(self) -> PGConfig:
        config = super(SafePGDriveEnv, self).default_config()
        config.update(
            {
                "accident_prob": 0.5,
                "crash_vehicle_cost": 1,
                "crash_object_cost": 1,
                "crash_vehicle_penalty": 0.,
                "crash_object_penalty": 0.,
                "out_of_road_cost": 0.,  # only give penalty for out_of_road
                "traffic_density": 0.2,
            }
        )
        return config

    def done_function(self, vehicle):
        done, done_info = super(SafePGDriveEnv, self).done_function(vehicle)
        if done_info["crash_vehicle"]:
            done = False
        elif done_info["crash_object"]:
            done = False
        return done, done_info


if __name__ == "__main__":
    env = SafePGDriveEnv(
        {
            "manual_control": True,
            "use_render": True,
            "environment_num": 100,
            "start_seed": 75,
            "debug": True,
            "cull_scene": True,
            "pg_world_config": {
                "pstats": True
            }
        }
    )

    o = env.reset()
    total_cost = 0
    for i in range(1, 100000):
        o, r, d, info = env.step([0, 1])
        total_cost += info["cost"]
        env.render(text={"cost": total_cost, "seed": env.current_map.random_seed})
        if d:
            total_cost = 0
            print("Reset")
            env.reset()
    env.close()
