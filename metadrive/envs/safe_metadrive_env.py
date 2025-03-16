from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.utils import Config


class SafeMetaDriveEnv(MetaDriveEnv):
    def default_config(self) -> Config:
        config = super(SafeMetaDriveEnv, self).default_config()
        config.update(
            {
                "num_scenarios": 100,
                "accident_prob": 0.8,
                "traffic_density": 0.05,
                "crash_vehicle_done": False,
                "crash_object_done": False,
                "cost_to_reward": False,
                "horizon": 1000,
            },
            allow_add_new_key=True
        )
        return config

    def __init__(self, config=None):
        super(SafeMetaDriveEnv, self).__init__(config)
        self.episode_cost = 0

    def reset(self, *args, **kwargs):
        self.episode_cost = 0
        return super(SafeMetaDriveEnv, self).reset(*args, **kwargs)

    def cost_function(self, vehicle_id: str):
        cost, step_info = super(SafeMetaDriveEnv, self).cost_function(vehicle_id)
        self.episode_cost += cost
        step_info["total_cost"] = self.episode_cost
        return cost, step_info


if __name__ == "__main__":
    env = SafeMetaDriveEnv(
        {
            "accident_prob": 0.5,
            "traffic_density": 0.15,
            # "manual_control": True,
            "agent_policy": IDMPolicy,
            "use_render": True,
            # "debug": True,
            'num_scenarios': 100,
            "start_seed": 129,
            # "traffic_density": 0.2,
            # "num_scenarios": 1,
            # # "start_seed": 187,
            # "out_of_road_cost": 1,
            # "debug": True,
            # "map": "X",
            # "cost_to_reward":True,
            "vehicle_config": {
                "spawn_lane_index": (FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, 2),
                # "show_lidar": True,
                # "show_side_detector": True,
                # "show_lane_line_detector": True,
                # "side_detector": dict(num_lasers=2, distance=50),  # laser num, distance
                # "lane_line_detector": dict(num_lasers=2, distance=20),  # laser num, distance
            }
        }
    )

    o, _ = env.reset()
    total_cost = 0
    for i in range(1, 100000):
        o, r, tm, tc, info = env.step([0, 0])
        total_cost += info["cost"]
        env.render(
            text={
                "cost": total_cost,
                "seed": env.current_seed,
                "reward": r,
                "total_cost": info["total_cost"],
            }
        )
        if info["crash_vehicle"]:
            print("crash_vehicle:cost {}, reward {}".format(info["cost"], r))
        if info["crash_object"]:
            print("crash_object:cost {}, reward {}".format(info["cost"], r))

        # if d:
        #     total_cost = 0
        #     print("done_cost:{}".format(info["cost"]), "done_reward;{}".format(r))
        #     print("Reset")
        #     env.reset()
    env.close()
