from pgdrive.constants import TerminationState
from pgdrive.component.blocks.first_block import FirstPGBlock
from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.utils import Config


class SafePGDriveEnv(PGDriveEnv):
    def default_config(self) -> Config:
        config = super(SafePGDriveEnv, self).default_config()
        config.update(
            {
                "environment_num": 100,
                "accident_prob": 0.5,
                "traffic_density": 0.1,
                "safe_rl_env": True,  # Should always be True. But we just leave it here for historical reason.
                "cost_to_reward": False,

                # ===== reward scheme =====
                "crash_vehicle_penalty": 0.,
                "crash_object_penalty": 0.,
                "out_of_road_penalty": 0.,

                # ===== cost scheme =====
                "crash_vehicle_cost": 1,
                "crash_object_cost": 0.5,
                "out_of_road_cost": 1.,  # only give penalty for out_of_road
                "use_lateral": False
            },
            allow_add_new_key=True
        )
        return config

    def _post_process_config(self, config):
        config = super(SafePGDriveEnv, self)._post_process_config(config)
        if config["cost_to_reward"]:
            config["crash_vehicle_penalty"] = config["crash_vehicle_cost"]
            config["crash_object_penalty"] = config["crash_object_cost"]
            config["out_of_road_penalty"] = config["out_of_road_cost"]
        return config

    def done_function(self, vehicle_id: str):
        done, done_info = super(SafePGDriveEnv, self).done_function(vehicle_id)
        if self.config["safe_rl_env"]:
            if done_info[TerminationState.CRASH_VEHICLE]:
                done = False
            elif done_info[TerminationState.CRASH_OBJECT]:
                done = False
        return done, done_info

    def setup_engine(self):
        from pgdrive.manager.object_manager import TrafficObjectManager
        super(SafePGDriveEnv, self).setup_engine()
        self.engine.register_manager("object_manager", TrafficObjectManager())

    def _is_out_of_road(self, vehicle):
        return vehicle.out_of_route


if __name__ == "__main__":
    env = SafePGDriveEnv(
        {
            # "accident_prob": 1.0,
            "manual_control": True,
            "use_render": True,
            # "debug": True,
            'environment_num': 1,
            "start_seed": 22,
            # "traffic_density": 0.2,
            # "environment_num": 1,
            # # "start_seed": 187,
            # "out_of_road_cost": 1,
            # "debug": True,
            "map": "X",
            # # "cull_scene": True,
            "vehicle_config": {
                "spawn_lane_index": (FirstPGBlock.NODE_2, FirstPGBlock.NODE_3, 2)
                # "show_lidar": True,
                # "show_side_detector": True,
                # "show_lane_line_detector": True,
                # "side_detector": dict(num_lasers=2, distance=50),  # laser num, distance
                # "lane_line_detector": dict(num_lasers=2, distance=20),  # laser num, distance
            }
        }
    )

    o = env.reset()
    total_cost = 0
    for i in range(1, 100000):
        o, r, d, info = env.step([0, 0])
        total_cost += info["cost"]
        env.render(text={"cost": total_cost, "seed": env.current_seed, "reward": r})
        if d:
            total_cost = 0
            print("done_cost:{}".format(info["cost"]), "done_reward;{}".format(r))
            print("Reset")
            env.reset()
    env.close()
