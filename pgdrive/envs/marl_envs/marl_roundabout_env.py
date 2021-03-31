from pgdrive.envs.multi_agent_pgdrive import MultiAgentPGDrive
from pgdrive.utils import get_np_random
from pgdrive.scene_creator.blocks.roundabout import Roundabout
from pgdrive.utils import PGConfig


class MultiAgentRoundaboutEnv(MultiAgentPGDrive):
    target_nodes = [
        Roundabout.node(1, 0, 0),
        Roundabout.node(1, 0, 1),
        Roundabout.node(1, 1, 0),
        Roundabout.node(1, 1, 1),
        Roundabout.node(1, 2, 0),
        Roundabout.node(1, 2, 1),
        Roundabout.node(1, 3, 0),
        Roundabout.node(1, 3, 1),
    ]

    def default_config(self) -> PGConfig:
        config = MultiAgentPGDrive.default_config()
        config["target_vehicle_configs"] = {}
        config.update(
            {
                "map": "O",
                "vehicle_config": {
                    "born_longitude": 0,
                    "born_lateral": 0
                },
                # clear base config
                "num_agents": 4,
                "map_config": {
                    "lane_num": 3
                },
            },
            allow_overwrite=True
        )
        config = self._update_agent_pos_configs(config)
        return config

    def _update_agent_pos_configs(self, config):
        target_vehicle_configs = []
        assert config["num_agents"] < config["map_config"]["lane_num"] * len(self.target_nodes), "Too many agents!"
        for i in range(-1, len(self.target_nodes) - 1):
            for lane_idx in range(config["map_config"]["lane_num"]):
                target_vehicle_configs.append(
                    (
                        "agent_{}_{}".format(i + 1, lane_idx),
                        (self.target_nodes[i], self.target_nodes[i + 1], lane_idx),
                    )
                )
        target_agents = get_np_random().choice(
            [i for i in range(len(self.target_nodes) * (config["map_config"]["lane_num"]))],
            config["num_agents"],
            replace=False
        )
        ret = {}
        for idx in target_agents:
            agent_name, v_config = target_vehicle_configs[idx]
            ret[agent_name] = dict(born_lane_index=v_config)
        config["target_vehicle_configs"] = ret
        return config

    def step(self, actions):
        o, r, d, i = super(MultiAgentRoundaboutEnv, self).step(actions)
        self._update_target()
        return o, r, d, i

    def _update_target(self):
        for v_id, v in self.vehicles.items():
            if v.lane_index[0] in self.target_nodes:
                last_idx = self.target_nodes.index(v.lane_index[0]) - 2
                dest = v.routing_localization.checkpoints[-1]
                new_target = self.target_nodes[last_idx]
                if new_target != dest:
                    v.routing_localization.set_route(v.lane_index[0], new_target)


if __name__ == "__main__":
    env = MultiAgentRoundaboutEnv(
        {
            "use_render": True,
            "debug": False,
            "manual_control": True,
            "pg_world_config": {
                "pstats": True
            },
            "crash_done": False,
        }
    )
    o = env.reset()
    total_r = 0
    for i in range(1, 100000):
        o, r, d, info = env.step(env.action_space.sample())
        for r_ in r.values():
            total_r += r_
        # o, r, d, info = env.step([0,1])
        d.update({"total_r": total_r})
        env.render(text=d)
        if len(env.vehicles) == 0:
            total_r = 0
            print("Reset")
            env.reset()
    env.close()
