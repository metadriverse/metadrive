from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.pg_config import PGConfig
from pgdrive.scene_creator.blocks.first_block import FirstBlock
from pgdrive.utils import setup_logger
from pgdrive.envs.multi_agent_pgdrive import MultiAgentPGDrive


def test_cull_scene(use_render=False):
    class TestCull(MultiAgentPGDrive):
        def default_config(self) -> PGConfig:
            config = PGDriveEnv.default_config()
            config.update({
                "target_vehicle_configs": {},
                "num_agents": 0,
            })
            config.extend_config_with_unknown_keys({"crash_done": True})
            return config

    env = TestCull(
        {
            "use_render": use_render,
            "manual_control": use_render,
            "map": "SSSSCS",
            "debug": True,
            "target_vehicle_configs": {
                "agent0": {
                    "born_longitude": 10,
                    "born_lateral": 1.5,
                    "born_lane_index": ("5C0_0_", "5C0_1_", 1),
                    # "show_lidar": True
                },
                "agent1": {
                    "born_longitude": 10,
                    # "show_lidar": True,
                    "born_lateral": -1,
                }
            },
            "num_agents": 2,
        }
    )
    try:
        pass_test = False
        o = env.reset()
        for i in range(1, 10000):
            o, r, d, info = env.step({"agent0": [-1, 0], "agent1": [0, 0]})
            if info["agent0"]["crash_vehicle"]:
                pass_test = True
                break
        assert pass_test, "Cull scene error! collision function is invalid!"
    finally:
        env.close()


if __name__ == "__main__":
    test_cull_scene(True)
