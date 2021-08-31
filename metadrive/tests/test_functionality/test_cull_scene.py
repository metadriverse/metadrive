from metadrive.constants import TerminationState
from metadrive.envs.marl_envs.multi_agent_metadrive import MultiAgentMetaDrive
from metadrive.utils import Config


def _test_cull_scene(use_render=True):
    # we do not do scene cull now
    class TestCull(MultiAgentMetaDrive):
        def default_config(self) -> Config:
            config = MultiAgentMetaDrive.default_config()
            config.update(
                {
                    "target_vehicle_configs": {},
                    "num_agents": 0,
                    "crash_done": True,
                }, allow_add_new_key=True
            )
            return config

    for _ in range(5):
        env = TestCull(
            {
                "use_render": use_render,
                "manual_control": False,
                "map": "SSSSCS",
                "debug": True,
                "target_vehicle_configs": {
                    "agent0": {
                        "spawn_longitude": 10,
                        "spawn_lateral": 2.5,
                        "spawn_lane_index": ("5C0_0_", "5C0_1_", 1),
                    },
                    "agent1": {
                        "spawn_longitude": 12,  # locate a little forward
                        "spawn_lateral": 2.2,
                        "spawn_lane_index": ("5C0_0_", "5C0_1_", 1),
                    }
                },
                "num_agents": 2,
                "traffic_density": 0.4,
            }
        )
        env._DEBUG_RANDOM_SEED = 1
        try:
            pass_test = False
            o = env.reset()
            for i in range(1, 200):
                actions = {"agent0": [1, 0.2], "agent1": [0, 0]}
                if "agent0" not in env.vehicles:
                    actions.pop("agent0")
                if "agent1" not in env.vehicles:
                    actions.pop("agent1")
                o, r, d, info = env.step(actions)
                if any(d.values()):
                    if info["agent0"][TerminationState.CRASH_VEHICLE]:
                        pass_test = True
                    break
            assert pass_test, "Cull scene error! collision function is invalid!"
        finally:
            env.close()
            env._DEBUG_RANDOM_SEED = None


if __name__ == "__main__":
    # test_cull_scene(True)
    test_cull_scene(False)
