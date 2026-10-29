from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.policy.idm_policy import IDMPolicy
from metadrive.envs.marl_envs.multi_agent_metadrive import MULTI_AGENT_METADRIVE_DEFAULT_CONFIG
MULTI_AGENT_METADRIVE_DEFAULT_CONFIG["force_seed_spawn_manager"] = True


def test_ma_bidirection_idm(render=False):
    env = MetaDriveEnv(
        {
            "num_scenarios": 1,
            "traffic_density": 0,
            "start_seed": 22,
            "manual_control": False,
            "use_render": render,
            "map": "yBY",
            "agent_policy": IDMPolicy,
        }
    )
    index = ('1y0_1_', '2B0_0_', 0)
    try:
        o, _ = env.reset()
        env.agent.set_velocity([1, 0.1], 10)
        # print(env.agent.speed)
        pass_test = False
        for s in range(1, 10000):
            o, r, tm, tc, info = env.step(env.action_space.sample())
            _, lat = env.agent.lane.local_coordinates(env.agent.position)
            if abs(lat) > env.agent.lane.width / 2 + 0.1 and len(env.agent.navigation.current_ref_lanes) == 1:
                raise ValueError("IDM can not pass bidirection block")
            if env.agent.lane.index == index and abs(lat) < 0.1:
                pass_test = True
            if (tm or tc) and info["arrive_dest"]:
                break
        assert pass_test
        env.close()
    finally:
        env.close()


if __name__ == "__main__":
    test_ma_bidirection_idm(True)
