"""
Similar to native MetaDriveEnv, which allows PG map generation to create infinite number of scenarios for
generalization experiments, this file provides a environment where you can further randomize the dynamics of ego
vehicle.

Note that the sampled dynamics parameters will not be changed if you don't change the global seed.
This means that if num_scenarios = 1, then you will deterministically sample the same agent with
the same dynamics. Set num_scenarios > 1 to allow more diverse dynamics.
"""

from metadrive.envs.metadrive_env import MetaDriveEnv
from metadrive.manager.agent_manager import VehicleAgentManager

VaryingDynamicsConfig = dict(
    vehicle_config=dict(vehicle_model="varying_dynamics", ),
    random_dynamics=dict(
        # We will sample each parameter from (min_value, max_value)
        # You can set it to None to stop randomizing the parameter.
        max_engine_force=(100, 3000),
        max_brake_force=(20, 600),
        wheel_friction=(0.1, 2.5),
        max_steering=(10, 80),  # The maximum steering angle if action = +-1
        mass=(300, 3000)
    )
)


class VaryingDynamicsAgentManager(VehicleAgentManager):
    def reset(self):
        # Randomize ego vehicle's dynamics here
        random_fields = self.engine.global_config["random_dynamics"]
        dynamics = {}
        for parameter, para_range in random_fields.items():
            if para_range is None:
                continue
            elif isinstance(para_range, (tuple, list)):
                assert len(para_range) == 2
                assert para_range[1] >= para_range[0]
                if para_range[1] == para_range[0]:
                    dynamics[parameter] = para_range[0]
                else:
                    dynamics[parameter] = self.np_random.uniform(para_range[0], para_range[1])
            else:
                raise ValueError("Unknown parameter range: {}".format(para_range))

        assert len(self.engine.global_config["agent_configs"]) == 1, "Only supporting single-agent now!"
        self.engine.global_config["agent_configs"]["default_agent"].update(dynamics)
        super(VaryingDynamicsAgentManager, self).reset()


class VaryingDynamicsEnv(MetaDriveEnv):
    @classmethod
    def default_config(cls):
        config = super(VaryingDynamicsEnv, cls).default_config()
        config.update(VaryingDynamicsConfig)
        return config

    def _get_agent_manager(self):
        return VaryingDynamicsAgentManager(init_observations=self._get_observations())


if __name__ == '__main__':
    # Local test
    env = VaryingDynamicsEnv({
        "num_scenarios": 10,  # Allow 10 random envs.
    })
    for ep in range(3):
        obs, _ = env.reset()
        print("Current Dynamics Parameters:", env.agent.get_dynamics_parameters())
        for step in range(1000):
            o, r, tm, tc, i = env.step(env.action_space.sample())
            if tm or tc:
                break
