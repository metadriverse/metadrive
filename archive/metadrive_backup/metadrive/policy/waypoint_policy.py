import gymnasium as gym
import numpy as np

from metadrive.policy.base_policy import BasePolicy
from metadrive.utils import waypoint_utils


class WaypointPolicy(BasePolicy):
    """
    This policy will have the trajectory data being overwritten on the fly.
    """
    def __init__(self, obj, seed):
        super(WaypointPolicy, self).__init__(control_object=obj, random_seed=seed)
        self.horizon = self.engine.global_config.get("waypoint_horizon", 10)
        self.cache = None
        self.cache_last_update = 0

    @classmethod
    def get_input_space(cls):
        from metadrive.engine.engine_utils import get_global_config
        horizon = get_global_config().get("waypoint_horizon", 10)
        return gym.spaces.Dict(
            dict(position=gym.spaces.Box(float("-inf"), float("inf"), shape=(horizon, 2), dtype=np.float32), )
        )

    def _convert_to_world_coordinates(self, waypoint_positions):
        """
        This function is used to convert the waypoint positions from the local frame to the world frame
        """
        obj_heading = np.array(self.control_object.heading_theta).reshape(1, ).repeat(waypoint_positions.shape[0])
        obj_position = np.array(self.control_object.position).reshape(1, 2)
        rotated = waypoint_utils.rotate(
            waypoint_positions[:, 0],
            waypoint_positions[:, 1],
            obj_heading,
        )
        translated = rotated + obj_position
        return translated

    def reset(self):
        """
        Reset the policy
        """
        self.cache = None
        self.cache_last_update = 0
        super(WaypointPolicy, self).reset()

    def act(self, agent_id):
        assert self.engine.external_actions is not None
        actions = self.engine.external_actions[agent_id]

        if actions is not None:

            waypoint_positions = actions["position"]
            assert waypoint_positions.ndim == 2
            assert waypoint_positions.shape[1] == 2

            world_positions = self._convert_to_world_coordinates(waypoint_positions)
            headings = np.array(waypoint_utils.reconstruct_heading(world_positions))

            # dt should be 0.1s in default settings
            dt = self.engine.global_config["physics_world_step_size"] * self.engine.global_config["decision_repeat"]

            angular_velocities = np.array(waypoint_utils.reconstruct_angular_velocity(headings, dt))
            velocities = np.array(waypoint_utils.reconstruct_velocity(world_positions, dt))

            duration = len(waypoint_positions)
            assert duration == self.horizon, "The length of the waypoint positions should be equal to the horizon: {} vs {}".format(
                duration, self.horizon
            )

            self.cache = dict(
                position=world_positions,
                velocity=velocities,
                heading=headings,
                angular_velocity=angular_velocities,
            )
            self.cache_last_update = self.engine.episode_step

        assert self.cache is not None

        cache_index = self.engine.episode_step - self.cache_last_update
        assert cache_index < self.horizon, "Cache index out of range: {} vs {}".format(cache_index, self.horizon)

        self.control_object.set_position(self.cache["position"][cache_index])
        self.control_object.set_velocity(self.cache["velocity"][cache_index])
        self.control_object.set_heading_theta(self.cache["heading"][cache_index])
        self.control_object.set_angular_velocity(self.cache["angular_velocity"][cache_index])

        # A legacy code to set the static mode of the agent
        # If set_static, then the agent will not "fall from the sky".
        # However, the physics simulation will not apply too to the agent.
        # So in the visualization, the image will be very chunky as the agent will suddenly move to the next
        # position for each step.
        if self.engine.global_config.get("set_static", False):
            self.control_object.set_static(True)

        return None  # Return None action so the base vehicle will not overwrite the steering & throttle
