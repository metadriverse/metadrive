import gymnasium as gym
import numpy as np

from metadrive.policy.base_policy import BasePolicy
from metadrive.utils import waypoint_utils

class ClosedLoopPolicy(BasePolicy):
    """
    This policy will have the trajectory data being overwritten on the fly.
    """
    def __init__(self, control_object, track, random_seed=None):
        super(ClosedLoopPolicy, self).__init__(control_object=control_object, random_seed=random_seed)
        self.horizon = self.engine.global_config.get("waypoint_horizon", 10)
        self.agent_mask = None

    @property
    def is_current_step_valid(self):
        return self.agent_mask[self.engine.episode_step]

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
        super(ClosedLoopPolicy, self).reset()

    def get_actions(self, scenario_id):
        tracks = self.engine.data_manager.current_scenario["tracks"]
        assert scenario_id in tracks, "Scenario ID {} not found in tracks".format(scenario_id)
        track = tracks[scenario_id]
        assert track is not None, "Track {} is None".format(scenario_id)
        agent_motion = track["state"]

        return agent_motion


    # def act(self, agent_id, scenario_id):
    def act(self, *args, **kwargs):
        obj_id_to_scenario_id = self.engine.traffic_manager.get_obj_id_to_scenario_id()

        agent_id = self.control_object.id
        scenario_id = obj_id_to_scenario_id[agent_id]

        agent_motion = self.get_actions(scenario_id)
        assert agent_motion is not None

        start_idx = (self.engine.episode_step // self.horizon)* self.horizon
        end_idx = start_idx + self.horizon
        update_idx = self.engine.episode_step % self.horizon

        waypoint_positions = agent_motion["position"][start_idx: end_idx, :2]
        waypoint_headings = agent_motion["heading"][start_idx : end_idx]
        waypoint_velocities = agent_motion["velocity"][start_idx : end_idx]
        waypoint_masks = agent_motion["valid"][start_idx : end_idx]


        self.agent_mask = agent_motion["valid"]

        assert waypoint_positions is not None and waypoint_headings is not None and waypoint_velocities is not None
        assert waypoint_positions.ndim == 2
        assert waypoint_positions.shape[1] == 2

        dt = self.engine.global_config["physics_world_step_size"] * self.engine.global_config["decision_repeat"]
        assert dt == 0.1 # dt should be 0.1s in default settings

        world_positions = waypoint_positions # since we are using SD's GT data now
        velocities = np.array(waypoint_utils.reconstruct_velocity(world_positions, dt))
        angular_velocities = np.array(waypoint_utils.reconstruct_angular_velocity(waypoint_headings, dt))

        duration = len(waypoint_positions)
        assert duration == self.horizon, "The length of the waypoint positions should be equal to the horizon: {} vs {}".format(
            duration, self.horizon
        )

        agent_waypoint= dict(
            position=world_positions,
            velocity=velocities,
            heading=waypoint_headings,
            angular_velocity=angular_velocities,
            valid=waypoint_masks,
        )

        if agent_waypoint["valid"][update_idx]:
            self.control_object.set_position(agent_waypoint["position"][update_idx])
            self.control_object.set_velocity(agent_waypoint["velocity"][update_idx])
            self.control_object.set_heading_theta(agent_waypoint["heading"][update_idx])
            self.control_object.set_angular_velocity(agent_waypoint["angular_velocity"][update_idx])

        # A legacy code to set the static mode of the agent
        # If set_static, then the agent will not "fall from the sky".
        # However, the physics simulation will not apply too to the agent.
        # So in the visualization, the image will be very chunky as the agent will suddenly move to the next
        # position for each step.
        if self.engine.global_config.get("set_static", False):
            self.control_object.set_static(True)

        return None  # Return None action so the base vehicle will not overwrite the steering & throttle
