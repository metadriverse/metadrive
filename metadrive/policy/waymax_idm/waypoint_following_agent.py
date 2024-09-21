# Copyright 2023 The Waymax Authors.
#
# Licensed under the Waymax License Agreement for Non-commercial Use
# Use (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at
#
#     https://github.com/waymo-research/waymax/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Implementations for waypoint following sim agents."""
import abc
from typing import Optional, Callable

import chex
import jax
import numpy as np
from jax import numpy as jnp

from metadrive.policy.waymax_idm import datatypes
from metadrive.policy.waymax_idm import geometry
from metadrive.policy.waymax_idm import sim_agent

# Default values to use for invalid objects.
_DEFAULT_LEAD_DISTANCE = -1.0  # Units: m
_DEFAULT_LEAD_VELOCITY = -1.0  # Units: m/s
# Minimum lead distance to prevent divide-by-zero errors.
_MINIMUM_LEAD_DISTANCE = 0.1  # Units: m
_DEFAULT_TIME_DELTA = 0.1  # Units: m/s

# Distance to the final waypoint at which to consider an object to have
# reached the end of its logged trajectory. If an agent reaches the end
# of a trajectory, it will be invalidated.
_REACHED_END_OF_TRAJECTORY_THRESHOLD = 5e-2  # Units: m
# Invalidate an object if it deviates from the reference traj by this amount.
# Generally this should be lower than the IDM agent's
# additional_lookahead_distance to avoid situations where agent will miss
# collisions because the collisions are too far ahead of the logs.
_DISTANCE_TO_REF_THRESHOLD = 5.0  # Units: m
# Max speed until which to consider an object stationary. Stationary objects
# will not be updated by the sim agent.
_STATIC_SPEED_THRESHOLD = 1.0  # Units: m/s


class WaypointFollowingPolicy(sim_agent.SimAgentActor):
    """A base class for all waypoint-following sim agents.

    The WaypointFollowingPolicy will force sim agents to travel along a
    pre-defined path (the agent's future in the log trajectory). The behavior
    of the vehicle is determined by setting its speed via the update_speed()
    method, which will update the velocity of the vehicle.
    """

    def __init__(
            self,
            is_controlled_func: Optional[
                Callable[[datatypes.SimulatorState], jax.Array]
            ] = None,
            invalidate_on_end: bool = False,
    ):
        super().__init__(is_controlled_func=is_controlled_func)
        self.invalidate_on_end = invalidate_on_end

    def update_trajectory(
            self, state: datatypes.SimulatorState
    ) -> datatypes.TrajectoryUpdate:
        """Returns a trajectory update of shape (..., num_objects, 1)."""
        # new_speed = (..., num_objects), new_valid = (..., num_objects)
        new_speed, new_valid = self.update_speed(state)
        next_traj = self._get_next_trajectory_by_projection(
            state.log_trajectory,
            state.current_sim_trajectory,
            new_speed,
            new_valid,
            dt=_DEFAULT_TIME_DELTA,
        )

        return datatypes.TrajectoryUpdate(
            x=next_traj.x,
            y=next_traj.y,
            yaw=next_traj.yaw,
            vel_x=next_traj.vel_x,
            vel_y=next_traj.vel_y,
            valid=next_traj.valid,
        )

    def _get_next_trajectory_by_projection(
            self,
            log_traj: datatypes.Trajectory,
            cur_sim_traj: datatypes.Trajectory,
            new_speed: jax.Array,
            new_speed_valid: jax.Array,
            dt: float = _DEFAULT_TIME_DELTA,
    ) -> datatypes.Trajectory:
        """Computes the next trajectory.

        Args:
          log_traj: Logged trajectory for the simulation of shape (..., num_objects,
            num_timesteps).
          cur_sim_traj: Current simulated trajectory for the simulation of shape
            (..., num_objects, num_timesteps=1).
          new_speed: Updated speed for the agents after solving for velocity of
            shape (..., num_objects).
          new_speed_valid: Updated validity for the speed updates of the agents
            after (..., num_objects).
          dt: Delta between timesteps of the simulator state.

        Returns:
          The next Trajectory projected onto log_traj of shape
            (..., num_objects, num_timesteps=1).
        """

        # Shape: (..., num_objects, 1).
        cur_speed = cur_sim_traj.speed
        valid = new_speed_valid[..., jnp.newaxis]
        # If new speed is not available, use cur_speed.
        new_speed = jnp.where(valid, new_speed[..., jnp.newaxis], cur_speed)

        # Shape: (..., num_objects, 1).
        dist_travel = (new_speed + cur_speed) / 2 * dt

        next_x = cur_sim_traj.x + dist_travel * jnp.cos(cur_sim_traj.yaw)
        next_y = cur_sim_traj.y + dist_travel * jnp.sin(cur_sim_traj.yaw)

        # Project onto log_traj, i.e. along the direction of closest point.
        # Shape: (..., num_objects, 1, 2)
        next_xy, next_yaw, reached_last_waypoint = _project_to_a_trajectory(
            jnp.stack([next_x, next_y], axis=-1),
            log_traj,
            extrapolate_traj=not self.invalidate_on_end,
        )

        # Freeze the speed for agents that have reached the last waypoint to
        # prevent drift.
        if self.invalidate_on_end:
            default_x_vel = jnp.zeros_like(cur_sim_traj.vel_x)
            default_y_vel = jnp.zeros_like(cur_sim_traj.vel_y)
        else:
            default_x_vel = cur_sim_traj.vel_x
            default_y_vel = cur_sim_traj.vel_y
        new_vel_x = jnp.where(
            reached_last_waypoint,
            default_x_vel,
            new_speed * jnp.cos(cur_sim_traj.yaw),
        )
        new_vel_y = jnp.where(
            reached_last_waypoint,
            default_y_vel,
            new_speed * jnp.sin(cur_sim_traj.yaw),
        )

        # Invalidate moving objects that have reached their final waypoint.
        # This is to avoid invalidating parked cars. Use a threshold velocity
        # since some sim agents will tell the parked cars to move forward since
        # nothing is in front (e.g. IDM).
        if self.invalidate_on_end:
            moving_after_last_waypoint = reached_last_waypoint & (
                    new_speed > _STATIC_SPEED_THRESHOLD
            )
            valid = valid & ~moving_after_last_waypoint

        next_traj = cur_sim_traj.replace(
            x=next_xy[..., 0],
            y=next_xy[..., 1],
            yaw=next_yaw,
            vel_x=new_vel_x,
            vel_y=new_vel_y,
            valid=valid & cur_sim_traj.valid,
        )
        next_traj.validate()
        # Fill invalid values with -1/False
        invalid_traj = jax.tree_util.tree_map(
            datatypes.make_invalid_data, next_traj
        )
        next_traj = jax.tree_util.tree_map(
            lambda x, y: jnp.where(next_traj.valid, x, y), next_traj, invalid_traj
        )
        return next_traj

    @abc.abstractmethod
    def update_speed(
            self, state: datatypes.SimulatorState, dt: float = _DEFAULT_TIME_DELTA
    ) -> tuple[jax.Array, jax.Array]:
        """Updates the speed for each agent in the current simulation step.

        Args:
          state: The simulator state of shape (...).
          dt: Delta between timesteps of the simulator state.

        Returns:
          speeds: A (..., num_objects) float array of speeds.
          valids: A (..., num_objects) bool array of valids.
        """


class IDMRoutePolicy(WaypointFollowingPolicy):
    """A policy implementing the intelligent driver model (IDM).

    This policy uses IDM to compute the acceleration and velocities for the
    agent while it follows its own logged future.
    """

    def __init__(
            self,
            is_controlled_func: Optional[
                Callable[[datatypes.SimulatorState], jax.Array]
            ] = None,
            desired_vel: float = 30.0,
            min_spacing: float = 2.0,
            safe_time_headway: float = 2.0,
            max_accel: float = 2.0,
            max_decel: float = 4.0,
            delta: float = 4.0,
            max_lookahead: int = 10,
            lookahead_from_current_position: bool = True,
            additional_lookahead_points: int = 10,
            additional_lookahead_distance: float = 10.0,
            invalidate_on_end: bool = False,
    ):
        super().__init__(
            is_controlled_func=is_controlled_func,
            invalidate_on_end=invalidate_on_end,
        )
        self.desired_vel = desired_vel
        self.min_spacing_s0 = min_spacing
        self.safe_time_headway = safe_time_headway
        self.max_accel = max_accel
        self.max_decel = max_decel
        self.delta = delta
        self.max_lookahead = max_lookahead
        self.lookahead_from_current_position = lookahead_from_current_position
        self.additional_lookahead_distance = additional_lookahead_distance
        self.additional_headway_points = additional_lookahead_points

    def update_speed(
            self, state: datatypes.SimulatorState, dt: float = _DEFAULT_TIME_DELTA
    ) -> tuple[jax.Array, jax.Array]:
        """Returns the new speed for each agent in the current simulation step.

        Args:
          state: The simulator state of shape (...).
          dt: Delta between timesteps of the simulator state.

        Returns:
          speeds: A (..., num_objects) float array of speeds.
          valids: A (..., num_objects) bool array of valids.
        """
        # Shape: (..., num_objects, num_timesteps).
        log_waypoints = state.log_trajectory
        # Shape: (..., num_objects, 3).
        cur_position = state.current_sim_trajectory.xyz[..., 0, :]
        cur_speed = state.current_sim_trajectory.speed[..., 0]

        # Shape: (..., num_objects).
        accel = self._get_accel(
            log_waypoints, cur_position, cur_speed, state.current_sim_trajectory
        )

        valid = jnp.ones_like(cur_speed, dtype=jnp.bool_)

        speed = cur_speed + dt * accel
        # Speed non-negative.
        speed = jnp.where(speed < 0, 0, speed)
        return speed, valid

    def _get_accel(
            self,
            log_waypoints: datatypes.Trajectory,
            cur_position: jax.Array,
            cur_speed: jax.Array,
            obj_curr_traj: datatypes.Trajectory,
    ) -> jax.Array:
        """Computes vehicle accelerations according to IDM for a single vehicle.

        Note log_waypoints and obj_curr_traj contain the same set of objects, thus
        need to remove collision against oneself when computing pairwise collision.

        Args:
          log_waypoints: A trajectory of the agents' future of shape (...,
            num_objects, num_timesteps).
          cur_position: Current positions for the agents of shape (..., num_objects,
            3).
          cur_speed: Current speeds for the agents of shape (..., num_objects).
          obj_curr_traj: Trajectory containing the state for all current objects of
            shape (..., num_objects, num_timesteps=1).

        Returns:
          A vector of all vehicles' accelerations after solving for them of shape
            (..., num_objects).
        """
        num_obj = obj_curr_traj.num_objects
        prefix_shape = obj_curr_traj.shape[:-2]
        chex.assert_equal_shape_prefix(
            [log_waypoints, cur_position, cur_speed, obj_curr_traj],
            len(prefix_shape) + 1,
        )
        log_waypoints.validate()
        obj_curr_traj.validate()
        # 1. Find the closest waypoint and slice the future from that waypoint.
        if self.lookahead_from_current_position:
            traj = _find_reference_traj_from_log_traj(cur_position, obj_curr_traj, 1)
            chex.assert_shape(traj.xyz, prefix_shape + (num_obj, 1, 3))
            total_lookahead = 1 + self.additional_headway_points
        else:
            traj = _find_reference_traj_from_log_traj(
                cur_position, log_waypoints, self.max_lookahead
            )
            chex.assert_shape(
                traj.xyz, prefix_shape + (num_obj, self.max_lookahead, 3)
            )
            total_lookahead = self.max_lookahead + self.additional_headway_points

        if self.additional_headway_points > 0:
            traj = _add_headway_waypoints(
                traj,
                distance=self.additional_lookahead_distance,
                num_points=self.additional_headway_points,
            )

        # 2. Computes collision indicator as (..., num_objects, num_objects,
        # max_lookahead) between traj (..., num_objects, max_lookahead) and
        # obj_curr_traj (..., num_objects, 1). Make common shape for bboxes:
        # (..., num_objects, num_objects, max_lookahead, 5).
        broadcast_shape = prefix_shape + (num_obj, num_obj, total_lookahead, 5)
        traj_5dof = traj.stack_fields(['x', 'y', 'length', 'width', 'yaw'])
        traj_bbox = jnp.broadcast_to(
            jnp.expand_dims(traj_5dof, axis=-3), broadcast_shape
        )
        obj_curr_5dof = obj_curr_traj.stack_fields(
            ['x', 'y', 'length', 'width', 'yaw']
        )
        obj_bbox = jnp.broadcast_to(
            jnp.expand_dims(obj_curr_5dof, axis=-4), broadcast_shape
        )
        # collision[..., i, j, t] represents if obj i collides with obj j at time t.
        collision_pairwise = geometry.has_overlap(traj_bbox, obj_bbox)
        self_mask = jnp.expand_dims(
            jnp.eye(num_obj, dtype=jnp.bool_)[..., jnp.newaxis],
            axis=np.arange(len(prefix_shape)),
        )
        collision_valid = (
                traj.valid[..., jnp.newaxis, :]
                & obj_curr_traj.valid[..., jnp.newaxis, :, :]
                & ~self_mask
        )

        # (..., num_objects, num_objects, max_lookahead).
        collision_pairwise = jnp.where(collision_valid, collision_pairwise, False)

        # 3. Gets velocity and distance to leading obj (defined as the one with
        # collision).
        # (..., num_objects, 1) -> (..., num_objects, num_objects, max_lookahead).
        obj_speed_tiled = jnp.broadcast_to(
            jnp.expand_dims(obj_curr_traj.speed, axis=-3), collision_pairwise.shape
        )
        lead_vel = self._compute_lead_velocity(
            obj_speed_tiled,
            collision_pairwise,
            obj_curr_traj.valid[..., jnp.newaxis, :, :],
        )

        # agent_future Shape: (..., num_objects, max_lookahead, 3).
        # collision_indicator: (..., num_objects, max_lookahead).
        lead_dist = self._compute_lead_distance(
            agent_future=traj.xyz,
            collision_indicator=collision_pairwise.any(axis=-2),
            current_position=obj_curr_traj.xyz,
            agent_future_valid=traj.valid,
        )

        # 4. Compute accel according to IDM formula.
        s_star = self.min_spacing_s0 + jnp.maximum(
            0,
            cur_speed * self.safe_time_headway
            + cur_speed
            * (cur_speed - lead_vel)
            / (2 * jnp.sqrt(self.max_accel * self.max_decel)),
        )
        # Set 0 for free-road behaviour.
        s_star = jnp.where(
            (lead_dist == _DEFAULT_LEAD_DISTANCE)
            | (lead_vel == _DEFAULT_LEAD_VELOCITY),
            0,
            s_star,
        )

        lead_dist = jnp.where(lead_dist == 0, _MINIMUM_LEAD_DISTANCE, lead_dist)
        return self.max_accel * (
                1
                - (cur_speed / self.desired_vel) ** self.delta
                - (s_star / lead_dist) ** 2
        )

    def _compute_lead_velocity(
            self,
            future_speeds: jax.Array,
            collisions_per_agent: jax.Array,
            future_speeds_valid: Optional[jax.Array] = None,
    ) -> jax.Array:
        """Computes the velocity of the object at the closest collision.

        Args:
          future_speeds: Future speeds per agent of shape (..., num_objects,
            num_timesteps).
          collisions_per_agent: Future collision indications of shape (...,
            num_objects, num_timesteps).
          future_speeds_valid: Boolean mask for future speeds of shape (...,
            num_objects, num_timesteps).

        Returns:
          An array containing the velocity of the colliding object at the
            closest collision of shape (...).
        """
        # Shape: (..., num_objects, num_timesteps).
        collision_vels_at = jnp.where(collisions_per_agent, future_speeds, jnp.inf)
        if future_speeds_valid is not None:
            collision_vels_at = jnp.where(
                future_speeds_valid, collision_vels_at, jnp.inf
            )
        # In the rare case of multiple collisions, take the closest one.
        # Shape: (..., num_timesteps).
        collision_vels_t = jnp.min(collision_vels_at, axis=-2)
        mask_t = jnp.isfinite(collision_vels_t)
        cumsum_t = jnp.cumsum(jnp.where(mask_t, collision_vels_t, 0), axis=-1)
        # Shape: (..., num_timesteps).
        collision_vels_cum = jnp.where(mask_t, cumsum_t, jnp.inf)
        # Shape: (...)
        lead_vel = jnp.min(collision_vels_cum, axis=-1)

        return jnp.where(jnp.isfinite(lead_vel), lead_vel, _DEFAULT_LEAD_VELOCITY)

    def _compute_lead_distance(
            self,
            agent_future: jax.Array,
            collision_indicator: jax.Array,
            agent_future_valid: Optional[jax.Array] = None,
            current_position: Optional[jax.Array] = None,
            use_arclength=False,
    ) -> jax.Array:
        """Computes the distance between the agent and the nearest collision.

        Args:
          agent_future: Agent's future positions {x, y, z} of shape (...,
            num_timesteps, 3).
          collision_indicator: Collision indications of shape (..., num_timesteps).
          agent_future_valid: Boolean mask for agent's future positions of shape
            (..., num_timesteps).
          current_position: Array of the vehicle's current positions {x, y, z} of
            shape  (..., 1, 3). If None, will use the first element of agent_future
            as the current position.
          use_arclength: Whether to use arclength for computing collisions.
            Arclength is more accurate but is not robust to futures with mixed
            valids.

        Returns:
          An array of distances to the agent's closest collision of shape (...).
        """
        if use_arclength:
            arc_lengths = _compute_arclengths(agent_future, agent_future_valid)
            cum_arc_length = jnp.cumsum(arc_lengths, axis=-1)
        else:
            if current_position is None:
                current_position = agent_future[..., 0:1, :]
            dists = jnp.linalg.norm(current_position - agent_future, axis=-1)
            cum_arc_length = dists

        # Shape: (..., num_timesteps).
        dists_to_collision = jnp.where(collision_indicator, cum_arc_length, jnp.inf)
        # Shape: (...).
        lead_dist = jnp.min(dists_to_collision, axis=-1)
        return jnp.where(jnp.isfinite(lead_dist), lead_dist, _DEFAULT_LEAD_DISTANCE)


def _find_reference_traj_from_log_traj(
        xyz: jax.Array, traj: datatypes.Trajectory, num_pts: int
) -> datatypes.Trajectory:
    """Finds sub-trajectory for given position xyz.

    Args:
      xyz: Position of the agent of shape (..., 3).
      traj: Full trajectory for a given agent (..., T).
      num_pts: The number of waypoints to retrieve for future Trajectory.

    Returns:
      Trajectory with (..., T_out=num_pts).
    """
    prefix_shape = traj.shape[:-1]
    chex.assert_shape(xyz, prefix_shape + (3,))

    def find_reference_helper(
            local_xyz: jax.Array, local_traj: datatypes.Trajectory
    ) -> datatypes.Trajectory:
        chex.assert_shape(local_xyz, (3,))
        chex.assert_equal(len(local_traj.shape), 1)
        dists = jnp.linalg.norm(local_xyz[jnp.newaxis, :] - local_traj.xyz, axis=-1)
        dists = jnp.where(local_traj.valid, dists, jnp.inf)
        top_idx = jnp.argmin(dists, axis=-1)
        return datatypes.dynamic_slice(local_traj, top_idx, num_pts, axis=-1)

    find_reference_fn = find_reference_helper
    for _ in range(jnp.ndim(xyz) - 1):
        find_reference_fn = jax.vmap(find_reference_fn)

    return find_reference_fn(xyz, traj)


def _project_to_a_trajectory(
        xy: jax.Array, traj: datatypes.Trajectory, extrapolate_traj: bool = False
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Projects points on to a Trajectory.

    Args:
      xy: Current xy position of the agent of shape (..., 1, 2).
      traj: Full trajectory of the agent of shape (..., num_timesteps).
      extrapolate_traj: Whether to extrapolate a projection beyond the final
        waypoint in direction of the final waypoint.

    Returns:
      Updated xy with shape (..., 1, 2).
      Updated yaw with shape (..., 1).
      Boolean indicating whether the agent reached the last waypoint
        with shape (..., 1).
    """

    def project_point_to_traj(
            xy: jax.Array, traj: datatypes.Trajectory
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        chex.assert_shape(xy, (1, 2))
        chex.assert_equal(len(traj.shape), 1)

        # Shape: (num_timesteps).
        dist = jnp.where(
            traj.valid, jnp.linalg.norm(traj.xy - xy, axis=-1), jnp.inf
        )
        # Shape: (1).
        idx = jnp.argmin(dist, axis=-1)
        # Shape: (2).
        src_xy = traj.xy[idx]
        src_yaw = traj.yaw[idx]
        src_dir = jnp.stack([jnp.cos(src_yaw), jnp.sin(src_yaw)], axis=-1)

        last_valid_idx = jnp.where(traj.valid, jnp.arange(traj.shape[0]), 0)
        last_valid_idx = jnp.argmax(last_valid_idx, axis=-1)
        last_point = traj.xy[last_valid_idx, :]
        reached_last_point = (
                jnp.linalg.norm(last_point - src_xy, axis=-1)
                < _REACHED_END_OF_TRAJECTORY_THRESHOLD
        )
        # Secondary detection: If a vehicle strays too far from the traj,
        # also mark it as reaching the end.
        reached_last_point = jnp.logical_or(
            reached_last_point, dist[idx] > _DISTANCE_TO_REF_THRESHOLD
        )

        # Prevent points from extrapolating beyond traj.
        if not extrapolate_traj:
            src_dir = jnp.where(reached_last_point, jnp.zeros_like(src_dir), src_dir)

        # Shape: (2).
        proj_xy = src_dir * jnp.inner(src_dir, xy - src_xy) + src_xy
        return (
            proj_xy[jnp.newaxis, :],
            src_yaw[jnp.newaxis],
            reached_last_point[jnp.newaxis],
        )

    proj_func = project_point_to_traj

    for _ in range(xy.ndim - 2):
        proj_func = jax.vmap(proj_func)

    return proj_func(xy, traj)


def _compute_arclengths(
        waypoints: jax.Array, valid: Optional[jax.Array] = None
) -> jax.Array:
    """Helper function for computing the arclengths of a series of waypoints."""
    # Shape: (..., num_timesteps, dim).
    arc_lengths = jnp.linalg.norm(
        waypoints[..., :-1, :] - waypoints[..., 1:, :], axis=-1
    )
    first_values = jnp.zeros_like(waypoints[..., :1, 0])
    if valid is not None:
        chex.assert_equal_shape([waypoints[..., 0], valid])
        arc_lengths_valid = valid[..., :-1] & valid[..., 1:]
        arc_lengths = jnp.where(arc_lengths_valid, arc_lengths, jnp.inf)
        first_values = jnp.where(valid[..., :1], first_values, jnp.inf)
    # Shape: (..., num_timesteps).
    return jnp.concatenate([first_values, arc_lengths], axis=-1)


def _add_headway_waypoints(
        traj: datatypes.Trajectory, distance: float = 2.0, num_points: int = 10
) -> datatypes.Trajectory:
    """Adds additional waypoints after a trajectory for collision detection.

    This function adds `num_points` additional waypoints, spaced evenly over
    `distance`, to the end of a series of waypoints following the same heading as
    the final timestep.
    The use case is to add additional waypoints for collision detection, since
    we only check for collisions along future waypoints. This is especially
    useful for vehicles that move slowly or are parked in the logs and do not
    have many future logged waypoints.

    Args:
      traj: A trajectory of waypoints of shape (...., num_timesteps).
      distance: Distance over which to add points.
      num_points: Number of points to add.

    Returns:
      A trajectory of shape (..., num_timesteps+N) containing augmented waypoints,
        where N is num_points.
    """
    # Shape: (..., 2).
    final_xyz = traj.xy[..., -1:, :]
    final_yaw = traj.yaw[..., -1:]
    # Shape: (..., 2).
    final_dir = jnp.stack([jnp.cos(final_yaw), jnp.sin(final_yaw)], axis=-1)
    spacings = jnp.linspace(0, distance, num=num_points + 1, endpoint=True)
    spacings = jnp.reshape(
        spacings, (1,) * len(final_dir.shape[:-2]) + (num_points + 1, 1)
    )
    new_xy_points = ((final_dir * spacings) + final_xyz)[..., 1:, :]
    new_xy_points = jnp.concatenate([traj.xy, new_xy_points], axis=-2)

    def _repeat_last(item: jax.Array) -> jax.Array:
        tile_shape = ((1,) * len(item.shape[:-1])) + (num_points,)
        new_item = jnp.tile(item[..., -1:], tile_shape)
        return jnp.concatenate([item, new_item], axis=-1)

    new_traj = jax.tree_map(_repeat_last, traj)
    new_traj = new_traj.replace(
        x=new_xy_points[..., 0],
        y=new_xy_points[..., 1],
    )
    new_traj.validate()
    return new_traj
