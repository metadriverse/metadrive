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

"""All customized data structures for Waymax.

Note the shape of a data classes here is defined as the most common prefix shape
among all attributes.
The validate function is implemented separately instead of as __post_init__, to
have better support with jax utils.
"""

import enum
from typing import Any, Optional

import chex
import jax
import jax.numpy as jnp

from metadrive.policy.waymax_idm.datatypes import array
from metadrive.policy.waymax_idm.datatypes import object_state
from metadrive.policy.waymax_idm.datatypes import operations
from metadrive.policy.waymax_idm.datatypes import roadgraph
from metadrive.policy.waymax_idm.datatypes import route
from metadrive.policy.waymax_idm.datatypes import traffic_lights

ArrayLike = jax.typing.ArrayLike
PyTree = array.PyTree


class ObjectType(enum.Enum):
    """Types of objects that can be controlled by Waymax."""

    SDC = 'SDC'  # The ego-vehicle.
    MODELED = 'MODELED'  # tracks_to_predict objects defined in WOMD.
    VALID = 'VALID'  # All valid objects in the scene.
    NON_SDC = 'NON_SDC'  # Any objects expect the SDC (ego-vehicle).


@chex.dataclass
class SimulatorState:
    """A dataclass holding the simulator state, all data in global coordinates.
  
    Attributes:
      sim_trajectory: Simulated trajectory for all objects of shape (...,
        num_objects, num_timesteps). The number of timesteps is the same as in the
        log, but future trajectory points that have not been simulated will be
        marked invalid.
      log_trajectory: Logged trajectory for all objects of shape (...,
        num_objects, num_timesteps).
      log_traffic_light: Logged traffic light information for the entire run
        segment of shape (..., num_traffic_lights, num_timesteps).
      object_metadata: Metadata for all objects of shape (..., num_objects).
      timestep: The current simulation timestep index of shape (...). Note that
        sim_trajectory at `timestep` is the last executed step by the simulator.
      sdc_paths: Paths for SDC, representing where the SDC can drive of shape
        (..., num_paths, num_points_per_path).
      roadgraph_points: A optional RoadgraphPoints holding subsampled roadgraph
        points of shape (..., num_points).
    """

    sim_trajectory: object_state.Trajectory
    # TODO Support testset, i.e. no log_trajectory for all steps.
    log_trajectory: object_state.Trajectory
    log_traffic_light: traffic_lights.TrafficLights
    object_metadata: object_state.ObjectMetadata
    timestep: jax.typing.ArrayLike
    sdc_paths: Optional[route.Paths] = None
    roadgraph_points: Optional[roadgraph.RoadgraphPoints] = None

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape is defined as the most common prefix shape of all attributes."""
        # Here, shape is equivalent to batch dimensions, and can be ().
        return self.object_metadata.shape[:-1]

    @property
    def batch_dims(self) -> tuple[int, ...]:
        """Batch dimensions."""
        return self.shape

    @property
    def num_objects(self) -> int:
        """The number of objects included in this trajectory per example."""
        return self.object_metadata.num_objects

    @property
    def is_done(self) -> bool:
        """Returns whether the simulation is at the end of the logged history."""
        return jnp.array(  # pytype: disable=bad-return-type  # jnp-type
            (self.timestep + 1) >= self.log_trajectory.num_timesteps, bool
        )

    @property
    def remaining_timesteps(self) -> int:
        """Returns the number of remaining timesteps in the episode."""
        return jnp.array(self.log_trajectory.num_timesteps - self.timestep - 1,
                         int)  # pytype: disable=bad-return-type  # jnp-type

    @property
    def current_sim_trajectory(self) -> object_state.Trajectory:
        """Returns the trajectory corresponding to the current sim state."""
        return operations.dynamic_slice(
            self.sim_trajectory, self.timestep, 1, axis=-1
        )

    def __eq__(self, other: Any) -> bool:
        return operations.compare_all_leaf_nodes(self, other)

    @property
    def current_log_trajectory(self) -> object_state.Trajectory:
        """Returns the trajectory corresponding to the current sim state."""
        return operations.dynamic_slice(
            self.log_trajectory, self.timestep, 1, axis=-1
        )

    def validate(self):
        """Validates shape and type."""
        data = [
            self.sim_trajectory,
            self.log_trajectory,
            self.log_traffic_light,
            self.object_metadata,
            self.timestep,
        ]
        if self.roadgraph_points is not None:
            data.append(self.roadgraph_points)
        chex.assert_equal_shape_prefix(data, len(self.shape))


def update_state_by_log(
        state: SimulatorState, num_steps: int
) -> SimulatorState:
    """Advances SimulatorState by num_steps using logged data."""
    # TODO jax runtime check num_steps > state.remaining_timesteps
    return state.replace(
        timestep=state.timestep + num_steps,
        sim_trajectory=operations.update_by_slice_in_dim(
            inputs=state.sim_trajectory,
            updates=state.log_trajectory,
            inputs_start_idx=state.timestep + 1,
            slice_size=num_steps,
            axis=-1,
        ),
    )


def get_control_mask(
        metadata: object_state.ObjectMetadata, obj_type: ObjectType
) -> jax.Array:
    """Returns binary mask for selected object type.
  
    Args:
      metadata: An ObjectMetadata, having shape (..., num_objects).
      obj_type: Represents which type of objects should be selected.
  
    Returns:
      A binary mask with shape (..., num_objects).
    """

    if obj_type == ObjectType.SDC:
        is_controlled = metadata.is_sdc
    elif obj_type == ObjectType.MODELED:
        is_controlled = metadata.is_modeled
    elif obj_type == ObjectType.VALID:
        is_controlled = metadata.is_valid
    else:
        raise ValueError(f'Invalid ObjectType {obj_type}')
    return is_controlled
