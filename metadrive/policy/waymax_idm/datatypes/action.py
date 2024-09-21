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

"""Dataclass definitions for dynamics models."""
from typing import Any

import chex
import jax
import jax.numpy as jnp

from metadrive.policy.waymax_idm.datatypes import operations


# TODO: make Actions inherit from datatypes.MaskedArray.
@chex.dataclass
class Action:
    """Raw actions tensor and validity mask.

    Attributes:
      data: Action array for all agents in the scene of shape (..., num_objects,
        dim).
      valid: Whether or not an action is valid for a given agent of shape (...,
        num_objects, 1).
    """

    data: jax.Array
    valid: jax.Array

    @property
    def shape(self) -> tuple[int, ...]:
        """The tensor shape of actions."""
        return self.valid.shape[:-1]

    def __eq__(self, other: Any) -> bool:
        return operations.compare_all_leaf_nodes(self, other)

    def validate(self) -> None:
        """Validates shape and type."""
        chex.assert_equal_shape_prefix([self.data, self.valid], self.valid.ndim - 1)
        chex.assert_equal(self.valid.shape[-1], 1)
        chex.assert_type([self.valid], [jnp.bool_])


@chex.dataclass
class TrajectoryUpdate:
    """A datastructure holding the controllable parts of a Trajectory.

    The TrajectoryUpdate class contains the fields that a dynamics model is
    allowed to update (pose and velocity). Remaining fields, such as object
    dimensions and timestamps, are computed using common code
    (see `update_state_with_dynamics_trajectory`).

    As all dynamics produce a TrajectoryUpdate (via the `compute_update` method),
    the TrajectoryUpdate serves as an intermediate update format that is common
    to all dynamics models. This allows handling of multiple agents using
    heterogeneous dynamics models.
    """

    x: jax.Array  # (..., num_objects, 1)
    y: jax.Array  # (..., num_objects, 1)
    yaw: jax.Array  # (..., num_objects, 1)
    vel_x: jax.Array  # (..., num_objects, 1)
    vel_y: jax.Array  # (..., num_objects, 1)
    valid: jax.Array  # (..., num_objects, 1)

    @property
    def shape(self) -> tuple[int, ...]:
        """The tensor shape of actions."""
        return self.valid.shape

    def __eq__(self, other: Any) -> bool:
        return operations.compare_all_leaf_nodes(self, other)

    def validate(self) -> None:
        """Validates shape and type."""
        # Verifies that each element has the same dimensions.
        chex.assert_equal_shape(
            [self.x, self.y, self.yaw, self.vel_x, self.vel_y, self.valid],
        )
        chex.assert_type(
            [self.x, self.y, self.yaw, self.vel_x, self.vel_y, self.valid],
            [
                jnp.float32,
                jnp.float32,
                jnp.float32,
                jnp.float32,
                jnp.float32,
                jnp.bool_,
            ],
        )

    def as_action(self) -> Action:
        """Returns this trajectory update as a 5D Action for StateDynamics.

        Returns:
          An action data structure with data of shape (..., 5) containing
          x, y, yaw, vel_x, and vel_y.
        """
        action = jnp.concatenate(
            [self.x, self.y, self.yaw, self.vel_x, self.vel_y], axis=-1
        )
        return Action(data=action, valid=self.valid)
