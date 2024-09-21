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

"""Custom data structures for traffic light signals in Wayamx."""
import enum
from typing import Any

import chex
import jax
from jax import numpy as jnp

from metadrive.policy.waymax_idm.datatypes import operations


class TrafficLightStates(enum.IntEnum):
    """Represents the integer values for all Traffic Light State values."""

    UNKNOWN = 0
    ARROW_STOP = 1
    ARROW_CAUTION = 2
    ARROW_GO = 3
    STOP = 4
    CAUTION = 5
    GO = 6
    FLASHING_STOP = 7
    FLASHING_CAUTION = 8


@chex.dataclass
class TrafficLights:
    """Data structure representing the dynamic traffic light state information.

    All attributes have shape (..., num_traffic_lights, num_timesteps).

    Attributes:
      x: The X coordinate of the stop light position of dtype float32.
      y: The Y coordinate of the stop light position of dtype float32.
      z: The Z coordinate of the stop light position. This point is at the
        beginning of the lane segment controlled by the traffic signal of dtype
        float32.
      state: The state of each traffic light at each time step of dtype int32. See
        above enum for integer values of all traffic lights states.
      lane_ids: which lane it controls.
      valid: A valid flag for all elements of features traffic_light.XX. If set to
        True, the element is populated with valid data.
    """

    x: jax.Array
    y: jax.Array
    z: jax.Array
    state: jax.Array
    lane_ids: jax.Array
    valid: jax.Array

    @property
    def shape(self) -> tuple[int, ...]:
        """The tensor shape of the traffic lights."""
        return self.x.shape

    @property
    def num_traffic_lights(self) -> int:
        """The number of points included in this traffic light per example."""
        return self.shape[-2]

    @property
    def num_timesteps(self) -> int:
        """The number of timesteps included in this traffic light per example."""
        return self.shape[-1]

    @property
    def xy(self) -> jax.Array:
        """Stacked xy location for all points."""
        return jnp.stack([self.x, self.y], axis=-1)

    def __eq__(self, other: Any) -> bool:
        return operations.compare_all_leaf_nodes(self, other)

    def validate(self):
        """Validates shape and type."""
        chex.assert_equal_shape(
            [self.x, self.y, self.z, self.state, self.lane_ids, self.valid]
        )
        chex.assert_type(
            [self.x, self.y, self.z, self.state, self.lane_ids, self.valid],
            [
                jnp.float32,
                jnp.float32,
                jnp.float32,
                jnp.int32,
                jnp.int32,
                jnp.bool_,
            ],
        )
