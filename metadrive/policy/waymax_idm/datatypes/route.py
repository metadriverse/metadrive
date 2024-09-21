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

"""Customized route paths data structure for Waymax datatypes."""
from typing import Any

import chex
import jax
from jax import numpy as jnp

from metadrive.policy.waymax_idm.datatypes import operations


@chex.dataclass
class Paths:
    """Data structure for a set of paths represented by roadgraph points.

    A `path` consists of a set of roadgraph points (usually along the lane center)
    and represents where an object can legally traverse giving its starting point.

    Attributes:
      x: Path coordinate x, shape is (..., num_paths, num_points_per_path) and
        dtype is float32.
      y: Path coordinate y, shape is (..., num_paths, num_points_per_path) and
        dtype is float32.
      z: Path coordinate z, shape is (..., num_paths, num_points_per_path) and
        dtype is float32.
      ids: IDs representing which lane the points belong to, shape is (...,
        num_paths, num_points_per_path) and dtype is int32.
      valid: Validity flag, shape is (..., num_paths, num_points_per_path) and
        dtype is bool.
      arc_length: Represents the arc length for each point from the starting point
        along the path with shape (..., num_paths, num_points_per_path) and dtype
        is float32.
      on_route: Flag for each path representing whether it is on the road route
        corresponding to the logged trajectory, shape is (..., num_paths, 1) and
        the dtype is bool.
    """

    x: jax.Array
    y: jax.Array
    z: jax.Array
    ids: jax.Array
    valid: jax.Array
    arc_length: jax.Array
    on_route: jax.Array

    @property
    def shape(self) -> tuple[int, ...]:
        """The Array shape of the routes."""
        return self.x.shape

    @property
    def num_points_per_path(self) -> int:
        """The number of points included in the paths per example."""
        return self.shape[-1]

    @property
    def num_paths(self) -> int:
        """The number of paths included in the example."""
        return self.shape[-2]

    @property
    def xy(self) -> jax.Array:
        """Stacked xy location for all points."""
        return jnp.stack([self.x, self.y], axis=-1)

    @property
    def xyz(self) -> jax.Array:
        """Stacked xyz location for all points."""
        return jnp.stack([self.x, self.y, self.z], axis=-1)

    def __eq__(self, other: Any) -> bool:
        return operations.compare_all_leaf_nodes(self, other)

    def validate(self) -> None:
        """Validates shape and type."""
        chex.assert_equal_shape([self.x[..., :1], self.on_route])
        chex.assert_equal_shape([
            self.x,
            self.y,
            self.z,
            self.ids,
            self.valid,
            self.arc_length,
        ])
        chex.assert_type(
            [
                self.x,
                self.y,
                self.z,
                self.ids,
                self.valid,
                self.arc_length,
                self.on_route,
            ],
            [
                jnp.float32,
                jnp.float32,
                jnp.float32,
                jnp.int32,
                jnp.bool_,
                jnp.float32,
                jnp.bool_,
            ],
        )
