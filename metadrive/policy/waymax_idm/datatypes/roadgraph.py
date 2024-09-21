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

"""Roadgraph based datastructures for Waymax based on WOMD."""
import enum
from typing import Any

import chex
import jax
from jax import numpy as jnp

from metadrive.policy.waymax_idm.datatypes import array
from metadrive.policy.waymax_idm.datatypes import operations

PyTree = array.PyTree


class MapElementIds(enum.IntEnum):
    """Ids for different map elements to be mapped into a tensor.

    These integers represent the ID of these specific types as defined in:
      https://waymo.com/open/data/motion/tfexample.
    """

    LANE_UNDEFINED = 0
    LANE_FREEWAY = 1
    LANE_SURFACE_STREET = 2
    LANE_BIKE_LANE = 3
    # Original definition skips 4.
    ROAD_LINE_UNKNOWN = 5
    ROAD_LINE_BROKEN_SINGLE_WHITE = 6
    ROAD_LINE_SOLID_SINGLE_WHITE = 7
    ROAD_LINE_SOLID_DOUBLE_WHITE = 8
    ROAD_LINE_BROKEN_SINGLE_YELLOW = 9
    ROAD_LINE_BROKEN_DOUBLE_YELLOW = 10
    ROAD_LINE_SOLID_SINGLE_YELLOW = 11
    ROAD_LINE_SOLID_DOUBLE_YELLOW = 12
    ROAD_LINE_PASSING_DOUBLE_YELLOW = 13
    ROAD_EDGE_UNKNOWN = 14
    ROAD_EDGE_BOUNDARY = 15
    ROAD_EDGE_MEDIAN = 16
    STOP_SIGN = 17
    CROSSWALK = 18
    SPEED_BUMP = 19
    UNKNOWN = -1


@chex.dataclass
class RoadgraphPoints:
    """Data structure representing roadgraph points.

    It holds the coordinates of the sampled map data points.  Note all attributes
    have shape (..., num_points), where num_points is the
    number of road graph points.

    Attributes:
      x: X coordinate of positions of the sampled map data points of dtype
        float32.
      y: Y coordinate of positions of the sampled map data points of dtype
        float32.
      z: Z coordinate of positions of the sampled map data points of dtype
        float32.
      dir_x: X coordinate of a unit direction vector for each map feature sample
        point of dtype float32.
      dir_y: Y coordinate of a unit direction vector for each map feature sample
        point of dtype float32.
      dir_z: Z coordinate of a unit direction vector for each map feature sample
        point of dtype float32.
      types: A unique integer for each combination of map feature type and
        properties of dtype int32. See the table in the Value Range column of
        website: https://waymo.com/open/data/motion/tfexample.
      ids: A unique Integer ID for the vector map feature each sample is from of
        dtype int32.
      valid: A valid flag for each map sample point of dtype bool.
    """

    x: jax.Array
    y: jax.Array
    z: jax.Array
    dir_x: jax.Array
    dir_y: jax.Array
    dir_z: jax.Array
    types: jax.Array
    ids: jax.Array
    valid: jax.Array

    @property
    def shape(self) -> tuple[int, ...]:
        """The Array shape of the RoadgraphPoints."""
        return self.x.shape

    @property
    def num_points(self) -> int:
        """The number of points included in this RoadgraphPoints per example."""
        return self.shape[-1]

    @property
    def xy(self) -> jax.Array:
        """Stacked xy location for all points."""
        return jnp.stack([self.x, self.y], axis=-1)

    @property
    def xyz(self) -> jax.Array:
        """Stacked xyz location for all points."""
        return jnp.stack([self.x, self.y, self.z], axis=-1)

    @property
    def dir_xy(self) -> jax.Array:
        """Stacked xy direction for all points."""
        return jnp.stack([self.dir_x, self.dir_y], axis=-1)

    @property
    def dir_xyz(self) -> jax.Array:
        """Stacked xy direction for all points."""
        return jnp.stack([self.dir_x, self.dir_y, self.dir_z], axis=-1)

    def __eq__(self, other: Any) -> bool:
        return operations.compare_all_leaf_nodes(self, other)

    def validate(self):
        """Validates shape and type."""
        chex.assert_equal_shape([
            self.x,
            self.y,
            self.z,
            self.dir_x,
            self.dir_y,
            self.dir_z,
            self.types,
            self.ids,
            self.valid,
        ])
        chex.assert_type(
            [
                self.x,
                self.y,
                self.z,
                self.dir_x,
                self.dir_y,
                self.dir_z,
                self.types,
                self.ids,
                self.valid,
            ],
            [
                jnp.float32,
                jnp.float32,
                jnp.float32,
                jnp.float32,
                jnp.float32,
                jnp.float32,
                jnp.int32,
                jnp.int32,
                jnp.bool_,
            ],
        )


@jax.named_scope('filter_topk_roadgraph_points')
def filter_topk_roadgraph_points(
        roadgraph: RoadgraphPoints, reference_points: jax.Array, topk: int
) -> RoadgraphPoints:
    """Returns the topk closest roadgraph points to a reference point.

    If `topk` is larger than the number of points, exception will be raised.

    Args:
      roadgraph: Roadgraph information to filter, (..., num_points).
      reference_points: A tensor of shape (..., 2) - the reference point used to
        measure distance.
      topk: Number of points to keep.

    Returns:
      Roadgraph data structure that has been filtered to only contain the `topk`
        closest points to a reference point.
    """
    chex.assert_equal_shape_prefix(
        [roadgraph, reference_points], reference_points.ndim - 1
    )
    chex.assert_equal(len(roadgraph.shape), reference_points.ndim)
    chex.assert_equal(reference_points.shape[-1], 2)

    if topk > roadgraph.num_points:
        raise NotImplementedError('Not enough points in roadgraph.')
    elif topk < roadgraph.num_points:
        distances = jnp.linalg.norm(
            reference_points[..., jnp.newaxis, :] - roadgraph.xy, axis=-1
        )
        valid_distances = jnp.where(roadgraph.valid, distances, float('inf'))
        _, top_idx = jax.lax.top_k(-valid_distances, topk)

        stacked = jnp.stack(
            [
                roadgraph.x,
                roadgraph.y,
                roadgraph.z,
                roadgraph.dir_x,
                roadgraph.dir_y,
                roadgraph.dir_z,
                roadgraph.types,
                roadgraph.ids,
                roadgraph.valid,
            ],
            axis=-1,
            dtype=jnp.float32,
        )
        filtered = jnp.take_along_axis(stacked, top_idx[..., None], axis=-2)

        return RoadgraphPoints(
            x=filtered[..., 0],
            y=filtered[..., 1],
            z=filtered[..., 2],
            dir_x=filtered[..., 3],
            dir_y=filtered[..., 4],
            dir_z=filtered[..., 5],
            types=filtered[..., 6].astype(jnp.int32),
            ids=filtered[..., 7].astype(jnp.int32),
            valid=filtered[..., 8].astype(jnp.bool_),
        )
    else:
        return roadgraph


def is_road_edge(types: jax.Array) -> jax.Array:
    """Determines which map elements in a tensor are road edges.

    Args:
      types: An array of integer values with each integer value representing a
        unique map type. These integers are based on a schema defined in
        https://waymo.com/open/data/motion/tfexample. This is of shape (...,
          num_points).

    Returns:
      A bool array where an element is true if the map element is a road edge.
    """
    return jnp.logical_or(
        types == MapElementIds.ROAD_EDGE_BOUNDARY,
        types == MapElementIds.ROAD_EDGE_MEDIAN,
    )
