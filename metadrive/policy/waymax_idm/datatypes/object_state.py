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

"""Data structures for trajectory and metadata information for scene objects."""
import enum
from collections.abc import Sequence
from typing import Any

import chex
import jax
from jax import numpy as jnp

from metadrive.policy.waymax_idm import geometry
from metadrive.policy.waymax_idm.datatypes import operations

_INVALID_FLOAT_VALUE = -1.0
_INVALID_INT_VALUE = -1


class ObjectTypeIds(enum.IntEnum):
    """Ids for different map elements to be mapped into a tensor.

    These integers represent the ID of these specific types as defined in:
      https://waymo.com/open/data/motion/tfexample.
    """

    UNSET = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    CYCLIST = 3
    OTHER = 4


@chex.dataclass
class ObjectMetadata:
    """Time-independent object metadata.

    All arrays are of shape (..., num_objects).

    Attributes:
      ids: A unique integer id for each object which is consistent over time of
        data type int32.
      object_types: An integer representing each different class of object
        (Unset=0, Vehicle=1, Pedestrian=2, Cyclist=3, Other=4) of data type int32.
        This definition is from Waymo Open Motion Dataset (WOMD).
      is_sdc: Binary mask of data type bool representing whether an object
        represents the sdc or some other object.
      is_modeled: Whether a specific object is one designated by WOMD to be
        predicted of data type bool.
      is_valid: Whether an object is valid at any part of the run segment of data
        type bool.
      objects_of_interest: A vector of type bool to indicate which objects in the
        scene corresponding to the first dimension of the object tensors have
        interactive behavior. Up to 2 objects will be selected. The objects in
        this list form an interactive group.
      is_controlled: Whether an object will be controlled by external agents in an
        environment.
    """

    ids: jax.Array
    object_types: jax.Array
    is_sdc: jax.Array
    is_modeled: jax.Array
    is_valid: jax.Array
    objects_of_interest: jax.Array
    is_controlled: jax.Array

    @property
    def shape(self) -> tuple[int, ...]:
        """The Array shape of the metadata."""
        return self.ids.shape

    @property
    def num_objects(self) -> int:
        """The number of objects in metadata."""
        return self.shape[-1]

    def __eq__(self, other: Any) -> bool:
        return operations.compare_all_leaf_nodes(self, other)

    def validate(self):
        """Validates shape and type."""
        chex.assert_equal_shape([
            self.ids,
            self.object_types,
            self.is_sdc,
            self.is_modeled,
            self.is_valid,
            self.objects_of_interest,
            self.is_controlled,
        ])
        chex.assert_type(
            [
                self.ids,
                self.object_types,
                self.is_sdc,
                self.is_modeled,
                self.is_valid,
                self.objects_of_interest,
                self.is_controlled,
            ],
            [
                jnp.int32,
                jnp.int32,
                jnp.bool_,
                jnp.bool_,
                jnp.bool_,
                jnp.bool_,
                jnp.bool_,
            ],
        )
        # TODO runtime checks only one sdc exist for self.is_sdc


@chex.dataclass
class Trajectory:
    """Data structure representing a trajectory.

    The shapes of all objects are of shape (..., num_objects, num_timesteps).

    Attributes:
      x: The x coordinate of each object at each time step of data type float32.
      y: The y coordinate of each object at each time step of data type float32.
      z: The z coordinate of each object at each time step of data type float32.
      vel_x: The x component of the object velocity at each time step of data type
        float32.
      vel_y: The y component of the object velocity at each time step of data type
        float32.
      yaw: Counter-clockwise yaw in top-down view (rotation about the Z axis from
        a unit X vector to the object direction vector) of shape of data type
        float32.
      valid: Validity bit for all object at all times steps of data type bool.
      timestamp_micros: A timestamp in microseconds for each time step of data
        type int32.
      length: The length of each object at each time step of data type float32.
        Note for each object, its length is fixed for all time steps.
      width: The width of each object at each time step of data type float32. Note
        for each object, its width is fixed for all time steps.
      height: The height of each object at each time step of data type float32.
        Note for each object, its height is fixed for all time steps.
    """

    x: jax.Array
    y: jax.Array
    z: jax.Array
    vel_x: jax.Array
    vel_y: jax.Array
    yaw: jax.Array
    valid: jax.Array
    timestamp_micros: jax.Array
    length: jax.Array
    width: jax.Array
    height: jax.Array

    @property
    def shape(self) -> tuple[int, ...]:
        """The Array shape of this trajectory."""
        return self.x.shape

    @property
    def num_objects(self) -> int:
        """The number of objects included in this trajectory per example."""
        return self.shape[-2]

    @property
    def num_timesteps(self) -> int:
        """The length of this trajectory in time."""
        return self.shape[-1]

    @property
    def xy(self) -> jax.Array:
        """Stacked xy location."""
        return jnp.stack([self.x, self.y], axis=-1)

    @property
    def xyz(self) -> jax.Array:
        """Stacked xyz location."""
        return jnp.stack([self.x, self.y, self.z], axis=-1)

    @property
    def vel_xy(self) -> jax.Array:
        """Stacked xy velocity."""
        return jnp.stack([self.vel_x, self.vel_y], axis=-1)

    @property
    def speed(self) -> jax.Array:
        """Speed on x-y plane."""
        speed = jnp.linalg.norm(self.vel_xy, axis=-1)
        # Make sure those that were originally invalid are still invalid.
        return jnp.where(self.valid, speed, _INVALID_FLOAT_VALUE)

    @property
    def vel_yaw(self) -> jax.Array:
        """Angle of the velocity on x-y plane."""
        vel_yaw = jnp.arctan2(self.vel_y, self.vel_x)
        # Make sure those that were originally invalid are still invalid.
        return jnp.where(self.valid, vel_yaw, _INVALID_FLOAT_VALUE)

    def __eq__(self, other: Any) -> bool:
        return operations.compare_all_leaf_nodes(self, other)

    def stack_fields(self, field_names: Sequence[str]) -> jax.Array:
        """Returns a concatenated version of a set of field names for Trajectory."""
        return jnp.stack(
            [getattr(self, field_name) for field_name in field_names], axis=-1
        )

    @property
    def bbox_corners(self) -> jax.Array:
        """Corners of the bounding box spanning the object's shape.

        Returns:
          Box corners' (x, y) coordinates spanning the object of shape
            (..., num_objects, num_timesteps, 4, 2). The 4 corners start from the
            objects' front right corner and go counter-clockwise.
        """
        traj_5dof = self.stack_fields(['x', 'y', 'length', 'width', 'yaw'])
        return geometry.corners_from_bboxes(traj_5dof)

    @classmethod
    def zeros(cls, shape: Sequence[int]) -> 'Trajectory':
        """Creates a Trajectory containing zeros of the specified shape."""
        return cls(
            x=jnp.zeros(shape, jnp.float32),
            y=jnp.zeros(shape, jnp.float32),
            z=jnp.zeros(shape, jnp.float32),
            vel_x=jnp.zeros(shape, jnp.float32),
            vel_y=jnp.zeros(shape, jnp.float32),
            yaw=jnp.zeros(shape, jnp.float32),
            valid=jnp.zeros(shape, jnp.bool_),
            length=jnp.zeros(shape, jnp.float32),
            width=jnp.zeros(shape, jnp.float32),
            height=jnp.zeros(shape, jnp.float32),
            timestamp_micros=jnp.zeros(shape, jnp.int32),
        )

    def validate(self):
        """Validates shape and type."""
        chex.assert_equal_shape([
            self.x,
            self.y,
            self.z,
            self.vel_x,
            self.vel_y,
            self.yaw,
            self.valid,
            self.timestamp_micros,
            self.length,
            self.width,
            self.height,
        ])
        chex.assert_type(
            [
                self.x,
                self.y,
                self.z,
                self.vel_x,
                self.vel_y,
                self.yaw,
                self.valid,
                self.timestamp_micros,
                self.length,
                self.width,
                self.height,
            ],
            [
                jnp.float32,
                jnp.float32,
                jnp.float32,
                jnp.float32,
                jnp.float32,
                jnp.float32,
                jnp.bool_,
                jnp.int32,
                jnp.float32,
                jnp.float32,
                jnp.float32,
            ],
        )


def fill_invalid_trajectory(traj: Trajectory) -> Trajectory:
    """Fills a trajectory with invalid values.

    An invalid value is -1 for numerical fields and False for booleans.

    Args:
      traj: Trajectory to fill.

    Returns:
      A new trajectory with invalid values.
    """

    def _fill_fn(x: jax.Array) -> jax.Array:
        if x.dtype in [jnp.int64, jnp.int32, jnp.int16, jnp.int8]:
            return jnp.ones_like(x) * _INVALID_INT_VALUE
        elif x.dtype in [jnp.float32, jnp.float64, jnp.float16]:
            return jnp.ones_like(x) * _INVALID_FLOAT_VALUE
        elif x.dtype == jnp.bool_:
            return jnp.zeros_like(x).astype(jnp.bool_)
        else:
            raise ValueError('Unsupport dtype: %s' % x.dtype)

    return jax.tree_util.tree_map(_fill_fn, traj)
