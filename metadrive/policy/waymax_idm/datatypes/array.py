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

"""Array data structures."""
from typing import Any, Type

import chex
import jax
from jax import numpy as jnp

PyTree = Any


@chex.dataclass
class MaskedArray:
    """A dataclass holding values and a validity mask.

    Attributes:
      value: A valid.shape + (...) array of values.
      valid: A boolean validity mask.
    """

    value: jax.Array
    valid: jax.Array

    @property
    def shape(self) -> tuple[int, ...]:
        """The Array prefix shape of the value."""
        return self.valid.shape

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, MaskedArray):
            return False
        flags = jax.tree_util.tree_map(jnp.array_equal, self, other)
        return jax.tree_util.tree_all(flags)

    def validate(self) -> None:
        """Validates shape and type."""
        chex.assert_equal_shape_prefix([self.value, self.valid], self.valid.ndim)
        chex.assert_type(self.valid, jnp.bool_)

    # TODO: replace type annotation with typing.Self when Python 3.11
    # is available.
    @classmethod
    def create_and_validate(
            cls: Type['MaskedArray'], value: jax.Array, valid: jax.Array
    ) -> 'MaskedArray':
        """Creates an instance of the class."""
        out = cls(value=value, valid=valid)
        out.validate()
        return out

    def masked_value(self, fill_value: Any = 0.0) -> jax.Array:
        """Returns `value` where invalid elements are replaced with `fill_value`.

        Args:
          fill_value: Value with which to replace invalid elements. Must be castable
            to the dtype of `value`.

        Returns:
          `value` where invalid elements are replaced with `fill_value`.
        """
        fill_array = jnp.full_like(self.value, fill_value, dtype=self.value.dtype)
        return jnp.where(self.valid, self.value, fill_array)
