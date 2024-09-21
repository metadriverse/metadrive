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

"""Common data operations made to work on PyTree data structures."""

import functools
from typing import Callable, Optional, Sequence, TypeVar, Union

import chex
import jax
import numpy as onp
from jax import numpy as jnp

from metadrive.policy.waymax_idm.datatypes import array

ArrayLike = jax.typing.ArrayLike
F = TypeVar('F', bound=Callable)
PyTree = array.PyTree
Axis = Union[None, int, Sequence[int]]


def update_by_mask(inputs: PyTree, updates: PyTree, mask: jax.Array) -> PyTree:
    """Updates inputs by updates (with same struct) for masked values.

    Uses `jax.tree_util` to update each field as such:
      inputs.field = jnp.where(mask, updates.field, inputs.field)

    Args:
      inputs: Represents the data to be updated and returned.
      updates: Represents the data that is passed in to update all fields. It is
        assumed that the `updates` and the `inputs` have the same structure. These
        fields must be broadcastable to the same shape as `inputs` after applying
        the mask.
      mask: Mask that represents which elements of the `inputs` array tree to
        update with the `updates` field. Shape must be broadcastable to the leaf
        nodes of inputs and updates.

    Returns:
      Updated tree array of the same structure as `inputs` which has had all its
        fields updated with the corresponding fields in `updates` conditioned on
        whether `mask` requests it.
    """
    return jax.tree_util.tree_map(
        lambda x, y: jnp.where(mask, x, y), updates, inputs
    )


def _vmap_over_batch_dims(func: F, batch_element: ArrayLike) -> F:
    """Apply vmap to a function based on the number of batch dimensions."""
    if hasattr(batch_element, 'shape'):
        ndims = len(batch_element.shape)
    else:
        return func

    for _ in range(ndims):
        func = jax.vmap(func)
    return func


@jax.named_scope('dynamic_slice')
def dynamic_slice(
        inputs: PyTree, start_index: ArrayLike, slice_size: int, axis: int = 0
) -> PyTree:
    """Wraps dynamic_slice_in_dim on a PyTree.

    Args:
      inputs: Object where each element in its tree is to be sliced.
      start_index: Start index of slicing function.
      slice_size: Size of slice applied to `inputs` starting from `start_index` to
        `start_index` + `slice_size`.
      axis: Axis of each array to perform the slicing in.

    Returns:
      Array tree which has been sliced along an axis which maintains the same
        rank as `inputs`.
    """

    def _slice(x, start_index):
        return jax.lax.dynamic_slice_in_dim(x, start_index, slice_size, axis)

    map_fn = _vmap_over_batch_dims(_slice, start_index)
    map_fn = functools.partial(map_fn, start_index=start_index)
    return jax.tree_util.tree_map(map_fn, inputs)


@jax.named_scope('dynamic_index')
def dynamic_index(
        inputs: PyTree, index: ArrayLike, axis: int = 0, keepdims: bool = True
) -> PyTree:
    """Wraps dynamic_index_in_dim on a PyTree.

    Args:
      inputs: Object where each element in it is to be indexed.
      index: Element number to index into each array in the tree.
      axis: Axis of each array to perform the indexing in.
      keepdims: Whether or not to keep the same rank as `inputs`. If this is set
        to `True` then the return value will be such that `.shape[axis]` == 1.

    Returns:
      Array tree where every element of the tree has been indexed at the same
        specified axis.
    """

    def _index(x, index):
        return jax.lax.dynamic_index_in_dim(x, index, axis, keepdims)

    map_fn = _vmap_over_batch_dims(_index, index)
    map_fn = functools.partial(map_fn, index=index)
    return jax.tree_util.tree_map(map_fn, inputs)


@jax.named_scope('update_by_slice_in_dim')
def update_by_slice_in_dim(
        inputs: PyTree,
        updates: PyTree,
        inputs_start_idx: ArrayLike,
        updates_start_idx: Optional[ArrayLike] = None,
        slice_size: Optional[int] = None,
        axis: int = 0,
) -> PyTree:
    """Updates a slice in inputs using slices from updates.

    It replaces inputs[inputs_start_idx:inputs_start_idx+slice_size] by
    updates[updates_start_idx:updates_start_idx+slice_size] for specified axis
    only.

    If updates_start_idx and slice_size are both None, the behavior of this
    function will default to dynamic_update_slice_in_dim.

    Args:
      inputs: Represents the data to be updated and returned.
      updates: Represents the data that is passed in to update all fields. It is
        assumed that the `updates` and the `inputs` have the same structure.
      inputs_start_idx: starting index in inputs.
      updates_start_idx: starting index in updates.
      slice_size: the slice size used for update. If slice size is None, uses the
        entire array and updates_start_idx will be ignored.
      axis: Axis of each array to perform the slicing in.

    Returns:
      A PyTree with same structure as inputs.
    """

    # TODO add proper boundary runtime check, `dynamic_slice_in_dim`
    # does not do a boundary check.

    if updates_start_idx is None:
        updates_start_idx = inputs_start_idx

    def _update(source, update, inputs_start_idx, updates_start_idx):
        if slice_size is not None:
            update = jax.lax.dynamic_slice_in_dim(
                update, updates_start_idx, slice_size, axis
            )
        return jax.lax.dynamic_update_slice_in_dim(
            source, update, inputs_start_idx, axis
        )

    map_fn = _vmap_over_batch_dims(_update, inputs_start_idx)
    map_fn = functools.partial(
        map_fn,
        inputs_start_idx=inputs_start_idx,
        updates_start_idx=updates_start_idx,
    )
    return jax.tree_util.tree_map(map_fn, inputs, updates)


# Add an alias for dynamic_update_slice_in_dim to match the JAX api.
# pylint:disable=g-long-lambda
dynamic_update_slice_in_dim = lambda inp, dst, start, axis: (
    update_by_slice_in_dim(inp, dst, start, axis=axis)
)


def compare_all_leaf_nodes(  # pytype: disable=annotation-type-mismatch  # jnp-type
        first: PyTree,
        second: PyTree,
        compare_func: Callable[[jax.Array, jax.Array], bool] = jnp.array_equal,
) -> bool:
    """Checks if all leaf nodes are consistent via compare_func.

    The default behaviour (with compare_func as jnp.array_equal) is to
    compare if two PyTree are equal (i.e. all leaf nodes of array are equal).
    One can also use compare_func as jnp.allclose to give some tolerance
    for numerical difference.

    Args:
      first: A PyTree for comparison.
      second: A PyTree for comparison.
      compare_func: A function comparing nodes between two input arrays.

    Returns:
      True if inputs PyTrees are consistent by func.
    """
    if not isinstance(second, type(first)):
        return False
    flags = jax.tree_util.tree_map(compare_func, first, second)
    return jax.tree_util.tree_all(flags)


@jax.named_scope('select_by_onehot')
def select_by_onehot(
        data: PyTree, selection: jax.Array, keepdims: bool = False
) -> PyTree:
    """Selects data using a one-hot vector.

    Args:
      data: A pytree with leaf shapes (..., N, ...).
      selection: A one-hot vector with shape (..., N).
      keepdims: Whether to leave a (1,) dimension on the selected axis.

    Returns:
      A pytree with leaf shapes (..., 1) if keepdims=True.
      A pytree with leaf shapes (..., ) if keepdims=False.
    """
    select_dims = len(selection.shape)

    def _select(x: jax.Array) -> jax.Array:
        chex.assert_equal_shape_prefix([x, selection], selection.ndim)
        xdims = len(x.shape)
        if xdims > select_dims:
            diff = xdims - select_dims
            reshaped_selection = jnp.expand_dims(
                selection, axis=-(onp.arange(diff) + 1)
            )
        else:
            reshaped_selection = selection
        # Since reshaped_selection is one-hot, use sum to perform the selection.
        return jnp.sum(
            x * reshaped_selection, axis=(selection.ndim - 1), keepdims=keepdims
        ).astype(x.dtype)

    return jax.tree_util.tree_map(_select, data)


def make_invalid_data(data: jax.Array) -> jax.Array:
    """Returns a tensor of invalid values with identical shape to data.

    Invalid values are defined as False for booleans, and -1 for numerical values.

    Args:
      data: Tensor to invalidate.

    Returns:
      A tensor of invalid values of the same shape and dtype as data.
    """
    if data.dtype == jnp.bool_:
        return jnp.zeros_like(data)
    else:
        return -1 * jnp.ones_like(data)


def masked_mean(
        x: jax.Array, valid: jax.Array, axis: Axis = 0, invalid_value: float = -1.0
) -> jax.Array:
    """Calculates the mean of the array removing invalid values.

    Args:
      x: Input to the masked mean function.
      valid: Boolean array with the same shape as x which indicates which values
        should be included in the mean.
      axis: Axis to reduce along.
      invalid_value: If there is no valid fields, the value will be replaced by
        this invalid value.

    Returns:
      Array representing the mean of the array of all valid values.
    """
    num_valid = jnp.sum(valid.astype(jnp.float32), axis=axis)
    masked_sum = jnp.sum(x, where=valid, axis=axis)
    mean = masked_sum / num_valid
    mean = jnp.where(num_valid == 0, invalid_value, mean)
    # Broadcast back to the original shape.
    return jnp.broadcast_to(mean[..., None], x.shape)
