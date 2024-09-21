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

"""JIT-able util functions on array (data_struct agnostic)."""

import chex
import jax
import jax.numpy as jnp

EPS = 1e-10


def rotation_matrix_2d(angle: jax.Array) -> jax.Array:
    """Returns a 2D rotation matrix.

    If an angle with batched dimensions is given, the result will be
    a batched rotation matrix with the same leading batch dimensions as angle.
    E.g. if angle is of shape (3, 7), the result will be of shape (3, 7, 2, 2)

    Args:
      angle: Angle to rotate by. The positive direction is counter-clockwise.

    Returns:
      A 2x2 2D rotation matrix corresponding to angle.
    """
    # Note typing annotation of jax.Array doesn't error out for inputs of type
    # float/int, thus converting `angle` explicitly to jax.Array.
    angle = jnp.array(angle)
    cos = jnp.cos(angle)
    sin = jnp.sin(angle)
    return jnp.stack([cos, -sin, sin, cos], axis=-1).reshape(angle.shape + (2, 2))


def rotate_points(matrix: jax.Array, points: jax.Array) -> jax.Array:
    """Rotates points by a rotation matrix.

    Args:
      matrix: Matrix specifying the rotation to apply shape (prefix, dof, dof).
      points: Points to rotate given `matrix` of shape (prefix, ..., dof).

    Returns:
      A rotated set of points of shape (prefix, ..., dof).
    """
    prefix_len = matrix.ndim - 2
    chex.assert_equal_shape_prefix((matrix, points), prefix_len)

    num_dof = matrix.shape[-1]
    matrix_p = matrix.reshape(-1, num_dof, num_dof)
    points_p = points.reshape(matrix_p.shape[0], -1, num_dof)
    return jnp.einsum(
        'pab,pnb->pna', matrix_p, points_p, precision='float32'
    ).reshape(points.shape)


@jax.jit
def pose_from_xy_yaw(
        xy: jax.Array, yaw: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Gets 2D transformation matrix and delta yaw with any prefix shape.

    Applying the transformation using returned values will rotate
    counter-clockwise by yaw, and then translate by xy.
    Example: a unit vector at xy with direction equals yaw will be at (0, 0)
    with direction equals 0 after the transformation.

    Args:
      xy: XY positions of points of shape (..., 2).
      yaw: Orientation in radians with shape (...).

    Returns:
      Transformation matrix and delta yaw. Note it is used as matmul(pose_matrix,
      pts).
    """

    def _pose_from_xy_yaw(
            local_xy: jax.Array, local_yaw: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Helper function for pose_from_xy_yaw without extra prefix shape."""
        pose_yaw = -jnp.array(local_yaw)  # rotate x-axis by yaw count-clockwise.
        c, s = jnp.cos(pose_yaw), jnp.sin(pose_yaw)
        rotation_matrix = jnp.array([[c, -s], [s, c]])
        translation_vector = -jnp.array([
            [c * local_xy[0] - s * local_xy[1]],
            [s * local_xy[0] + c * local_xy[1]],
        ])
        pose_matrix = jnp.concatenate(
            [rotation_matrix, translation_vector], axis=-1
        )
        pose_matrix = jnp.concatenate(
            [pose_matrix, jnp.array([[0.0, 0.0, 1.0]])], axis=0
        )
        return pose_matrix, pose_yaw

    prefix_len = xy.ndim - 1
    chex.assert_equal_shape_prefix((xy, yaw), prefix_len)
    func = _pose_from_xy_yaw
    for _ in range(prefix_len):
        # Vectorization, one vmap for one extra dimension.
        func = jax.vmap(func)

    return func(xy, yaw)


def _transform_points(matrix: jax.Array, pts: jax.Array) -> jax.Array:
    """Helper function for transform_points."""
    # Transform 2D points using a transformation matrix.
    dof = pts.shape[-1]
    pad_width = [(0, 1) if i == pts.ndim - 1 else (0, 0) for i in range(pts.ndim)]
    # Add homogeneous dimension.
    out_pts = jnp.pad(pts, pad_width, mode='constant', constant_values=1.0)
    # Explicitly set high precision (for TPU).
    out_pts = out_pts.dot(matrix.T, precision='float32')[..., :dof]
    chex.assert_equal_shape((pts, out_pts))
    return out_pts


@jax.named_scope('transform_points')
def transform_points(pose_matrix: jax.Array, pts: jax.Array) -> jax.Array:
    """Transforms points into new coordinates with any prefix shape.

    Args:
      pose_matrix: Matrix representing the transformation into the frame of some
        pose of shape (prefix, dof+1, dof+1).
      pts: Points to translate of shape (prefix, ..., dof).

    Returns:
      Points transformed by the `pose_matrix` of shape (prefix, ..., dof).
    """
    prefix_len = pose_matrix.ndim - 2
    chex.assert_equal_shape_prefix((pose_matrix, pts), prefix_len)
    func = _transform_points
    for _ in range(prefix_len):
        func = jax.vmap(func)

    return func(pose_matrix, pts)


def transform_yaw(pose_yaw: jax.Array, yaw: jax.Array) -> jax.Array:
    """Transforms yaw with any prefix shape."""
    return yaw + pose_yaw


@jax.named_scope('transform_trajectories')
def transform_trajectories(
        traj: jax.Array, pose_matrix: jax.Array, pose_yaw: jax.Array
) -> jax.Array:
    """Transforms trajectories by given pose with any prefix shape.

    Args:
      traj: jax.Array of shape (prefix_shape, ..., dof), where prefix_shape can be
        any dimensions, dof must be either 5 or 7. Dof 5 represents [x, y, length,
        width, yaw], and 7 represents [x, y, length, width, yaw, vel_x, vel_y]
      pose_matrix: Matrix representing the transformation into the frame of some
        pose of shape (prefix_shape, 3, 3).
      pose_yaw: Rotation angle of the transformation of shape (prefix_shape).

    Returns:
      Transformed trajectories with same shape as inputs traj.
    """
    prefix_len = pose_matrix.ndim - 2
    chex.assert_equal_shape_prefix((traj, pose_matrix, pose_yaw), prefix_len)

    def _transform_trajectories(
            local_traj: jax.Array, local_matrix: jax.Array, local_yaw: jax.Array
    ) -> jax.Array:
        """Helper function for transform_trajectories."""
        out_xy = _transform_points(local_matrix, local_traj[..., :2])
        out_yaw = local_traj[..., 4:5] + local_yaw
        out_traj = jnp.concatenate([out_xy, local_traj[..., 2:4], out_yaw], -1)

        dof = local_traj.shape[-1]
        if dof == 7:
            vel_xy = local_traj[..., 5:7]
            out_vel_xy = _transform_points(local_matrix, vel_xy)
            origin = _transform_points(local_matrix, jnp.zeros_like(vel_xy))
            out_traj = jnp.concatenate([out_traj, out_vel_xy - origin], -1)
        elif dof != 5:
            raise ValueError(
                f'Trajectory must have degree of freedom 5 or 7, got {dof}'
            )
        return out_traj

    func = _transform_trajectories
    for _ in range(prefix_len):
        func = jax.vmap(func)
    return func(traj, pose_matrix, pose_yaw)


@jax.named_scope('transform_direction')
def transform_direction(
        pose_matrix: jax.Array, pts_dir: jax.Array
) -> jax.Array:
    """Transforms direction with any prefix shape.

    Args:
      pose_matrix: Matrix representing the transformation into the frame of some
        pose of shape (prefix_shape, 3, 3).
      pts_dir: Vector direction to transform of shape (prefix_shape, ..., 2).

    Returns:
      Transformed direction.
    """
    prefix_len = pose_matrix.ndim - 2
    chex.assert_equal_shape_prefix((pts_dir, pose_matrix), prefix_len)
    return rotate_points(pose_matrix[..., :2, :2], pts_dir)


def corners_from_bbox(bbox: jax.Array) -> jax.Array:
    """Computes corners for one 5 dof bbox."""
    chex.assert_shape(bbox, (5,))
    c, s = jnp.cos(bbox[4]), jnp.sin(bbox[4])
    lc = bbox[2] / 2 * c
    ls = bbox[2] / 2 * s
    wc = bbox[3] / 2 * c
    ws = bbox[3] / 2 * s
    points = jnp.array([
        [lc + ws, ls - wc],
        [lc - ws, ls + wc],
        [-lc - ws, -ls + wc],
        [-lc + ws, -ls - wc],
    ])
    points += bbox[jnp.newaxis, :2]
    return points


def corners_from_bboxes(bboxes: jax.Array) -> jax.Array:
    """Computes corners for 5 dof bboxes with any prefix shape."""
    chex.assert_shape(bboxes, (..., 5))
    func = corners_from_bbox
    for _ in range(bboxes.ndim - 1):
        func = jax.vmap(func)
    return func(bboxes)


def has_overlap(bboxes_a: jax.Array, bboxes_b: jax.Array) -> jax.Array:
    """Checks if 5 dof bboxes (with any prefix shape) overlap with each other.

    It does a 1:1 comparison of equivalent batch indices.

    The algorithm first computes bboxes_a's projection on bboxes_b's axes and
    check if there is an overlap between the projection. It then computes
    bboxes_b's projection on bboxes_a's axes and check overlap. Two bboxes are
    overlapped if and only if there is overlap in both steps.

    Args:
      bboxes_a: Bounding boxes of the above format of shape (..., 5). The last
        dimension represents [x, y, length, width, yaw].
      bboxes_b: Bounding boxes of the above format of shape (..., 5).

    Returns:
      Boolean array which specifies whether `bboxes_a` and `bboxes_b` overlap each
        other of shape (...).
    """
    chex.assert_shape(bboxes_a, (..., 5))
    chex.assert_equal_shape((bboxes_a, bboxes_b))

    def _overlap_a_over_b(first, second):
        c, s = jnp.cos(first[..., 4]), jnp.sin(first[..., 4])
        # same as rotation matrix.
        normals_t = jnp.stack(
            [jnp.stack([c, -s], axis=-1), jnp.stack([s, c], axis=-1)], axis=-2
        )

        # 1) Computes corners for bboxes.
        corners_a = corners_from_bboxes(first)
        corners_b = corners_from_bboxes(second)
        # 2) Project corners of first bbox to second bbox's axes.
        # Forces float32 computation for better accuracy.
        # Otherwise running on TPU will default to bfloat and does not produce
        # accurate results.
        # (..., 4, 2).
        proj_a = jnp.matmul(corners_a, normals_t, precision='float32')
        # (..., 2).
        min_a = jnp.min(proj_a, axis=-2)
        max_a = jnp.max(proj_a, axis=-2)
        proj_b = jnp.matmul(corners_b, normals_t, precision='float32')
        min_b = jnp.min(proj_b, axis=-2)
        max_b = jnp.max(proj_b, axis=-2)
        # 3) Check if the projection along axis overlaps.
        distance = jnp.minimum(max_a, max_b) - jnp.maximum(min_a, min_b)
        return jnp.all(distance > 0, axis=-1)

    return jnp.logical_and(
        _overlap_a_over_b(bboxes_a, bboxes_b),
        _overlap_a_over_b(bboxes_b, bboxes_a),
    )
    return batched_fn(traj)
