import numpy as np


def test_utils():
    from metadrive.utils.math_utils import safe_clip, safe_clip_for_small_array, get_vertical_vector, distance_greater

    arr = np.array([np.nan, np.inf, -np.inf, 1000000000000000000000000000, -10000000000000000, 0, 1])
    ans = np.array([0, 1, -1, 1, -1, 0, 1])
    assert np.array_equal(safe_clip(arr, -1, 1), ans)
    assert np.array_equal(safe_clip_for_small_array(arr, -1, 1), ans)

    assert np.dot(get_vertical_vector([0, 1])[0], [0, 1]) == 0
    assert np.dot(get_vertical_vector([0, 1])[1], [0, 1]) == 0

    for _ in range(20):
        pos = np.random.normal(0, 1, size=(2, ))
        assert distance_greater(pos, (0, 0), 0.5) == (abs(np.linalg.norm(pos, ord=2)) > 0.5)


if __name__ == '__main__':
    test_utils()
