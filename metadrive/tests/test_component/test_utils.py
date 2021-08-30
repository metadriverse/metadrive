import numpy as np

from metadrive.utils.cutils import import_cutils


def _test_cutils(cutils):
    for _ in range(20):
        pos = np.random.normal(1000, 1000, size=(2, ))
        n = cutils.cutils_norm(*pos.tolist())
        assert abs(n - abs(np.linalg.norm(pos, ord=2))) < 1e-4

        clip0 = cutils.cutils_clip(pos[0], 999, 1001)
        clip1 = cutils.cutils_clip(pos[1], 999, 1001)
        assert np.array_equal(np.clip(pos, 999, 1001), np.array([clip0, clip1]))

        ppos = cutils.cutils_panda_position(*pos.tolist())
        assert ppos[0] == pos[0]
        assert ppos[1] == -pos[1]


def test_cutils():
    cutils = import_cutils(use_fake_cutils=False)
    _test_cutils(cutils)


def test_fake_cutils():
    cutils = import_cutils(use_fake_cutils=True)
    _test_cutils(cutils)


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
    test_cutils()
    test_fake_cutils()
    test_utils()
