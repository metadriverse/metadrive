import unittest

from pgdrive import PGDriveEnv


class TestObsActionSpace(unittest.TestCase):
    def setUp(self):
        self.env = PGDriveEnv()

    def test_obs_space(self):
        obs = self.env.reset()
        assert self.env.observation_space.contains(obs), (self.env.observation_space, obs.shape)
        obs, _, _, _ = self.env.step(self.env.action_space.sample())
        assert self.env.observation_space.contains(obs), (self.env.observation_space, obs.shape)

    def tearDown(self):
        self.env.close()


if __name__ == '__main__':
    unittest.main()
