from gym.envs.registration import register

from pg_drive.envs import GeneralizationRacing

register(
    id='GeneralizationRacing-v0',
    entry_point='pg_drive:GeneralizationRacing',
)
