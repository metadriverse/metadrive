from pg_drive.envs import GeneralizationRacing

from gym.envs.registration import register

register(
    id='GeneralizationRacing-v0',
    entry_point='pg_drive:GeneralizationRacing',
)
