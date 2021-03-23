from pgdrive.envs.action_repeat_env import ActionRepeat
from pgdrive.envs.generation_envs.change_density_env import ChangeDensityEnv
from pgdrive.envs.generation_envs.change_friction_env import ChangeFrictionEnv
from pgdrive.envs.multi_agent_pgdrive import MultiAgentPGDrive
from pgdrive.envs.pgdrive_env import PGDriveEnv
from pgdrive.envs.pgdrive_env_v2 import PGDriveEnvV2
from pgdrive.envs.remote_pgdrive_env import RemotePGDrive
from pgdrive.envs.top_down_env import TopDownSingleFramePGDriveEnv, TopDownPGDriveEnv

# For compatibility
GeneralizationRacing = PGDriveEnv
