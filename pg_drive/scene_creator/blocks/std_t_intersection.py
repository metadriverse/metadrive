from pg_drive.pg_config.parameter_space import Parameter
from pg_drive.scene_creator.blocks.t_intersection import TInterSection


class StdTInterSection(TInterSection):
    def _try_plug_into_previous_block(self) -> bool:
        self._config[Parameter.change_lane_num] = 0
        success = super(StdTInterSection, self)._try_plug_into_previous_block()
        return success
