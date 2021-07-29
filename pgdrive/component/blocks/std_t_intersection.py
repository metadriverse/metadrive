from pgdrive.component.blocks.t_intersection import TInterSection
from pgdrive.utils.space import Parameter


class StdTInterSection(TInterSection):
    def _try_plug_into_previous_block(self) -> bool:
        self._config[Parameter.change_lane_num] = 0
        success = super(StdTInterSection, self)._try_plug_into_previous_block()
        return success
