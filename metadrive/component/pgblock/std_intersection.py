from metadrive.component.pgblock.intersection import InterSection
from metadrive.component.pg_space import Parameter


class StdInterSection(InterSection):
    def _try_plug_into_previous_block(self) -> bool:
        self._config[Parameter.change_lane_num] = 0
        success = super(StdInterSection, self)._try_plug_into_previous_block()
        return success
