from metadrive.component.pg_space import Parameter
from metadrive.component.pgblock.intersection import InterSection, InterSectionWithUTurn


class StdInterSection(InterSection):
    def _try_plug_into_previous_block(self) -> bool:
        self._config[Parameter.change_lane_num] = 0
        success = super(StdInterSection, self)._try_plug_into_previous_block()
        return success


class StdInterSectionWithUTurn(InterSectionWithUTurn):
    def _try_plug_into_previous_block(self) -> bool:
        self._config[Parameter.change_lane_num] = 0
        success = super(StdInterSectionWithUTurn, self)._try_plug_into_previous_block()
        return success
