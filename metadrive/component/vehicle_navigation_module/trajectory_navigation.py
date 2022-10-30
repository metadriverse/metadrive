from metadrive.component.vehicle_navigation_module.base_navigation import BaseNavigation
from metadrive.manager.waymo_map_manager import WaymoMapManager


class WaymoTrajectoryNavigation(BaseNavigation):
    """
    This module enabling follow a given reference trajectory given a map
    # TODO(LQY): make this module a general module for navigation
    """
    DESCRETE_LEN = 4  # m

    def __init__(
            self,
            engine,
            show_navi_mark: bool = False,
            random_navi_mark_color=False,
            show_dest_mark=False,
            show_line_to_dest=False,
            panda_color=None
    ):
        super(NodeNetworkNavigation, self).__init__(
            engine, show_navi_mark, random_navi_mark_color, show_dest_mark, show_line_to_dest, panda_color=panda_color
        )
        self.reference_trajectory = None

    def reset(self, map: BaseMap, current_lane):
        super(TrajectoryNavigation, self).reset(map, current_lane)
        self.reference_trajectory = self.get_trajectory()
        self.set_route(None, None)

    def set_route(self, current_lane_index: str, destination: str):
        self.checkpoints = self.descretize_reference_trajectory()
        self._target_checkpoints_index = [0, 1]
        # update routing info
        if len(self.checkpoints) <= 2:
            self.checkpoints = [current_lane_index[0], current_lane_index[1]]
            self._target_checkpoints_index = [0, 0]
        assert len(self.checkpoints
                   ) >= 2, "Can not find a route from {} to {}".format(current_lane_index[0], destination)

        self._navi_info.fill(0.0)

        if self._dest_node_path is not None:
            check_point = self.reference_trajectory.end
            self._dest_node_path.setPos(check_point[0], -check_point[1], 1.8)

    def get_trajectory(self):
        return self.engine.map_manager.current_sdc_route

    def descretize_reference_trajectory(self):
        ret = []
        length = self.reference_trajectory.length
        num = int(length / self.DESCRETE_LEN)
        for i in range(num):
            ret.append(self.reference_trajectory.local_coordinates(i*self.DESCRETE_LEN, 0))
        ret.append(self.reference_trajectory.end)
        return ret

    def update_localization(self, ego_vehicle):
        """
        It is called every step
        """
        raise NotImplementedError

    def get_current_lateral_range(self, current_position, engine) -> float:
        raise NotImplementedError

    def get_current_lane_width(self) -> float:
        return self.current_lane.width

    def get_current_lane_num(self) -> float:
        return 1

    def _get_current_lane(self, ego_vehicle):
        raise NotImplementedError