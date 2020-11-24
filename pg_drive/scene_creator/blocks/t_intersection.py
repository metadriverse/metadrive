from pg_drive.pg_config.parameter_space import Parameter, BlockParameterSpace
from pg_drive.scene_creator.lanes.lane import LineType, LineColor
from pg_drive.pg_config.pg_space import PgSpace
from pg_drive.scene_creator.road.road import Road
from pg_drive.scene_creator.blocks.intersection import InterSection
from pg_drive.scene_creator.basic_utils import Goal
from pg_drive.scene_creator.basic_utils import Decoration


class TInterSection(InterSection):
    """
    A structure like X Intersection, code copied from it mostly
    """

    ID = "T"
    SOCKET_NUM = 2
    PARAMETER_SPACE = PgSpace(BlockParameterSpace.T_INTERSECTION)

    def _try_plug_into_previous_block(self) -> bool:
        no_cross = super(TInterSection, self)._try_plug_into_previous_block()
        self._excluede_lanes()
        return no_cross

    def _change_vis(self, t_type):
        # not good here,
        next_part_socket = self._sockets[(t_type + 1) % 4]
        next_positive = next_part_socket.positive_road
        next_negative = next_part_socket.negative_road

        last_part_socket = self._sockets[(t_type + 3) % 4]
        last_positive = last_part_socket.positive_road
        last_negative = last_part_socket.negative_road
        if t_type == Goal.LEFT:
            next_positive = next_part_socket.negative_road
            next_negative = next_part_socket.positive_road
        if t_type == Goal.RIGHT:
            last_positive = last_part_socket.negative_road
            last_negative = last_part_socket.positive_road

        for i, road in enumerate([Road(last_negative.end_node, next_positive.start_node),
                                  Road(next_negative.end_node, last_positive.start_node)]):
            lanes = road.get_lanes(self.block_network)
            outside_type = LineType.SIDE if i == 0 else LineType.NONE
            for k, lane in enumerate(lanes):
                line_types = [LineType.STRIPED, LineType.STRIPED
                              ] if k != len(lanes) - 1 else [LineType.STRIPED, outside_type]
                lane.line_types = line_types
                if k == 0:
                    lane.line_color = [LineColor.YELLOW, LineColor.GREY]
                    if i == 1:
                        lane.line_types[0] = LineType.NONE

    def _excluede_lanes(self):
        para = self.get_config()
        t_type = para[Parameter.t_intersection_type]
        radius = para[Parameter.radius]
        # use a small trick here
        self.add_sockets(self._pre_block_socket)
        start_node = self._sockets[t_type].negative_road.end_node
        end_node = self._sockets[t_type].positive_road.start_node
        for i in range(4):
            if i == t_type:
                continue
            exit_node = self._sockets[i].positive_road.start_node if i != Goal.ADVERSE else self._sockets[
                i].negative_road.start_node
            lanes_on_p_road = self.block_network.remove_road(Road(start_node, exit_node))
            entry_node = self._sockets[i].negative_road.end_node if i != Goal.ADVERSE else self._sockets[
                i].positive_road.end_node
            lanes_on_n_road = self.block_network.remove_road(Road(entry_node, end_node))
            if i == (t_type + 2) % 4:
                for lane in lanes_on_n_road:
                    lane.end = lane.position(radius, 0)
                    lane.update_length()
                for lane in lanes_on_p_road:
                    lane.start = lane.position(lane.length - radius, 0)
                    lane.update_length()
                self.block_network.add_road(Road(start_node, exit_node), lanes_on_p_road)
                self.block_network.add_road(Road(entry_node, end_node), lanes_on_n_road)
        self._change_vis(t_type)
        self._sockets.pop(-1)
        socket = self._sockets.pop(t_type)
        self.block_network.remove_road(socket.positive_road)
        self.block_network.remove_road(socket.negative_road)
        self._reborn_roads.remove(socket.negative_road)
