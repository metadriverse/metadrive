from pgdrive.pg_config.parameter_space import Parameter, BlockParameterSpace
from pgdrive.pg_config.pg_space import PGSpace
from pgdrive.utils.constans import Decoration
from pgdrive.scene_creator.blocks.intersection import InterSection
from pgdrive.scene_creator.lanes.lane import LineType, LineColor
from pgdrive.scene_creator.road.road import Road
from pgdrive.utils.constans import Goal


class TInterSection(InterSection):
    """
    A structure like X Intersection, code copied from it mostly
    """

    ID = "T"
    SOCKET_NUM = 2
    PARAMETER_SPACE = PGSpace(BlockParameterSpace.T_INTERSECTION)

    def _try_plug_into_previous_block(self) -> bool:
        no_cross = super(TInterSection, self)._try_plug_into_previous_block()
        self._exclude_lanes()
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

    def _exclude_lanes(self):
        para = self.get_config()
        t_type = para[Parameter.t_intersection_type]
        self.add_sockets(self._pre_block_socket)
        start_node = self._sockets[t_type].negative_road.end_node
        end_node = self._sockets[t_type].positive_road.start_node
        for i in range(4):
            if i == t_type:
                continue
            exit_node = self._sockets[i].positive_road.start_node if i != Goal.ADVERSE else self._sockets[
                i].negative_road.start_node
            pos_lanes = self.block_network.remove_all_roads(start_node, exit_node)
            entry_node = self._sockets[i].negative_road.end_node if i != Goal.ADVERSE else self._sockets[
                i].positive_road.end_node
            neg_lanes = self.block_network.remove_all_roads(entry_node, end_node)

            # TODO small vis bug may raise in a corner case,
            #  these code can fix it but will introduce a new get_closest_lane_index bug
            # if (i + 2) % 4 == t_type:
            #     # for vis only, solve a bug existing in a corner case,
            #     for lane in neg_lanes:
            #         lane.reset_start_end(lane.start, lane.position(lane.length / 2, 0))
            #     self.block_network.add_road(Road(Decoration.start, Decoration.end), neg_lanes)
            #
            #     for lane in pos_lanes:
            #         lane.reset_start_end(lane.position(lane.length / 2, 0), lane.end)
            #     self.block_network.add_road(Road(Decoration.start, Decoration.end), pos_lanes)

        self._change_vis(t_type)
        self._sockets.pop(-1)
        socket = self._sockets.pop(t_type)
        self.block_network.remove_all_roads(socket.positive_road.start_node, socket.positive_road.end_node)
        self.block_network.remove_all_roads(socket.negative_road.start_node, socket.negative_road.end_node)
        self._reborn_roads.remove(socket.negative_road)
