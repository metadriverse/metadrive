from metadrive.component.pgblock.intersection import InterSection
from metadrive.component.pgblock.pg_block import PGBlockSocket
from metadrive.component.road_network import Road
from metadrive.constants import Goal, PGLineType, PGLineColor
from metadrive.component.pg_space import ParameterSpace, Parameter, BlockParameterSpace


class TInterSection(InterSection):
    """
    A structure like X Intersection, code copied from it mostly
    """

    ID = "T"
    SOCKET_NUM = 2
    PARAMETER_SPACE = ParameterSpace(BlockParameterSpace.T_INTERSECTION)

    def _try_plug_into_previous_block(self) -> bool:
        no_cross = super(TInterSection, self)._try_plug_into_previous_block()
        self._exclude_lanes()
        return no_cross

    def _change_vis(self, t_type):
        # not good here,
        next_part_socket = self.get_socket_list()[(t_type + 1) % 4]
        # next_part_socket = self._sockets[(t_type + 1) % 4]  # FIXME pzh: Help! @LQY What is in this part?
        next_positive = next_part_socket.positive_road
        next_negative = next_part_socket.negative_road

        last_part_socket = self.get_socket_list()[(t_type + 3) % 4]
        # last_part_socket = self._sockets[(t_type + 3) % 4]  # FIXME pzh: Help! @LQY
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
            outside_type = PGLineType.SIDE if i == 0 else PGLineType.NONE
            for k, lane in enumerate(lanes):
                line_types = [PGLineType.NONE, PGLineType.NONE
                              ] if k != len(lanes) - 1 else [PGLineType.NONE, outside_type]
                lane.line_types = line_types
                if k == 0:
                    lane.line_colors = [PGLineColor.YELLOW, PGLineColor.GREY]
                    if i == 1:
                        lane.line_types[0] = PGLineType.NONE

    def _exclude_lanes(self):
        para = self.get_config()
        t_type = para[Parameter.t_intersection_type]
        self.add_sockets(self.pre_block_socket)

        start_node = self._sockets[PGBlockSocket.get_real_index(self.name, t_type)].negative_road.end_node
        end_node = self._sockets[PGBlockSocket.get_real_index(self.name, t_type)].positive_road.start_node
        for i in range(4):
            if i == t_type:
                continue
            index_i = PGBlockSocket.get_real_index(self.name, i) if i < 3 else self.pre_block_socket_index
            exit_node = self._sockets[index_i].positive_road.start_node if i != Goal.ADVERSE else self._sockets[
                index_i].negative_road.start_node
            pos_lanes = self.block_network.remove_all_roads(start_node, exit_node)
            entry_node = self._sockets[index_i].negative_road.end_node if i != Goal.ADVERSE else self._sockets[
                index_i].positive_road.end_node
            neg_lanes = self.block_network.remove_all_roads(entry_node, end_node)

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
        self._sockets.pop(self.pre_block_socket.index)
        socket = self._sockets.pop(PGBlockSocket.get_real_index(self.name, t_type))
        self.block_network.remove_all_roads(socket.positive_road.start_node, socket.positive_road.end_node)
        self.block_network.remove_all_roads(socket.negative_road.start_node, socket.negative_road.end_node)
        self._respawn_roads.remove(socket.negative_road)

    def _add_one_socket(self, socket: PGBlockSocket):
        assert isinstance(socket, PGBlockSocket), "Socket list only accept BlockSocket Type"

        # mute warning in T interection

        # if socket.index is not None and not socket.index.startswith(self.name):
        #     logging.warning(
        #         "The adding socket has index {}, which is not started with this block name {}. This is dangerous! "
        #         "Current block has sockets: {}.".format(socket.index, self.name, self.get_socket_indices())
        #     )
        if socket.index is None:
            # if this socket is self block socket
            socket.set_index(self.name, len(self._sockets))
        self._sockets[socket.index] = socket

    def enable_u_turn(self, enable_u_turn: bool):
        raise ValueError("T intersection didn't support u_turn now")
