def encode_road_section_lane_width_id(roadId, sectionId, laneId, widthId):
    """

    Args:
      roadId:
      sectionId:
      laneId:
      widthId:

    Returns:

    """
    return ".".join([str(roadId), str(sectionId), str(laneId), str(widthId)])


class LinkIndex:
    """
    Overall index of all links in the file, save everything as successors,
    predecessors can be found via a reverse search
    """
    def __init__(self):
        self._successors = {}

    def create_from_opendrive(self, opendrive):
        """Create a LinkIndex from an OpenDrive object.

        Args:
          opendrive: OpenDrive style object.

        Returns:

        """
        self._add_junctions(opendrive)
        # Extract link information from road lanes
        for road in opendrive.roads:
            for lane_section in road.lanes.lane_sections:
                for lane in lane_section.allLanes:
                    parametric_lane_id = encode_road_section_lane_width_id(road.id, lane_section.idx, lane.id, -1)

                    # Not the last lane section? > Next lane section in same road
                    if lane_section.idx < road.lanes.getLastLaneSectionIdx():
                        successorId = encode_road_section_lane_width_id(
                            road.id, lane_section.idx + 1, lane.link.successorId, -1
                        )

                        self.add_link(parametric_lane_id, successorId, lane.id >= 0)

                    # Last lane section! > Next road in first lane section
                    # Try to get next road
                    elif (road.link.successor is not None and road.link.successor.elementType != "junction"):

                        next_road = opendrive.getRoad(road.link.successor.element_id)

                        if next_road is not None:

                            if road.link.successor.contactPoint == "start":
                                successorId = encode_road_section_lane_width_id(
                                    next_road.id, 0, lane.link.successorId, -1
                                )

                            else:  # contact point = end
                                successorId = encode_road_section_lane_width_id(
                                    next_road.id,
                                    next_road.lanes.getLastLaneSectionIdx(),
                                    lane.link.successorId,
                                    -1,
                                )
                            self.add_link(parametric_lane_id, successorId, lane.id >= 0)

                    # Not first lane section? > Previous lane section in same road
                    if lane_section.idx > 0:
                        predecessorId = encode_road_section_lane_width_id(
                            road.id, lane_section.idx - 1, lane.link.predecessorId, -1
                        )

                        self.add_link(predecessorId, parametric_lane_id, lane.id >= 0)

                    # First lane section! > Previous road
                    # Try to get previous road
                    elif (road.link.predecessor is not None and road.link.predecessor.elementType != "junction"):

                        prevRoad = opendrive.getRoad(road.link.predecessor.element_id)

                        if prevRoad is not None:

                            if road.link.predecessor.contactPoint == "start":
                                predecessorId = encode_road_section_lane_width_id(
                                    prevRoad.id, 0, lane.link.predecessorId, -1
                                )

                            else:  # contact point = end
                                predecessorId = encode_road_section_lane_width_id(
                                    prevRoad.id,
                                    prevRoad.lanes.getLastLaneSectionIdx(),
                                    lane.link.predecessorId,
                                    -1,
                                )
                            self.add_link(predecessorId, parametric_lane_id, lane.id >= 0)

    def add_link(self, parametric_lane_id, successor, reverse: bool = False):
        """

        Args:
          parametric_lane_id:
          successor:
          reverse:  (Default value = False)

        Returns:

        """

        # if reverse, call function recursively with switched parameters
        if reverse:
            self.add_link(successor, parametric_lane_id)
            return

        if parametric_lane_id not in self._successors:
            self._successors[parametric_lane_id] = []

        if successor not in self._successors[parametric_lane_id]:
            self._successors[parametric_lane_id].append(successor)

    def _add_junctions(self, opendrive):
        """

        Args:
          opendrive:

        Returns:

        """
        # add junctions to link_index
        # if contact_point is start, and laneId from connecting_road is negative
        # the connecting_road is the successor
        # if contact_point is start, and laneId from connecting_road is positive
        # the connecting_road is the predecessor
        # for contact_point == end it's exactly the other way
        for junction in opendrive.junctions:
            for connection in junction.connections:
                incoming_road = opendrive.getRoad(connection.incomingRoad)
                connecting_road = opendrive.getRoad(connection.connectingRoad)
                contact_point = connection.contactPoint

                for lane_link in connection.laneLinks:
                    if contact_point == "start":

                        # decide which lane section to use (first or last)
                        if lane_link.fromId < 0:
                            lane_section_idx = (incoming_road.lanes.getLastLaneSectionIdx())
                        else:
                            lane_section_idx = 0
                        incoming_road_id = encode_road_section_lane_width_id(
                            incoming_road.id, lane_section_idx, lane_link.fromId, -1
                        )
                        connecting_road_id = encode_road_section_lane_width_id(
                            connecting_road.id, 0, lane_link.toId, -1
                        )
                        self.add_link(incoming_road_id, connecting_road_id, lane_link.toId > 0)
                    else:
                        # decide which lane section to use (first or last)
                        if lane_link.fromId < 0:
                            lane_section_idx = 0

                        else:
                            lane_section_idx = (incoming_road.lanes.getLastLaneSectionIdx())
                        incoming_road_id = encode_road_section_lane_width_id(incoming_road.id, 0, lane_link.fromId, -1)
                        connecting_road_id = encode_road_section_lane_width_id(
                            connecting_road.id,
                            connecting_road.lanes.getLastLaneSectionIdx(),
                            lane_link.toId,
                            -1,
                        )
                        self.add_link(incoming_road_id, connecting_road_id, lane_link.toId < 0)

    def remove(self, parametric_lane_id):
        """

        Args:
          parametric_lane_id:

        """
        # Delete key
        if parametric_lane_id in self._successors:
            del self._successors[parametric_lane_id]

        # Delete all occurances in successor lists
        for successorsId, successors in self._successors.items():
            if parametric_lane_id in successors:
                self._successors[successorsId].remove(parametric_lane_id)

    def get_successors(self, parametric_lane_id: str) -> list:
        """

        Args:
          parametric_lane_id: Id of ParametricLane for which to search
            successors.

        Returns:
          List of successors belonging the the ParametricLane.
        Par
        """
        if parametric_lane_id not in self._successors:
            return []

        return self._successors[parametric_lane_id]

    def get_predecessors(self, parametric_lane_id: str) -> list:
        """

        Args:
          parametric_lane_id: Id of ParametricLane for which to search predecessors.

        Returns:
          List of predecessors of a ParametricLane.
        """
        predecessors = []
        for successor_plane_id, successors in self._successors.items():
            if parametric_lane_id not in successors:
                continue

            if successor_plane_id in predecessors:
                continue

            predecessors.append(successor_plane_id)

        return predecessors
