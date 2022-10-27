# -*- coding: utf-8 -*-
"""
These scripts are copied from https://github.com/liuyf5231/opendriveparser. Credit: https://github.com/liuyf5231
"""
import numpy as np
from lxml import etree
from metadrive.utils.opendrive_map_utils.elements.opendrive import OpenDrive, Header
from metadrive.utils.opendrive_map_utils.elements.road import Road
from metadrive.utils.opendrive_map_utils.elements.roadLink import (
    Predecessor as RoadLinkPredecessor,
    Successor as RoadLinkSuccessor,
    Neighbor as RoadLinkNeighbor,
)
from metadrive.utils.opendrive_map_utils.elements.roadtype import (
    RoadType,
    Speed as RoadTypeSpeed,
)
from metadrive.utils.opendrive_map_utils.elements.roadElevationProfile import (
    ElevationRecord as RoadElevationProfile,
)
from metadrive.utils.opendrive_map_utils.elements.roadLateralProfile import (
    Superelevation as RoadLateralProfileSuperelevation,
    Crossfall as RoadLateralProfileCrossfall,
    Shape as RoadLateralProfileShape,
)
from metadrive.utils.opendrive_map_utils.elements.roadLanes import (
    LaneOffset as RoadLanesLaneOffset,
    Lane as RoadLaneSectionLane,
    LaneSection as RoadLanesSection,
    LaneWidth as RoadLaneSectionLaneWidth,
    LaneBorder as RoadLaneSectionLaneBorder,
)
from metadrive.utils.opendrive_map_utils.elements.junction import (
    Junction,
    Connection as JunctionConnection,
    LaneLink as JunctionConnectionLaneLink,
)

__author__ = "Benjamin Orthen, Stefan Urban"
__copyright__ = "TUM Cyber-Physical Systems Group"
__credits__ = ["Priority Program SPP 1835 Cooperative Interacting Automobiles"]
__version__ = "1.0.2"
__maintainer__ = "Benjamin Orthen"
__email__ = "commonroad-i06@in.tum.de"
__status__ = "Released"


def parse_opendrive(root_node) -> OpenDrive:
    """Tries to parse XML tree, returns OpenDRIVE object

    Args:
      root_node:

    Returns:
      The object representing an OpenDrive specification.

    """

    # Only accept lxml element
    if not etree.iselement(root_node):
        raise TypeError("Argument root_node is not a xml element")

    opendrive = OpenDrive()

    # Header
    header = root_node.find("header")
    if header is not None:
        parse_opendrive_header(opendrive, header)

    # Junctions
    for junction in root_node.findall("junction"):
        parse_opendrive_junction(opendrive, junction)

    # Load roads
    for road in root_node.findall("road"):
        parse_opendrive_road(opendrive, road)

    return opendrive


def parse_opendrive_road_link(newRoad, opendrive_road_link):
    """

    Args:
      newRoad:
      opendrive_road_link:

    """
    predecessor = opendrive_road_link.find("predecessor")

    if predecessor is not None:
        newRoad.link.predecessor = RoadLinkPredecessor(
            predecessor.get("elementType"),
            predecessor.get("elementId"),
            predecessor.get("contactPoint"),
        )

    successor = opendrive_road_link.find("successor")

    if successor is not None:
        newRoad.link.successor = RoadLinkSuccessor(
            successor.get("elementType"),
            successor.get("elementId"),
            successor.get("contactPoint"),
        )

    for neighbor in opendrive_road_link.findall("neighbor"):
        newNeighbor = RoadLinkNeighbor(neighbor.get("side"), neighbor.get("elementId"), neighbor.get("direction"))

        newRoad.link.neighbors.append(newNeighbor)


def parse_opendrive_road_type(road, opendrive_xml_road_type: etree.ElementTree):
    """Parse opendrive road type and append to road object.

    Args:
      road: Road to append the parsed road_type to types.
      opendrive_xml_road_type: XML element which contains the information.
      opendrive_xml_road_type: etree.ElementTree:

    """
    speed = None
    if opendrive_xml_road_type.find("speed") is not None:
        speed = RoadTypeSpeed(
            max_speed=opendrive_xml_road_type.find("speed").get("max"),
            unit=opendrive_xml_road_type.find("speed").get("unit"),
        )

    road_type = RoadType(
        s_pos=opendrive_xml_road_type.get("s"),
        use_type=opendrive_xml_road_type.get("type"),
        speed=speed,
    )
    road.types.append(road_type)


def parse_opendrive_road_geometry(newRoad, road_geometry):
    """

    Args:
      newRoad:
      road_geometry:

    """

    startCoord = [float(road_geometry.get("x")), float(road_geometry.get("y"))]

    if road_geometry.find("line") is not None:
        newRoad.planView.addLine(
            startCoord,
            float(road_geometry.get("hdg")),
            float(road_geometry.get("length")),
        )

    elif road_geometry.find("spiral") is not None:
        newRoad.planView.addSpiral(
            startCoord,
            float(road_geometry.get("hdg")),
            float(road_geometry.get("length")),
            float(road_geometry.find("spiral").get("curvStart")),
            float(road_geometry.find("spiral").get("curvEnd")),
        )

    elif road_geometry.find("arc") is not None:
        newRoad.planView.addArc(
            startCoord,
            float(road_geometry.get("hdg")),
            float(road_geometry.get("length")),
            float(road_geometry.find("arc").get("curvature")),
        )

    elif road_geometry.find("poly3") is not None:
        newRoad.planView.addPoly3(
            startCoord,
            float(road_geometry.get("hdg")),
            float(road_geometry.get("length")),
            float(road_geometry.find("poly3").get("a")),
            float(road_geometry.find("poly3").get("b")),
            float(road_geometry.find("poly3").get("c")),
            float(road_geometry.find("poly3").get("d")),
        )
        # raise NotImplementedError()

    elif road_geometry.find("paramPoly3") is not None:
        if road_geometry.find("paramPoly3").get("pRange"):

            if road_geometry.find("paramPoly3").get("pRange") == "arcLength":
                pMax = float(road_geometry.get("length"))
            else:
                pMax = None
        else:
            pMax = None

        newRoad.planView.addParamPoly3(
            startCoord,
            float(road_geometry.get("hdg")),
            float(road_geometry.get("length")),
            float(road_geometry.find("paramPoly3").get("aU")),
            float(road_geometry.find("paramPoly3").get("bU")),
            float(road_geometry.find("paramPoly3").get("cU")),
            float(road_geometry.find("paramPoly3").get("dU")),
            float(road_geometry.find("paramPoly3").get("aV")),
            float(road_geometry.find("paramPoly3").get("bV")),
            float(road_geometry.find("paramPoly3").get("cV")),
            float(road_geometry.find("paramPoly3").get("dV")),
            pMax,
        )

    else:
        raise Exception("invalid xml")


def parse_opendrive_road_elevation_profile(newRoad, road_elevation_profile):
    """

    Args:
      newRoad:
      road_elevation_profile:

    """

    for elevation in road_elevation_profile.findall("elevation"):
        newElevation = (
            RoadElevationProfile(
                float(elevation.get("a")),
                float(elevation.get("b")),
                float(elevation.get("c")),
                float(elevation.get("d")),
                start_pos=float(elevation.get("s")),
            ),
        )

        newRoad.elevationProfile.elevations.append(newElevation)


def parse_opendrive_road_lateral_profile(newRoad, road_lateral_profile):
    """

    Args:
      newRoad:
      road_lateral_profile:

    """

    for superelevation in road_lateral_profile.findall("superelevation"):
        newSuperelevation = RoadLateralProfileSuperelevation(
            float(superelevation.get("a")),
            float(superelevation.get("b")),
            float(superelevation.get("c")),
            float(superelevation.get("d")),
            start_pos=float(superelevation.get("s")),
        )

        newRoad.lateralProfile.superelevations.append(newSuperelevation)

    for crossfall in road_lateral_profile.findall("crossfall"):
        newCrossfall = RoadLateralProfileCrossfall(
            float(crossfall.get("a")),
            float(crossfall.get("b")),
            float(crossfall.get("c")),
            float(crossfall.get("d")),
            side=crossfall.get("side"),
            start_pos=float(crossfall.get("s")),
        )

        newRoad.lateralProfile.crossfalls.append(newCrossfall)

    for shape in road_lateral_profile.findall("shape"):
        newShape = RoadLateralProfileShape(
            float(shape.get("a")),
            float(shape.get("b")),
            float(shape.get("c")),
            float(shape.get("d")),
            start_pos=float(shape.get("s")),
            start_pos_t=float(shape.get("t")),
        )

        newRoad.lateralProfile.shapes.append(newShape)


def parse_opendrive_road_lane_offset(newRoad, lane_offset):
    """

    Args:
      newRoad:
      lane_offset:

    """

    newLaneOffset = RoadLanesLaneOffset(
        float(lane_offset.get("a")),
        float(lane_offset.get("b")),
        float(lane_offset.get("c")),
        float(lane_offset.get("d")),
        start_pos=float(lane_offset.get("s")),
    )

    newRoad.lanes.laneOffsets.append(newLaneOffset)


def parse_opendrive_road_lane_section(newRoad, lane_section_id, lane_section):
    """

    Args:
      newRoad:
      lane_section_id:
      lane_section:

    """

    newLaneSection = RoadLanesSection(road=newRoad)

    # Manually enumerate lane sections for referencing purposes
    newLaneSection.idx = lane_section_id

    newLaneSection.sPos = float(lane_section.get("s"))
    newLaneSection.singleSide = lane_section.get("singleSide")

    sides = dict(
        left=newLaneSection.leftLanes,
        center=newLaneSection.centerLanes,
        right=newLaneSection.rightLanes,
    )

    for sideTag, newSideLanes in sides.items():

        side = lane_section.find(sideTag)

        # It is possible one side is not present
        if side is None:
            continue

        for lane in side.findall("lane"):

            new_lane = RoadLaneSectionLane(parentRoad=newRoad, lane_section=newLaneSection)
            new_lane.id = lane.get("id")
            new_lane.type = lane.get("type")

            # In some sample files the level is not specified according to the OpenDRIVE spec
            new_lane.level = ("true" if lane.get("level") in [1, "1", "true"] else "false")

            # Lane Links
            if lane.find("link") is not None:

                if lane.find("link").find("predecessor") is not None:
                    new_lane.link.predecessorId = (lane.find("link").find("predecessor").get("id"))

                if lane.find("link").find("successor") is not None:
                    new_lane.link.successorId = (lane.find("link").find("successor").get("id"))

            # Width
            for widthIdx, width in enumerate(lane.findall("width")):
                newWidth = RoadLaneSectionLaneWidth(
                    float(width.get("a")),
                    float(width.get("b")),
                    float(width.get("c")),
                    float(width.get("d")),
                    idx=widthIdx,
                    start_offset=float(width.get("sOffset")),
                )

                new_lane.widths.append(newWidth)

            # Border
            for borderIdx, border in enumerate(lane.findall("border")):
                newBorder = RoadLaneSectionLaneBorder(
                    float(border.get("a")),
                    float(border.get("b")),
                    float(border.get("c")),
                    float(border.get("d")),
                    idx=borderIdx,
                    start_offset=float(border.get("sOffset")),
                )

                new_lane.borders.append(newBorder)

            if lane.find("width") is None and lane.find("border") is not None:
                new_lane.widths = new_lane.borders
                new_lane.has_border_record = True

            # Road Marks
            # TODO implementation
            if lane.find("roadMark") is not None:
                new_lane.roadMark = dict(lane.find("roadMark").attrib)
            else:
                new_lane.roadMark = None
                # new_lane.has_border_record = True

            # Material
            # TODO implementation

            # Visiblility
            # TODO implementation

            # Speed
            # TODO implementation

            # Access
            # TODO implementation

            # Lane Height
            # TODO implementation

            # Rules
            # TODO implementation

            newSideLanes.append(new_lane)

    newRoad.lanes.lane_sections.append(newLaneSection)


def parse_opendrive_road(opendrive, road):
    """

    Args:
      opendrive:
      road:

    """

    newRoad = Road()

    newRoad.id = int(road.get("id"))
    newRoad.name = road.get("name")

    junctionId = int(road.get("junction")) if road.get("junction") != "-1" else None

    if junctionId:
        newRoad.junction = opendrive.getJunction(junctionId)

    # TODO verify road length
    newRoad.length = float(road.get("length"))

    # Links
    opendrive_road_link = road.find("link")
    if opendrive_road_link is not None:
        parse_opendrive_road_link(newRoad, opendrive_road_link)

    # Type
    for opendrive_xml_road_type in road.findall("type"):
        parse_opendrive_road_type(newRoad, opendrive_xml_road_type)

    # Plan view
    for road_geometry in road.find("planView").findall("geometry"):
        parse_opendrive_road_geometry(newRoad, road_geometry)

    # Elevation profile
    road_elevation_profile = road.find("elevationProfile")
    if road_elevation_profile is not None:
        parse_opendrive_road_elevation_profile(newRoad, road_elevation_profile)

    # Lateral profile
    road_lateral_profile = road.find("lateralProfile")
    if road_lateral_profile is not None:
        parse_opendrive_road_lateral_profile(newRoad, road_lateral_profile)

    # Lanes
    lanes = road.find("lanes")

    if lanes is None:
        raise Exception("Road must have lanes element")

    # Lane offset
    for lane_offset in lanes.findall("laneOffset"):
        parse_opendrive_road_lane_offset(newRoad, lane_offset)

    # Lane sections
    for lane_section_id, lane_section in enumerate(road.find("lanes").findall("laneSection")):
        parse_opendrive_road_lane_section(newRoad, lane_section_id, lane_section)

    # Objects
    # TODO implementation

    # Signals
    # TODO implementation
    calculate_lane_section_lengths(newRoad)

    opendrive.roads.append(newRoad)


def calculate_lane_section_lengths(newRoad):
    """

    Args:
      newRoad:

    """
    # OpenDRIVE does not provide lane section lengths by itself, calculate them by ourselves
    for lane_section in newRoad.lanes.lane_sections:

        # Last lane section in road
        if lane_section.idx + 1 >= len(newRoad.lanes.lane_sections):
            lane_section.length = newRoad.planView.length - lane_section.sPos

        # All but the last lane section end at the succeeding one
        else:
            lane_section.length = (newRoad.lanes.lane_sections[lane_section.idx + 1].sPos - lane_section.sPos)

    # OpenDRIVE does not provide lane width lengths by itself, calculate them by ourselves
    for lane_section in newRoad.lanes.lane_sections:
        for lane in lane_section.allLanes:
            widthsPoses = np.array([x.start_offset for x in lane.widths] + [lane_section.length])
            widthsLengths = widthsPoses[1:] - widthsPoses[:-1]

            for widthIdx, width in enumerate(lane.widths):
                width.length = widthsLengths[widthIdx]


def parse_opendrive_header(opendrive, header):
    """

    Args:
      opendrive:
      header:

    """

    parsed_header = Header(
        header.get("revMajor"),
        header.get("revMinor"),
        header.get("name"),
        header.get("version"),
        header.get("date"),
        header.get("north"),
        header.get("south"),
        header.get("west"),
        header.get("vendor"),
    )
    # Reference
    if header.find("geoReference") is not None:
        pass
        # TODO not implemented

    opendrive.header = parsed_header


def parse_opendrive_junction(opendrive, junction):
    """

    Args:
      opendrive:
      junction:

    """
    newJunction = Junction()

    newJunction.id = int(junction.get("id"))
    newJunction.name = str(junction.get("name"))

    for connection in junction.findall("connection"):

        newConnection = JunctionConnection()

        newConnection.id = connection.get("id")
        newConnection.incomingRoad = connection.get("incomingRoad")
        newConnection.connectingRoad = connection.get("connectingRoad")
        newConnection.contactPoint = connection.get("contactPoint")

        for laneLink in connection.findall("laneLink"):
            newLaneLink = JunctionConnectionLaneLink()

            newLaneLink.fromId = laneLink.get("from")
            newLaneLink.toId = laneLink.get("to")

            newConnection.addLaneLink(newLaneLink)

        newJunction.addConnection(newConnection)

    opendrive.junctions.append(newJunction)
