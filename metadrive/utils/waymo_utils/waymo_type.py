TrafficSignal = {
    0: 'LANE_STATE_UNKNOWN',
    1: 'LANE_STATE_ARROW_STOP',
    2: 'LANE_STATE_ARROW_CAUTION',
    3: 'LANE_STATE_ARROW_GO',
    4: 'LANE_STATE_STOP',
    5: 'LANE_STATE_CAUTION',
    6: 'LANE_STATE_GO',
    7: 'LANE_STATE_FLASHING_STOP',
    8: 'LANE_STATE_FLASHING_CAUTION'
}


class LaneTypeClass:
    UNKNOWN = 0
    LANE_FREEWAY = 1
    LANE_SURFACE_STREET = 2
    LANE_BIKE_LANE = 3

    def __getitem__(self, item):
        return {0: 'UNKNOWN', 1: 'LANE_FREEWAY', 2: 'LANE_SURFACE_STREET', 3: 'LANE_BIKE_LANE'}[item]

    @staticmethod
    def is_lane(type):
        if type in {'LANE_FREEWAY', 'LANE_SURFACE_STREET', 'UNKNOWN', 'LANE_BIKE_LANE'}:
            return True
        else:
            return False


class RoadLineTypeClass:
    UNKNOWN = 0
    BROKEN_SINGLE_WHITE = 1
    SOLID_SINGLE_WHITE = 2
    SOLID_DOUBLE_WHITE = 3
    BROKEN_SINGLE_YELLOW = 4
    BROKEN_DOUBLE_YELLOW = 5
    SOLID_SINGLE_YELLOW = 6
    SOLID_DOUBLE_YELLOW = 7
    PASSING_DOUBLE_YELLOW = 8

    @staticmethod
    def is_road_line(line):
        return True if line.__class__ == RoadLineTypeClass else False

    @staticmethod
    def is_yellow(line):
        return True if line in [
            RoadLineTypeClass.SOLID_DOUBLE_YELLOW, RoadLineTypeClass.PASSING_DOUBLE_YELLOW,
            RoadLineTypeClass.SOLID_SINGLE_YELLOW,
            RoadLineTypeClass.BROKEN_DOUBLE_YELLOW, RoadLineTypeClass.BROKEN_SINGLE_YELLOW
        ] else False

    @staticmethod
    def is_broken(line):
        return True if line in [
            RoadLineTypeClass.BROKEN_DOUBLE_YELLOW, RoadLineTypeClass.BROKEN_SINGLE_YELLOW,
            RoadLineTypeClass.BROKEN_SINGLE_WHITE
        ] else False

    def __getitem__(self, item):
        return {
            RoadLineTypeClass.UNKNOWN: 'UNKNOWN',
            RoadLineTypeClass.BROKEN_SINGLE_WHITE: 'ROAD_LINE_BROKEN_SINGLE_WHITE',
            RoadLineTypeClass.SOLID_SINGLE_WHITE: 'ROAD_LINE_SOLID_SINGLE_WHITE',
            RoadLineTypeClass.SOLID_DOUBLE_WHITE: 'ROAD_LINE_DOUBLE_WHITE',
            RoadLineTypeClass.BROKEN_SINGLE_YELLOW: 'ROAD_LINE_BROKEN_SINGLE_YELLOW',
            RoadLineTypeClass.BROKEN_DOUBLE_YELLOW: 'ROAD_LINE_BROKEN_DOUBLE_YELLOW',
            RoadLineTypeClass.SOLID_SINGLE_YELLOW: 'ROAD_LINE_SOLID_SINGLE_YELLOW',
            RoadLineTypeClass.SOLID_DOUBLE_YELLOW: 'ROAD_LINE_SOLID_DOUBLE_YELLOW',
            RoadLineTypeClass.PASSING_DOUBLE_YELLOW: 'ROAD_LINE_PASSING_DOUBLE_YELLOW'
        }[item]


class RoadEdgeTypeClass:
    UNKNOWN = 0
    # Physical road boundary that doesn't have traffic on the other side (e.g., a curb or the k-rail on the right side of a freeway).
    BOUNDARY = 1
    # Physical road boundary that separates the car from other traffic (e.g. a k-rail or an island).
    MEDIAN = 2

    @staticmethod
    def is_road_edge(edge):
        return True if edge.__class__ == RoadEdgeTypeClass else False

    @staticmethod
    def is_sidewalk(edge):
        return True if edge == RoadEdgeTypeClass.BOUNDARY else False

    def __getitem__(self, item):
        return {RoadEdgeTypeClass.UNKNOWN: 'UNKNOWN',
                RoadEdgeTypeClass.BOUNDARY: 'ROAD_EDGE_BOUNDARY',
                RoadEdgeTypeClass.MEDIAN: 'ROAD_EDGE_MEDIAN'}[item]


class AgentTypeClass:
    UNSET = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    CYCLIST = 3
    OTHER = 4

    def __getitem__(self, item):
        return {AgentTypeClass.UNSET: 'UNSET',
                AgentTypeClass.VEHICLE: 'VEHICLE',
                AgentTypeClass.PEDESTRIAN: 'PEDESTRIAN',
                AgentTypeClass.CYCLIST: 'CYCLIST',
                AgentTypeClass.OTHER: 'OTHER'}[item]


LaneType = LaneTypeClass()
AgentType = AgentTypeClass()
RoadLineType = RoadLineTypeClass()
RoadEdgeType = RoadEdgeTypeClass()


class WaymoLaneProperty:
    LANE_TYPE = "center_lane"
    LANE_LINE_TYPE = "road_line"
    LANE_EDGE_TYPE = "road_edge"
    POLYLINE = "polyline"
    LEFT_BOUNDARIES = "left_boundaries"
    RIGHT_BOUNDARIES = "right_boundaries"
    LEFT_NEIGHBORS = "left_neighbor"
    RIGHT_NEIGHBORS = "right_neighbor"
    ENTRY = "entry_lanes"
    EXIT = "exit_lanes"

    @staticmethod
    def get_line_type_and_line_color(waymo_type):
        pass
