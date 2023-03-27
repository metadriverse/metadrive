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


class LaneType:
    UNKNOWN = 0
    LANE_FREEWAY = 1
    LANE_SURFACE_STREET = 2
    LANE_BIKE_LANE = 3

    ENUM_TO_STR = {
        UNKNOWN: 'UNKNOWN',
        LANE_FREEWAY: 'LANE_FREEWAY',
        LANE_SURFACE_STREET: 'LANE_SURFACE_STREET',
        LANE_BIKE_LANE: 'LANE_BIKE_LANE'
    }

    @classmethod
    def is_lane(cls, type):
        if type in cls.ENUM_TO_STR.values():
            return True
        else:
            return False

    @classmethod
    def from_waymo(cls, item):
        return cls.ENUM_TO_STR[item]


class RoadLineType:
    UNKNOWN = 0
    BROKEN_SINGLE_WHITE = 1
    SOLID_SINGLE_WHITE = 2
    SOLID_DOUBLE_WHITE = 3
    BROKEN_SINGLE_YELLOW = 4
    BROKEN_DOUBLE_YELLOW = 5
    SOLID_SINGLE_YELLOW = 6
    SOLID_DOUBLE_YELLOW = 7
    PASSING_DOUBLE_YELLOW = 8

    ENUM_TO_STR = {
        UNKNOWN: 'UNKNOWN',
        BROKEN_SINGLE_WHITE: 'ROAD_LINE_BROKEN_SINGLE_WHITE',
        SOLID_SINGLE_WHITE: 'ROAD_LINE_SOLID_SINGLE_WHITE',
        SOLID_DOUBLE_WHITE: 'ROAD_LINE_SOLID_DOUBLE_WHITE',
        BROKEN_SINGLE_YELLOW: 'ROAD_LINE_BROKEN_SINGLE_YELLOW',
        BROKEN_DOUBLE_YELLOW: 'ROAD_LINE_BROKEN_DOUBLE_YELLOW',
        SOLID_SINGLE_YELLOW: 'ROAD_LINE_SOLID_SINGLE_YELLOW',
        SOLID_DOUBLE_YELLOW: 'ROAD_LINE_SOLID_DOUBLE_YELLOW',
        PASSING_DOUBLE_YELLOW: 'ROAD_LINE_PASSING_DOUBLE_YELLOW'
    }

    @classmethod
    def is_road_line(cls, line):
        return True if line in cls.ENUM_TO_STR.values() else False

    @classmethod
    def is_yellow(cls, line):
        return True if line in [
            cls.ENUM_TO_STR[t] for t in [
                RoadLineType.SOLID_DOUBLE_YELLOW,
                RoadLineType.PASSING_DOUBLE_YELLOW,
                RoadLineType.SOLID_SINGLE_YELLOW,
                RoadLineType.BROKEN_DOUBLE_YELLOW,
                RoadLineType.BROKEN_SINGLE_YELLOW
            ]
        ] else False

    @classmethod
    def is_broken(cls, line):
        return True if line in [
            cls.ENUM_TO_STR[t] for t in [
                RoadLineType.BROKEN_DOUBLE_YELLOW,
                RoadLineType.BROKEN_SINGLE_YELLOW,
                RoadLineType.BROKEN_SINGLE_WHITE
            ]
        ] else False

    @classmethod
    def from_waymo(cls, item):
        return cls.ENUM_TO_STR[item]


class RoadEdgeType:
    UNKNOWN = 0
    # Physical road boundary that doesn't have traffic on the other side (e.g., a curb or the k-rail on the right side of a freeway).
    BOUNDARY = 1
    # Physical road boundary that separates the car from other traffic (e.g. a k-rail or an island).
    MEDIAN = 2

    ENUM_TO_STR = {UNKNOWN: 'UNKNOWN', BOUNDARY: 'ROAD_EDGE_BOUNDARY', MEDIAN: 'ROAD_EDGE_MEDIAN'}

    @classmethod
    def is_road_edge(cls, edge):
        return True if edge in cls.ENUM_TO_STR.values() else False

    @classmethod
    def is_sidewalk(cls, edge):
        return True if edge == cls.ENUM_TO_STR[RoadEdgeType.BOUNDARY] else False

    @classmethod
    def from_waymo(cls, item):
        return cls.ENUM_TO_STR[item]


class AgentType:
    UNSET = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    CYCLIST = 3
    OTHER = 4

    ENUM_TO_STR = {UNSET: 'UNSET', VEHICLE: 'VEHICLE', PEDESTRIAN: 'PEDESTRIAN', CYCLIST: 'CYCLIST', OTHER: 'OTHER'}

    @classmethod
    def from_waymo(cls, item):
        return cls.ENUM_TO_STR[item]

    @classmethod
    def is_vehicle(self, type):
        return True if type == self.ENUM_TO_STR[self.VEHICLE] else False


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

