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

    ENUM_TO_STR = {
        UNKNOWN: 'UNKNOWN',
        LANE_FREEWAY: 'LANE_FREEWAY',
        LANE_SURFACE_STREET: 'LANE_SURFACE_STREET',
        LANE_BIKE_LANE: 'LANE_BIKE_LANE'
    }

    def __getitem__(self, item):
        return self.ENUM_TO_STR[item]

    def is_lane(self, type):
        if type in self.ENUM_TO_STR.values():
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

    def is_road_line(self, line):
        return True if line in self.ENUM_TO_STR.values() else False

    def is_yellow(self, line):
        return True if line in [
            self.ENUM_TO_STR[t] for t in [
                RoadLineTypeClass.SOLID_DOUBLE_YELLOW, RoadLineTypeClass.PASSING_DOUBLE_YELLOW, RoadLineTypeClass.
                SOLID_SINGLE_YELLOW, RoadLineTypeClass.BROKEN_DOUBLE_YELLOW, RoadLineTypeClass.BROKEN_SINGLE_YELLOW
            ]
        ] else False

    def is_broken(self, line):
        return True if line in [
            self.ENUM_TO_STR[t] for t in [
                RoadLineTypeClass.BROKEN_DOUBLE_YELLOW, RoadLineTypeClass.BROKEN_SINGLE_YELLOW,
                RoadLineTypeClass.BROKEN_SINGLE_WHITE
            ]
        ] else False

    def __getitem__(self, item):
        return self.ENUM_TO_STR[item]


class RoadEdgeTypeClass:
    UNKNOWN = 0
    # Physical road boundary that doesn't have traffic on the other side (e.g., a curb or the k-rail on the right side of a freeway).
    BOUNDARY = 1
    # Physical road boundary that separates the car from other traffic (e.g. a k-rail or an island).
    MEDIAN = 2

    ENUM_TO_STR = {UNKNOWN: 'UNKNOWN', BOUNDARY: 'ROAD_EDGE_BOUNDARY', MEDIAN: 'ROAD_EDGE_MEDIAN'}

    def is_road_edge(self, edge):
        return True if edge in self.ENUM_TO_STR.values() else False

    def is_sidewalk(self, edge):
        return True if edge == self.ENUM_TO_STR[RoadEdgeTypeClass.BOUNDARY] else False

    def __getitem__(self, item):
        return self.ENUM_TO_STR[item]


class AgentTypeClass:
    UNSET = 0
    VEHICLE = 1
    PEDESTRIAN = 2
    CYCLIST = 3
    OTHER = 4

    ENUM_TO_STR = {UNSET: 'UNSET', VEHICLE: 'VEHICLE', PEDESTRIAN: 'PEDESTRIAN', CYCLIST: 'CYCLIST', OTHER: 'OTHER'}

    def __getitem__(self, item):
        return self.ENUM_TO_STR[item]

    def is_vehicle(self, type):
        return True if type == self.ENUM_TO_STR[self.VEHICLE] else False


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
