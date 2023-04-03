class MetaDriveType:
    """
    Following waymo style, this class defines a set of strings used to denote different types of objects.
    Those types are used within MetaDrive and might mismatch to the strings used in other dataset.

    NOTE: when add new keys, make sure class method works well for them
    """

    # ===== Lane, Road =====
    LANE_SURFACE_STREET = "LANE_SURFACE_STREET"
    LANE_UNKNOWN = "LANE_UNKNOWN"
    LANE_FREEWAY = "LANE_FREEWAY"
    LANE_BIKE_LANE = "LANE_BIKE_LANE"

    # ===== Lane Line =====
    LINE_UNKNOWN = "UNKNOWN_LINE"
    LINE_BROKEN_SINGLE_WHITE = "ROAD_LINE_BROKEN_SINGLE_WHITE"
    LINE_SOLID_SINGLE_WHITE = "ROAD_LINE_SOLID_SINGLE_WHITE"
    LINE_SOLID_DOUBLE_WHITE = "ROAD_LINE_SOLID_DOUBLE_WHITE"
    LINE_BROKEN_SINGLE_YELLOW = "ROAD_LINE_BROKEN_SINGLE_YELLOW"
    LINE_BROKEN_DOUBLE_YELLOW = "ROAD_LINE_BROKEN_DOUBLE_YELLOW"
    LINE_SOLID_SINGLE_YELLOW = "ROAD_LINE_SOLID_SINGLE_YELLOW"
    LINE_SOLID_DOUBLE_YELLOW = "ROAD_LINE_SOLID_DOUBLE_YELLOW"
    LINE_PASSING_DOUBLE_YELLOW = "ROAD_LINE_PASSING_DOUBLE_YELLOW"

    # ===== Edge/Boundary/SideWalk =====
    BOUNDARY_UNKNOWN = "UNKNOWN"
    BOUNDARY_LINE = "ROAD_EDGE_BOUNDARY"
    BOUNDARY_MEDIAN = "ROAD_EDGE_MEDIAN"

    # ===== Traffic Light =====
    LANE_STATE_UNKNOWN = "LANE_STATE_UNKNOWN"
    LANE_STATE_ARROW_STOP = "LANE_STATE_ARROW_STOP"
    LANE_STATE_ARROW_CAUTION = "LANE_STATE_ARROW_CAUTION"
    LANE_STATE_ARROW_GO = "LANE_STATE_ARROW_GO"
    LANE_STATE_STOP = "LANE_STATE_STOP"
    LANE_STATE_CAUTION = "LANE_STATE_CAUTION"
    LANE_STATE_GO = "LANE_STATE_GO"
    LANE_STATE_FLASHING_STOP = "LANE_STATE_FLASHING_STOP"
    LANE_STATE_FLASHING_CAUTION = "LANE_STATE_FLASHING_CAUTION"

    # ===== Agent type =====
    UNSET = "UNSET"
    VEHICLE = "VEHICLE"
    PEDESTRIAN = "PEDESTRIAN"
    CYCLIST = "CYCLIST"
    OTHER = "OTHER"

    # ===== Object type =====
    TRAFFIC_LIGHT = "TRAFFIC_LIGHT"
    TRAFFIC_OBJECT = "TRAFFIC_OBJECT"
    GROUND = "GROUND"
    INVISIBLE_WALL = "INVISIBLE_WALL"
    BUILDING = "BUILDING"

    # ===== Coordinate system =====
    COORDINATE_METADRIVE = "metadrive"
    COORDINATE_WAYMO = "waymo"

    @classmethod
    def has_type(cls, type_string: str):
        return type_string in cls.__dict__

    @classmethod
    def from_waymo(cls, waymo_type_string: str):
        assert cls.__dict__[waymo_type_string]
        return waymo_type_string

    @classmethod
    def from_nuplan(cls, waymo_type_string: str):
        # TODO: WIP
        return ""

    @classmethod
    def is_lane(cls, type):
        return type in [cls.LANE_SURFACE_STREET, cls.LANE_FREEWAY, cls.LANE_BIKE_LANE]

    @classmethod
    def is_road_line(cls, line):
        """
        This function relates to is_road_edge. We will have different processing when treating a line that
        is in the boundary or not.
        """
        return line in [
            cls.LINE_UNKNOWN, cls.LINE_BROKEN_SINGLE_WHITE, cls.LINE_SOLID_SINGLE_WHITE, cls.LINE_SOLID_DOUBLE_WHITE,
            cls.LINE_BROKEN_SINGLE_YELLOW, cls.LINE_BROKEN_DOUBLE_YELLOW, cls.LINE_SOLID_SINGLE_YELLOW,
            cls.LINE_SOLID_DOUBLE_YELLOW, cls.LINE_PASSING_DOUBLE_YELLOW
        ]

    @classmethod
    def is_yellow_line(cls, line):
        return line in [
            cls.LINE_SOLID_DOUBLE_YELLOW, cls.LINE_PASSING_DOUBLE_YELLOW, cls.LINE_SOLID_SINGLE_YELLOW,
            cls.LINE_BROKEN_DOUBLE_YELLOW, cls.LINE_BROKEN_SINGLE_YELLOW
        ]

    @classmethod
    def is_broken_line(cls, line):
        return line in [cls.LINE_BROKEN_DOUBLE_YELLOW, cls.LINE_BROKEN_SINGLE_YELLOW, cls.LINE_BROKEN_SINGLE_WHITE]

    @classmethod
    def is_road_edge(cls, edge):
        """
        This function relates to is_road_line.
        """
        return edge in [cls.BOUNDARY_UNKNOWN, cls.BOUNDARY_LINE, cls.BOUNDARY_MEDIAN]

    @classmethod
    def is_sidewalk(cls, edge):
        return edge == cls.BOUNDARY_LINE

    @classmethod
    def is_vehicle(cls, type):
        return type == cls.VEHICLE

    @classmethod
    def is_traffic_light_in_yellow(cls, light):
        return light in [cls.LANE_STATE_CAUTION, cls.LANE_STATE_ARROW_CAUTION, cls.LANE_STATE_FLASHING_CAUTION]

    @classmethod
    def is_traffic_light_in_green(cls, light):
        return light in [cls.LANE_STATE_GO, cls.LANE_STATE_ARROW_GO]

    @classmethod
    def is_traffic_light_in_red(cls, light):
        return light in [cls.LANE_STATE_STOP, cls.LANE_STATE_ARROW_STOP, cls.LANE_STATE_FLASHING_STOP]


class TrafficLightStatus:
    GREEN = 1
    RED = 2
    YELLOW = 3
    UNKNOWN = 4

    @classmethod
    def semantics(self, status):
        if status == self.GREEN:
            return "Traffic Light: Green"
        if status == self.RED:
            return "Traffic Light: Red"
        if status == self.YELLOW:
            return "Traffic Light: Yellow"
        if status == self.UNKNOWN:
            return "Traffic Light: Unknown"

    @classmethod
    def color(self, status):
        if status == self.GREEN:
            return [0, 255, 0]
        if status == self.RED:
            return [1, 255, 0]
        if status == self.YELLOW:
            return [255, 255, 0]
        if status == self.UNKNOWN:
            return [180, 180, 180]
