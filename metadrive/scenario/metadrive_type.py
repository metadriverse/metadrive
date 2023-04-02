class MetaDriveType:
    """
    Following waymo style, this class defines a set of strings used to denote different types of objects.
    Those types are used within MetaDrive and might mismatch to the strings used in other dataset.

    NOTE: when add new keys, make sure class method works well for them
    """

    # ===== Lane, Road =====
    LANE_CENTER_LINE = "LANE_SURFACE_STREET"
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
    TRAFFIC_CONE = "TRAFFIC_CONE"

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
        if type in [cls.LANE_CENTER_LINE, cls.LANE_FREEWAY, cls.LANE_BIKE_LANE]:
            return True
        else:
            return False

    @classmethod
    def is_road_line(cls, line):
        return True if line in [cls.LINE_UNKNOWN,
                                cls.LINE_BROKEN_SINGLE_WHITE,
                                cls.LINE_SOLID_SINGLE_WHITE,
                                cls.LINE_SOLID_DOUBLE_WHITE,
                                cls.LINE_BROKEN_SINGLE_YELLOW,
                                cls.LINE_BROKEN_DOUBLE_YELLOW,
                                cls.LINE_SOLID_SINGLE_YELLOW,
                                cls.LINE_SOLID_DOUBLE_YELLOW,
                                cls.LINE_PASSING_DOUBLE_YELLOW] else False

    @classmethod
    def is_yellow(cls, line):
        return True if line in [
            cls.LINE_SOLID_DOUBLE_YELLOW, cls.LINE_PASSING_DOUBLE_YELLOW, cls.LINE_SOLID_SINGLE_YELLOW,
            cls.LINE_BROKEN_DOUBLE_YELLOW, cls.LINE_BROKEN_SINGLE_YELLOW
        ] else False

    @classmethod
    def is_broken(cls, line):
        return True if line in [
            cls.LINE_BROKEN_DOUBLE_YELLOW, cls.LINE_BROKEN_SINGLE_YELLOW,
            cls.LINE_BROKEN_SINGLE_WHITE
        ] else False

    @classmethod
    def is_road_edge(cls, edge):
        return True if edge in [cls.BOUNDARY_UNKNOWN,
                                cls.BOUNDARY_LINE,
                                cls.BOUNDARY_MEDIAN] else False

    @classmethod
    def is_sidewalk(cls, edge):
        return True if edge == cls.BOUNDARY_LINE else False

    @classmethod
    def is_vehicle(cls, type):
        return True if type == cls.VEHICLE else False
