class MetaDriveType:
    """
    Following waymo style, this class defines a set of strings used to denote different types of objects.
    Those types are used within MetaDrive and might mismatch to the strings used in other dataset.
    """

    # ===== Lane, Line, Road =====
    LANE_CENTER_LINE = "LANE_SURFACE_STREET"
    CONTINUOUS_YELLOW_LINE = "ROAD_LINE_SOLID_SINGLE_YELLOW"
    CONTINUOUS_GREY_LINE = "ROAD_LINE_SOLID_SINGLE_WHITE"
    BROKEN_GREY_LINE = "ROAD_LINE_BROKEN_SINGLE_WHITE"
    BROKEN_YELLOW_LINE = "BROKEN_SINGLE_YELLOW"
    UNKNOWN_LINE = "UNKNOWN_LINE"

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
