class MetaDriveType:
    """
    Following waymo style, this class defines a set of strings used to denote different types of objects.
    Those types are used within MetaDrive and might mismatch to the strings used in other dataset.
    """

    LANE_CENTER_LINE = "LANE_SURFACE_STREET"
    CONTINUOUS_YELLOW_LINE = "ROAD_LINE_SOLID_SINGLE_YELLOW"
    CONTINUOUS_GREY_LINE = "ROAD_LINE_SOLID_SINGLE_WHITE"
    BROKEN_GREY_LINE = "ROAD_LINE_BROKEN_SINGLE_WHITE"
    BROKEN_YELLOW_LINE = "BROKEN_SINGLE_YELLOW"
    UNSET = 'UNSET'
    VEHICLE = 'VEHICLE'
    PEDESTRIAN = 'PEDESTRIAN'
    CYCLIST = 'CYCLIST'
    OTHER = 'OTHER'

    @classmethod
    def from_waymo(cls, waymo_type_string: str):
        # TODO: WIP
        return ""

    @classmethod
    def from_nuplan(cls, waymo_type_string: str):
        # TODO: WIP
        return ""