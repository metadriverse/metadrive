from pgdrive.constants import CollisionGroup
from pgdrive.scene_creator.lane.abs_lane import LineType


class BlockDefault:
    CENTER_LINE_TYPE = LineType.CONTINUOUS

    # road network property
    ID = None  # each block must have a unique ID
    SOCKET_NUM = None

    # visualization size property
    CIRCULAR_SEGMENT_LENGTH = 4
    STRIPE_LENGTH = 1.5
    LANE_LINE_WIDTH = 0.15
    LANE_LINE_THICKNESS = 0.01

    SIDEWALK_THICKNESS = 0.4
    SIDEWALK_LENGTH = 3
    SIDEWALK_WIDTH = 3
    SIDEWALK_LINE_DIST = 0.6

    # visualization color property
    LAND_COLOR = (0.4, 0.4, 0.4, 1)
    NAVI_COLOR = (0.709, 0.09, 0, 1)

    # lane line collision group
    LANE_LINE_COLLISION_MASK = CollisionGroup.LaneLine

    # for creating complex block, for example Intersection and roundabout consist of 4 part, which contain several road
    PART_IDX = 0
    ROAD_IDX = 0
    DASH = "_"

    #  when set to True, Vehicles will not generate on this block
    PROHIBIT_TRAFFIC_GENERATION = False
