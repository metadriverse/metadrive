from typing import List, Tuple
from panda3d.core import Vec3

from panda3d.bullet import BulletWorld
from panda3d.core import Vec4, BitMask32

PG_EDITION = "PGDrive v0.1.4"
DEFAULT_AGENT = "default_agent"
RENDER_MODE_NONE = "none"  # Do not render
RENDER_MODE_ONSCREEN = "onscreen"  # Pop up a window and draw image in it
RENDER_MODE_OFFSCREEN = "offscreen"  # Draw image in buffer and collect image from memory


class TerminationState:
    SUCCESS = "arrive_dest"
    OUT_OF_ROAD = "out_of_road"
    MAX_STEP = "max_step"
    CRASH = "crash"
    CRASH_VEHICLE = "crash_vehicle"
    CRASH_OBJECT = "crash_object"
    CRASH_BUILDING = "crash_building"


HELP_MESSAGE = "Keyboard Shortcuts:\n" \
               "  W: Acceleration\n" \
               "  S: Braking\n" \
               "  A: Moving Left\n" \
               "  D: Moving Right\n" \
               "  R: Reset the Environment\n" \
               "  H: Help Message\n" \
               "  F: Switch FPS between unlimited and realtime\n" \
               "  Q: Track Another Vehicle\n" \
               "  B: Top-down View Camera (control: WASD-=)\n" \
               "  +: Lift Camera\n" \
               "  -: Lower Camera\n" \
               "  Esc: Quit\n"

DEBUG_MESSAGE = "  1: Box Debug Mode\n" \
                "  2: WireFrame Debug Mode\n" \
                "  3: Texture Debug Mode\n" \
                "  4: Print Node Message\n"

# priority and color
COLLISION_INFO_COLOR = dict(
    red=(0, Vec4(195 / 255, 0, 0, 1)),
    orange=(1, Vec4(218 / 255, 80 / 255, 0, 1)),
    yellow=(2, Vec4(218 / 255, 163 / 255, 0, 1)),
    green=(3, Vec4(65 / 255, 163 / 255, 0, 1))
)


class BodyName:
    White_continuous_line = "White Continuous Line"
    Yellow_continuous_line = "Yellow Continuous Line"
    Broken_line = "Broken Line"
    Sidewalk = "Sidewalk"
    Ground = "Ground"
    InvisibleWall = "InvisibleWall"
    Base_vehicle = "Target Vehicle"
    Base_vehicle_beneath = "Target Vehicle Beneath"
    Traffic_vehicle = "Traffic Vehicle"
    Lane = "Lane"
    Traffic_cone = "Traffic Cone"
    Traffic_triangle = "Traffic Triangle"
    TollGate = "Toll Gate"


COLOR = {
    BodyName.Sidewalk: "red",
    BodyName.White_continuous_line: "orange",
    BodyName.Yellow_continuous_line: "red",
    BodyName.Broken_line: "yellow",
    BodyName.Traffic_vehicle: "red",
    BodyName.Traffic_cone: "orange",
    BodyName.Traffic_triangle: "orange",
    BodyName.Base_vehicle: "red",
    BodyName.Base_vehicle_beneath: "red",
    BodyName.InvisibleWall: "red",
    BodyName.TollGate: "red",
}


class Decoration:
    """
    Decoration lane didn't connect any nodes, they are individual or isolated.
    """
    start = "decoration"
    end = "decoration_"


class Goal:
    """
    Goal at intersection
    The keywords 0, 1, 2 should be reserved, and only be used in roundabout and intersection
    """

    RIGHT = 0
    STRAIGHT = 1
    LEFT = 2
    ADVERSE = 3  # Useless now


class CamMask:
    MainCam = BitMask32.bit(9)
    Shadow = BitMask32.bit(10)
    RgbCam = BitMask32.bit(11)
    MiniMap = BitMask32.bit(12)
    PARA_VIS = BitMask32.bit(13)
    DepthCam = BitMask32.bit(14)
    ScreenshotCam = BitMask32.bit(15)


class CollisionGroup:
    Terrain = 2
    EgoVehicle = 1
    EgoVehicleBeneath = 6
    BrokenLaneLine = 3
    ContinuousLaneLine = 7
    InvisibleWall = 8
    TrafficVehicle = 4
    LaneSurface = 5  # useless now, since it is in another physics world

    @classmethod
    def collision_rules(cls):
        return [
            # terrain collision
            (cls.Terrain, cls.Terrain, False),
            (cls.Terrain, cls.BrokenLaneLine, False),
            (cls.Terrain, cls.LaneSurface, False),
            (cls.Terrain, cls.EgoVehicle, True),
            (cls.Terrain, cls.EgoVehicleBeneath, False),
            (cls.Terrain, cls.TrafficVehicle, False),
            (cls.Terrain, cls.ContinuousLaneLine, False),
            (cls.Terrain, cls.InvisibleWall, False),

            # block collision
            (cls.BrokenLaneLine, cls.BrokenLaneLine, False),
            (cls.BrokenLaneLine, cls.LaneSurface, False),
            (cls.BrokenLaneLine, cls.EgoVehicle, False),
            (cls.BrokenLaneLine, cls.EgoVehicleBeneath, True),
            # change it after we design a new traffic system !
            (cls.BrokenLaneLine, cls.TrafficVehicle, False),
            (cls.BrokenLaneLine, cls.ContinuousLaneLine, False),
            (cls.BrokenLaneLine, cls.InvisibleWall, False),

            # traffic vehicles collision
            (cls.TrafficVehicle, cls.TrafficVehicle, False),
            (cls.TrafficVehicle, cls.LaneSurface, False),
            (cls.TrafficVehicle, cls.EgoVehicle, True),
            (cls.TrafficVehicle, cls.EgoVehicleBeneath, True),
            (cls.TrafficVehicle, cls.ContinuousLaneLine, False),
            # FIXME maybe set to True in the future
            (cls.TrafficVehicle, cls.InvisibleWall, False),

            # ego vehicle collision
            (cls.EgoVehicle, cls.EgoVehicle, True),
            (cls.EgoVehicle, cls.EgoVehicleBeneath, False),
            (cls.EgoVehicle, cls.LaneSurface, False),
            (cls.EgoVehicle, cls.ContinuousLaneLine, False),
            (cls.EgoVehicle, cls.InvisibleWall, True),

            # lane surface
            (cls.LaneSurface, cls.LaneSurface, False),
            (cls.LaneSurface, cls.EgoVehicleBeneath, False),
            (cls.LaneSurface, cls.ContinuousLaneLine, False),
            (cls.LaneSurface, cls.InvisibleWall, False),

            # vehicle beneath
            (cls.EgoVehicleBeneath, cls.EgoVehicleBeneath, True),
            (cls.EgoVehicleBeneath, cls.ContinuousLaneLine, True),
            (cls.EgoVehicleBeneath, cls.InvisibleWall, True),

            # continuous lane line
            (cls.ContinuousLaneLine, cls.ContinuousLaneLine, False),
            (cls.ContinuousLaneLine, cls.InvisibleWall, False),
        ]

    @classmethod
    def set_collision_rule(cls, world: BulletWorld):
        for rule in cls.collision_rules():
            world.setGroupCollisionFlag(*rule)


LaneIndex = Tuple[str, str, int]
Route = List[LaneIndex]
TARGET_VEHICLES = "target_vehicles"
TRAFFIC_VEHICLES = "traffic_vehicles"
OBJECT_TO_AGENT = "object_to_agent"
AGENT_TO_OBJECT = "agent_to_object"
BKG_COLOR = Vec3(179 / 255, 211 / 255, 216 / 255)


class LineType:
    """A lane side line type."""

    NONE = "none"
    BROKEN = "broken"
    CONTINUOUS = "continuous"
    SIDE = "side"

    @staticmethod
    def prohibit(line_type) -> bool:
        if line_type in [LineType.CONTINUOUS, LineType.SIDE]:
            return True
        else:
            return False


class LineColor:
    GREY = (1, 1, 1, 1)
    YELLOW = (245 / 255, 192 / 255, 67 / 255, 1)


class DrivableAreaProperty:
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

    # for detection
    LANE_LINE_GHOST_HEIGHT = 0.4

    # lane line collision group
    CONTINUOUS_COLLISION_MASK = CollisionGroup.ContinuousLaneLine
    BROKEN_COLLISION_MASK = CollisionGroup.BrokenLaneLine

    # for creating complex block, for example Intersection and roundabout consist of 4 part, which contain several road
    PART_IDX = 0
    ROAD_IDX = 0
    DASH = "_"

    #  when set to True, Vehicles will not generate on this block
    PROHIBIT_TRAFFIC_GENERATION = False