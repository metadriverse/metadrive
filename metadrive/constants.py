import math
from typing import List, Tuple

from panda3d.bullet import BulletWorld
from panda3d.core import Vec3
from panda3d.core import Vec4, BitMask32

EDITION = "MetaDrive v0.2.6.0"
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
    CURRENT_BLOCK = "current_block"
    ENV_SEED = "env_seed"


HELP_MESSAGE = "Keyboard Shortcuts:\n" \
               "  W: Acceleration\n" \
               "  S: Braking\n" \
               "  A: Moving Left\n" \
               "  D: Moving Right\n" \
               "  R: Reset the Environment\n" \
               "  H: Help Message\n" \
               "  F: Switch FPS to unlimited / realtime\n" \
               "  Q: Third-person View Camera\n" \
               "  B: Top-down View Camera (control: WASD-=)\n" \
               "  +: Lift Camera\n" \
               "  -: Lower Camera\n" \
               "  Mouse click: move camera (top-down view)\n" \
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
    Vehicle = "Vehicle"
    Lane = "Lane"
    Traffic_object = "Traffic Object"
    TollGate = "Toll Gate"


COLOR = {
    BodyName.Sidewalk: "red",
    BodyName.White_continuous_line: "orange",
    BodyName.Yellow_continuous_line: "red",
    BodyName.Broken_line: "yellow",
    BodyName.Vehicle: "red",
    BodyName.Traffic_object: "orange",
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


class Mask:
    AllOn = BitMask32.allOn()
    AllOff = BitMask32.allOff()


class CamMask(Mask):
    MainCam = BitMask32.bit(9)
    Shadow = BitMask32.bit(10)
    RgbCam = BitMask32.bit(11)
    MiniMap = BitMask32.bit(12)
    PARA_VIS = BitMask32.bit(13)
    DepthCam = BitMask32.bit(14)
    ScreenshotCam = BitMask32.bit(15)


class CollisionGroup(Mask):
    Vehicle = BitMask32.bit(1)
    Terrain = BitMask32.bit(2)
    BrokenLaneLine = BitMask32.bit(3)
    TrafficObject = BitMask32.bit(4)
    LaneSurface = BitMask32.bit(5)  # useless now, since it is in another bullet world
    Sidewalk = BitMask32.bit(6)
    ContinuousLaneLine = BitMask32.bit(7)
    InvisibleWall = BitMask32.bit(8)
    LidarBroadDetector = BitMask32.bit(9)

    @classmethod
    def collision_rules(cls):
        return [
            # terrain collision
            (cls.Terrain, cls.Terrain, False),
            (cls.Terrain, cls.BrokenLaneLine, False),
            (cls.Terrain, cls.LaneSurface, False),
            (cls.Terrain, cls.Vehicle, True),
            (cls.Terrain, cls.ContinuousLaneLine, False),
            (cls.Terrain, cls.InvisibleWall, False),
            (cls.Terrain, cls.Sidewalk, True),
            (cls.Terrain, cls.LidarBroadDetector, False),
            (cls.Terrain, cls.TrafficObject, True),

            # block collision
            (cls.BrokenLaneLine, cls.BrokenLaneLine, False),
            (cls.BrokenLaneLine, cls.LaneSurface, False),
            (cls.BrokenLaneLine, cls.Vehicle, True),
            # change it after we design a new traffic system !
            (cls.BrokenLaneLine, cls.ContinuousLaneLine, False),
            (cls.BrokenLaneLine, cls.InvisibleWall, False),
            (cls.BrokenLaneLine, cls.Sidewalk, False),
            (cls.BrokenLaneLine, cls.LidarBroadDetector, False),
            (cls.BrokenLaneLine, cls.TrafficObject, True),

            # ego vehicle collision
            (cls.Vehicle, cls.Vehicle, True),
            (cls.Vehicle, cls.LaneSurface, True),
            (cls.Vehicle, cls.ContinuousLaneLine, True),
            (cls.Vehicle, cls.InvisibleWall, True),
            (cls.Vehicle, cls.Sidewalk, True),
            (cls.Vehicle, cls.LidarBroadDetector, True),
            (cls.Vehicle, cls.TrafficObject, True),

            # lane surface
            (cls.LaneSurface, cls.LaneSurface, False),
            (cls.LaneSurface, cls.ContinuousLaneLine, False),
            (cls.LaneSurface, cls.InvisibleWall, False),
            (cls.LaneSurface, cls.Sidewalk, False),
            (cls.LaneSurface, cls.LidarBroadDetector, False),
            (cls.LaneSurface, cls.TrafficObject, True),

            # continuous lane line
            (cls.ContinuousLaneLine, cls.ContinuousLaneLine, False),
            (cls.ContinuousLaneLine, cls.InvisibleWall, False),
            (cls.ContinuousLaneLine, cls.Sidewalk, False),
            (cls.ContinuousLaneLine, cls.LidarBroadDetector, False),
            (cls.ContinuousLaneLine, cls.TrafficObject, False),

            # invisible wall
            (cls.InvisibleWall, cls.InvisibleWall, False),
            (cls.InvisibleWall, cls.Sidewalk, False),
            (cls.InvisibleWall, cls.LidarBroadDetector, True),
            (cls.InvisibleWall, cls.TrafficObject, False),

            # side walk
            (cls.Sidewalk, cls.Sidewalk, False),
            (cls.Sidewalk, cls.LidarBroadDetector, False),
            (cls.Sidewalk, cls.TrafficObject, True),

            #
            (cls.LidarBroadDetector, cls.LidarBroadDetector, False),
            (cls.LidarBroadDetector, cls.TrafficObject, True),
        ]

    @classmethod
    def set_collision_rule(cls, world: BulletWorld):
        for rule in cls.collision_rules():
            group_1 = int(math.log(rule[0].getWord(), 2))
            group_2 = int(math.log(rule[1].getWord(), 2))
            relation = rule[-1]
            world.setGroupCollisionFlag(group_1, group_2, relation)


LaneIndex = Tuple[str, str, int]
Route = List[LaneIndex]
TARGET_VEHICLES = "target_vehicles"
TRAFFIC_VEHICLES = "traffic_vehicles"
OBJECT_TO_AGENT = "object_to_agent"
AGENT_TO_OBJECT = "agent_to_object"
BKG_COLOR = Vec3(1, 1, 1)


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
    LANE_SEGMENT_LENGTH = 4
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
    LANE_LINE_GHOST_HEIGHT = 1.0

    # lane line collision group
    CONTINUOUS_COLLISION_MASK = CollisionGroup.ContinuousLaneLine
    BROKEN_COLLISION_MASK = CollisionGroup.BrokenLaneLine
    SIDEWALK_COLLISION_MASK = CollisionGroup.Sidewalk

    # for creating complex block, for example Intersection and roundabout consist of 4 part, which contain several road
    PART_IDX = 0
    ROAD_IDX = 0
    DASH = "_"

    #  when set to True, Vehicles will not generate on this block
    PROHIBIT_TRAFFIC_GENERATION = False


class ObjectState:
    POSITION = "position"
    HEADING_THETA = "heading_theta"
    VELOCITY = "velocity"
    PITCH = "pitch"
    ROLL = "roll"
    STATIC = "static"
    CLASS = "type"
    INIT_KWARGS = "config"
    NAME = "name"


class PolicyState:
    ARGS = "args"
    KWARGS = "kwargs"
    POLICY_CLASS = "policy_class"
    OBJ_NAME = "obj_name"


REPLAY_DONE = "replay_done"


class WaymoLaneProperty:
    LANE_TYPE = "center_lane"
    LANE_LINE_TYPE = "road_line"
    LANE_EDGE_TYPE = "road_edge"
    POLYLINE = "polyline"
    LEFT_BOUNDARIES = "left_boundaries"
    RIGHT_BOUNDARIES = "right_boundaries"
    LEFT_NEIGHBORS = "left_neighbor"
    RIGHT_NEIGHBORS = "right_neighbor"
    ENTRY = "entry"
    EXIT = "exit"

    @staticmethod
    def get_line_type_and_line_color(waymo_type):
        pass
