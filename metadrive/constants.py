import math
from collections import namedtuple
from typing import List, Tuple

import numpy as np
from panda3d.bullet import BulletWorld
from panda3d.core import Vec3
from panda3d.core import Vec4, BitMask32
from shapely.geometry import Polygon

from metadrive.type import MetaDriveType
from metadrive.version import VERSION

EDITION = "MetaDrive v{}".format(VERSION)
DATA_VERSION = EDITION  # Use MetaDrive version to mark the data version
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
    CRASH_HUMAN = "crash_human"
    CRASH_OBJECT = "crash_object"
    CRASH_BUILDING = "crash_building"
    CRASH_SIDEWALK = "crash_sidewalk"
    CURRENT_BLOCK = "current_block"
    ENV_SEED = "env_seed"
    IDLE = "idle"


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

# Used for rendering the banner in Interface.
COLOR = {
    MetaDriveType.BOUNDARY_LINE: "red",
    MetaDriveType.BOUNDARY_SIDEWALK: "red",
    MetaDriveType.CROSSWALK: "yellow",
    MetaDriveType.LINE_SOLID_SINGLE_WHITE: "orange",
    MetaDriveType.LINE_SOLID_SINGLE_YELLOW: "orange",
    MetaDriveType.LINE_BROKEN_SINGLE_YELLOW: "yellow",
    MetaDriveType.LINE_BROKEN_SINGLE_WHITE: "green",
    MetaDriveType.LANE_SURFACE_STREET: "green",
    MetaDriveType.LANE_SURFACE_UNSTRUCTURE: "green",
    MetaDriveType.LANE_BIKE_LANE: "green",
    MetaDriveType.VEHICLE: "red",
    MetaDriveType.GROUND: "yellow",
    MetaDriveType.TRAFFIC_OBJECT: "yellow",
    MetaDriveType.TRAFFIC_CONE: "yellow",
    MetaDriveType.TRAFFIC_BARRIER: "yellow",
    MetaDriveType.PEDESTRIAN: "red",
    MetaDriveType.CYCLIST: "red",
    MetaDriveType.INVISIBLE_WALL: "red",
    MetaDriveType.BUILDING: "red",
    MetaDriveType.LIGHT_RED: "red",
    MetaDriveType.LIGHT_YELLOW: "orange",
    MetaDriveType.LIGHT_GREEN: "green",
    MetaDriveType.UNSET: "orange",
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
    SemanticCam = BitMask32.bit(16)


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
    TrafficParticipants = BitMask32.bit(10)
    Crosswalk = BitMask32.bit(11)

    @classmethod
    def collision_rules(cls):
        """
        This should be a diagonal matrix
        """
        return [
            # terrain collision
            (cls.Terrain, cls.Terrain, False),
            (cls.Terrain, cls.BrokenLaneLine, False),
            (cls.Terrain, cls.LaneSurface, False),
            (cls.Terrain, cls.Vehicle, True),
            (cls.Terrain, cls.ContinuousLaneLine, False),
            (cls.Terrain, cls.InvisibleWall, False),
            (cls.Terrain, cls.Sidewalk, False),
            (cls.Terrain, cls.LidarBroadDetector, False),
            (cls.Terrain, cls.TrafficObject, True),
            (cls.Terrain, cls.TrafficParticipants, True),
            (cls.Terrain, cls.Crosswalk, False),

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
            (cls.BrokenLaneLine, cls.TrafficParticipants, True),
            (cls.BrokenLaneLine, cls.Crosswalk, False),

            # vehicle contact
            (cls.Vehicle, cls.Vehicle, True),
            (cls.Vehicle, cls.LaneSurface, True),
            (cls.Vehicle, cls.ContinuousLaneLine, True),
            (cls.Vehicle, cls.InvisibleWall, True),
            (cls.Vehicle, cls.Sidewalk, True),
            (cls.Vehicle, cls.LidarBroadDetector, True),
            (cls.Vehicle, cls.TrafficObject, True),
            (cls.Vehicle, cls.TrafficParticipants, True),
            (cls.Vehicle, cls.Crosswalk, True),

            # lane surface
            (cls.LaneSurface, cls.LaneSurface, False),
            (cls.LaneSurface, cls.ContinuousLaneLine, False),
            (cls.LaneSurface, cls.InvisibleWall, False),
            (cls.LaneSurface, cls.Sidewalk, False),
            (cls.LaneSurface, cls.LidarBroadDetector, False),
            (cls.LaneSurface, cls.TrafficObject, True),
            (cls.LaneSurface, cls.TrafficParticipants, True),
            (cls.LaneSurface, cls.Crosswalk, False),

            # continuous lane line
            (cls.ContinuousLaneLine, cls.ContinuousLaneLine, False),
            (cls.ContinuousLaneLine, cls.InvisibleWall, False),
            (cls.ContinuousLaneLine, cls.Sidewalk, False),
            (cls.ContinuousLaneLine, cls.LidarBroadDetector, False),
            (cls.ContinuousLaneLine, cls.TrafficObject, False),
            (cls.ContinuousLaneLine, cls.TrafficParticipants, True),
            (cls.ContinuousLaneLine, cls.Crosswalk, False),

            # invisible wall
            (cls.InvisibleWall, cls.InvisibleWall, False),
            (cls.InvisibleWall, cls.Sidewalk, False),
            (cls.InvisibleWall, cls.LidarBroadDetector, True),
            (cls.InvisibleWall, cls.TrafficObject, False),
            (cls.InvisibleWall, cls.TrafficParticipants, True),
            (cls.InvisibleWall, cls.Crosswalk, False),

            # side walk
            (cls.Sidewalk, cls.Sidewalk, False),
            (cls.Sidewalk, cls.LidarBroadDetector, False),
            (cls.Sidewalk, cls.TrafficObject, True),
            (cls.Sidewalk, cls.TrafficParticipants, True),  # don't allow sidewalk contact
            (cls.Sidewalk, cls.Crosswalk, False),  # don't allow sidewalk contact

            # LidarBroadDetector
            (cls.LidarBroadDetector, cls.LidarBroadDetector, False),
            (cls.LidarBroadDetector, cls.TrafficObject, True),
            (cls.LidarBroadDetector, cls.TrafficParticipants, True),
            (cls.LidarBroadDetector, cls.Crosswalk, False),

            # TrafficObject
            (cls.TrafficObject, cls.TrafficObject, True),
            (cls.TrafficObject, cls.TrafficParticipants, True),
            (cls.TrafficObject, cls.Crosswalk, False),

            # TrafficParticipant
            (cls.TrafficParticipants, cls.TrafficParticipants, True),
            (cls.TrafficParticipants, cls.Crosswalk, True)
        ]

    @classmethod
    def set_collision_rule(cls, world: BulletWorld, disable_collision: bool = False):
        for rule in cls.collision_rules():
            group_1 = int(math.log(rule[0].getWord(), 2))
            group_2 = int(math.log(rule[1].getWord(), 2))
            relation = rule[-1]
            if disable_collision:
                relation = False
            world.setGroupCollisionFlag(group_1, group_2, relation)

    @classmethod
    def can_be_lidar_detected(cls):
        return cls.Vehicle | cls.InvisibleWall | cls.TrafficObject | cls.TrafficParticipants

    # def make_collision_from_model(input_model, world):
    #     # tristrip generation from static models
    #     # generic tri-strip collision generator begins
    #     geom_nodes = input_model.findAllMatches('**/+GeomNode')
    #     geom_nodes = geom_nodes.getPath(0).node()
    #     # print(geom_nodes)
    #     geom_target = geom_nodes.getGeom(0)
    #     # print(geom_target)
    #     output_bullet_mesh = BulletTriangleMesh()
    #     output_bullet_mesh.addGeom(geom_target)
    #     tri_shape = BulletTriangleMeshShape(output_bullet_mesh, dynamic=False)
    #     print(output_bullet_mesh)
    #
    #     body = BulletRigidBodyNode('input_model_tri_mesh')
    #     np = self.render.attachNewNode(body)
    #     np.node().addShape(tri_shape)
    #     np.node().setMass(0)
    #     np.node().setFriction(0.5)
    #     # np.setPos(0, 0, 0)
    #     np.setScale(1)
    #     np.setCollideMask(BitMask32.allOn())
    #     world.attachRigidBody(np.node())
    #
    # make_collision_from_model(access_deck_1, world)  # world = BulletWorld()


LaneIndex = Tuple[str, str, int]
Route = List[LaneIndex]
TARGET_VEHICLES = "target_vehicles"
TRAFFIC_VEHICLES = "traffic_vehicles"
OBJECT_TO_AGENT = "object_to_agent"
AGENT_TO_OBJECT = "agent_to_object"
BKG_COLOR = Vec3(1, 1, 1)


class PGLineType:
    """A lane side line type."""

    NONE = MetaDriveType.LINE_UNKNOWN
    BROKEN = MetaDriveType.LINE_BROKEN_SINGLE_WHITE
    CONTINUOUS = MetaDriveType.LINE_SOLID_SINGLE_WHITE
    SIDE = MetaDriveType.BOUNDARY_LINE
    GUARDRAIL = MetaDriveType.GUARDRAIL

    @staticmethod
    def prohibit(line_type) -> bool:
        if line_type in [PGLineType.CONTINUOUS, PGLineType.SIDE]:
            return True
        else:
            return False


class PGLineColor:
    GREY = (1, 1, 1, 1)
    YELLOW = (255 / 255, 200 / 255, 0 / 255, 1)


class PGDrivableAreaProperty:
    """
    Defining some properties for creating PGMap
    """
    # road network property
    ID = None  # each block must have a unique ID
    SOCKET_NUM = None

    # visualization size property
    LANE_SEGMENT_LENGTH = 4
    STRIPE_LENGTH = 1.5
    LANE_LINE_WIDTH = 0.15
    LANE_LINE_THICKNESS = 0.016

    SIDEWALK_THICKNESS = 0.3
    SIDEWALK_LENGTH = 3
    SIDEWALK_WIDTH = 2
    SIDEWALK_LINE_DIST = 0.6

    GUARDRAIL_HEIGHT = 4.0

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
    # this is for internal recording/replaying system
    POSITION = "position"
    HEADING_THETA = "heading_theta"
    VELOCITY = "velocity"
    PITCH = "pitch"
    ROLL = "roll"
    STATIC = "static"
    CLASS = "type"
    INIT_KWARGS = "config"
    NAME = "name"
    SIZE = "size"
    TYPE = "type"


class PolicyState:
    ARGS = "args"
    KWARGS = "kwargs"
    POLICY_CLASS = "policy_class"
    OBJ_NAME = "obj_name"


REPLAY_DONE = "replay_done"

label_color = namedtuple("label_color", "label color")


class Semantics:
    """
    For semantic camera
    """
    # CitySpace colormap: https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/helpers/labels.py
    UNLABELED = label_color("UNLABELED", (0, 0, 0))
    CAR = label_color("CAR", (0, 0, 142))
    TRUCK = label_color("TRUCK", (0, 0, 70))
    PEDESTRIAN = label_color("PEDESTRIAN", (220, 20, 60))
    BIKE = label_color("BIKE", (119, 11, 32))  # bicycle
    TERRAIN = label_color("TERRAIN", (152, 251, 152))
    ROAD = label_color("ROAD", (128, 64, 128))  # road
    SIDEWALK = label_color("SIDEWALK", (244, 35, 232))
    SKY = label_color("SKY", (70, 130, 180))
    TRAFFIC_LIGHT = label_color("TRAFFIC_LIGHT", (250, 170, 30))
    FENCE = label_color("FENCE", (190, 153, 153))
    TRAFFIC_SIGN = label_color("TRAFFIC_SIGN", (220, 220, 0))

    # customized
    LANE_LINE = label_color("LANE_LINE", (255, 255, 255))
    CROSSWALK = label_color("CROSSWALK", (55, 176, 189))

    # These color might be prettier?
    # LANE_LINE = label_color("LANE_LINE", (128, 64, 128))
    # CROSSWALK = label_color("CROSSWALK", (128, 64, 128))

    BUS = label_color("BUS", (0, 60, 100))  # PZH: I just randomly choose a color.


class MapTerrainSemanticColor:
    """
    Do not modify this as it is for terrain generation. If you want your own palette, just add a new one or modify
    class lMapSemanticColor
    """
    YELLOW = 30
    WHITE = 10

    @staticmethod
    def get_color(type):
        """
        Each channel represents a type. This should be aligned with shader terrain.frag.glsl
        Args:
            type: MetaDriveType

        Returns:

        """
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Note: modify it with shaders together!
        if MetaDriveType.is_yellow_line(type):
            # return (255, 0, 0, 0)
            # return (1, 0, 0, 0)
            return MapTerrainSemanticColor.YELLOW
        elif MetaDriveType.is_lane(type):
            # return (0, 1, 0, 0)
            return 20
        elif type == MetaDriveType.GROUND:
            # return (0, 0, 1, 0)
            return 00
        elif MetaDriveType.is_white_line(type) or MetaDriveType.is_road_boundary_line(type):
            # return (0, 0, 0, 1)
            return MapTerrainSemanticColor.WHITE
        elif type == MetaDriveType.CROSSWALK:
            # The range of crosswalk value is 0.4 <= value < 0.76,
            # so people can save the angle (degree) of the crosswalk in attribute map
            # the value * 10 = angle of crosswalk. It is a trick for saving memory.
            return 40  # this value can be overwritten latter
        else:
            raise ValueError("Unsupported type: {}".format(type))
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # Note: modify it with shaders together!


class TopDownSemanticColor:
    """
    Do not modify this as it is for terrain generation. If you want your own palette, just add a new one or modify
    class lMapSemanticColor
    """
    @staticmethod
    def get_color(type):
        if MetaDriveType.is_lane(type):
            # intersection and others
            if type == MetaDriveType.LANE_SURFACE_UNSTRUCTURE:
                ret = np.array([186, 186, 186])
            # a set of lanes
            else:
                ret = np.array([210, 210, 210])
        # road divider
        elif MetaDriveType.is_yellow_line(type):
            ret = np.array([20, 20, 20])
        # lane divider
        elif MetaDriveType.is_road_boundary_line(type) or MetaDriveType.is_white_line(type):
            ret = np.array([140, 140, 140])
        # vehicle
        elif MetaDriveType.is_vehicle(type):
            ret = np.array([224, 177, 67])
        # construction object
        elif MetaDriveType.is_traffic_object(type):
            ret = np.array([67, 143, 224])
        # human
        elif type == MetaDriveType.PEDESTRIAN:
            ret = np.array([224, 67, 67])
        # cyclist and motorcycle
        elif type == MetaDriveType.CYCLIST:
            ret = np.array([75, 224, 67])
        else:
            ret = np.array([125, 67, 224])
        # else:
        #     raise ValueError("Unsupported type: {}".format(type))
        return ret


class TerrainProperty:
    """
    Define some constants/properties for the map and terrain
    """
    map_region_size = 2048

    @classmethod
    def get_semantic_map_pixel_per_meter(cls):
        """
        Get how many pixels are used to represent one-meter
        Returns: a constant

        """
        # assert cls.terrain_size <= 2048, "Terrain size should be fixed to 2048"
        return 22 if cls.map_region_size != 4096 else 11

    @classmethod
    def point_in_map(cls, point):
        """
        Return if the point is in the map region
        Args:
            map_center: center point of the map
            point: 2D point

        Returns: Boolean

        """
        x_, y_ = point[:2]
        x = y = cls.map_region_size / 2
        return -x <= x_ <= x and -y <= y_ <= y

    @classmethod
    def clip_polygon(cls, polygon):
        """
        Clip the Polygon. Make it fit into the map region and throw away the part outside the map region
        Args:
            map_center: center point of the map
            polygon: a list of 2D points

        Returns: A list of polygon or None

        """
        x = y = cls.map_region_size / 2
        _rect_polygon = Polygon([(-x, y), (x, y), (x, -y), (-x, -y)])
        polygon = Polygon(polygon)
        try:
            polygon = _rect_polygon.intersection(polygon)
            # Extract the points of the clipped polygon.
            if polygon.is_empty:
                return None
            else:
                # Handle cases where the intersection might result in multiple geometries
                return [list(polygon.exterior.coords)] if isinstance(polygon, Polygon) else \
                    [list(geom.exterior.coords) for geom in polygon.geoms]
        except Exception as error:
            return None


class CameraTagStateKey:
    """
    Enables multi-pass rendering
    """
    ID = "id"
    RGB = "rgb"
    Depth = "depth"
    Semantic = "semantic"


DEFAULT_SENSOR_OFFSET = (0., 0.8, 1.5)
DEFAULT_SENSOR_HPR = (0., -5, 0.0)

COLOR_PALETTE = (
    (0.00392156862745098, 0.45098039215686275,
     0.6980392156862745), (0.8705882352941177, 0.5607843137254902, 0.0196078431372549),
    (0.00784313725490196, 0.6196078431372549, 0.45098039215686275), (0.8352941176470589, 0.3686274509803922, 0.0),
    (0.8, 0.47058823529411764, 0.7372549019607844), (0.792156862745098, 0.5686274509803921, 0.3803921568627451),
    (0.984313725490196, 0.6862745098039216,
     0.8941176470588236), (0.5803921568627451, 0.5803921568627451, 0.5803921568627451),
    (0.9254901960784314, 0.8823529411764706, 0.2), (0.33725490196078434, 0.7058823529411765, 0.9137254901960784)
)


def get_color_palette():
    return list(COLOR_PALETTE)
