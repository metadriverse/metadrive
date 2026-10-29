import copy
import math
from abc import ABC
from typing import Dict

import numpy as np
from panda3d.bullet import BulletWorld, BulletBodyNode, BulletVehicle
from panda3d.core import LVector3, NodePath, PandaNode

from metadrive.base_class.base_runnable import BaseRunnable
from metadrive.constants import ObjectState
from metadrive.constants import Semantics, CameraTagStateKey, get_color_palette
from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.core.physics_world import PhysicsWorld
from metadrive.engine.logger import get_logger
from metadrive.engine.physics_node import BaseRigidBodyNode, BaseGhostBodyNode
from metadrive.type import MetaDriveType
from metadrive.utils import Vector
from metadrive.utils import get_np_random
from metadrive.utils import random_string
from metadrive.utils.coordinates_shift import panda_vector, metadrive_vector, panda_heading
from metadrive.utils.math import clip
from metadrive.utils.math import norm
from metadrive.utils.math import wrap_to_pi

logger = get_logger()


def _clean_a_node_path(node_path):
    if not node_path.isEmpty():
        for sub_node_path in node_path.getChildren():
            _clean_a_node_path(sub_node_path)
    node_path.detachNode()
    node_path.removeNode()


def clear_node_list(node_path_list):
    for node_path in node_path_list:
        if isinstance(node_path, NodePath):
            _clean_a_node_path(node_path)
            continue

        elif isinstance(node_path, BaseRigidBodyNode):
            # PZH: Note that this line is extremely important!!!
            # It breaks the cycle reference thus we can release nodes!!!
            # It saves Waymo env!!!
            node_path.destroy()

        elif isinstance(node_path, BaseGhostBodyNode):
            # PZH: Note that this line is extremely important!!!
            # It breaks the cycle reference thus we can release nodes!!!
            # It saves Waymo env!!!
            node_path.destroy()

        elif isinstance(node_path, PandaNode):
            node_path.removeAllChildren()
            node_path.clearPythonTag(node_path.getName())

        else:
            raise ValueError(node_path)


class PhysicsNodeList(list):
    def __init__(self):
        super(PhysicsNodeList, self).__init__()
        self.attached = False

    def attach_to_physics_world(self, bullet_world: BulletWorld):
        """
        Attach the nodes in this list to bullet world
        :param bullet_world: BulletWorld()
        :return: None
        """
        if self.attached:
            return
        for node in self:
            bullet_world.attach(node)
        self.attached = True

    def detach_from_physics_world(self, bullet_world: BulletWorld):
        """
         Detach the nodes in this list from bullet world
         :param bullet_world: BulletWorld()
         :return: None
         """
        if not self.attached:
            return
        for node in self:
            bullet_world.remove(node)
            if isinstance(node, BulletVehicle):
                break
        self.attached = False

    def destroy_node_list(self):
        for node in self:
            if isinstance(node, BaseGhostBodyNode) or isinstance(node, BaseRigidBodyNode):
                node.destroy()
            if isinstance(node, BulletBodyNode):
                node.removeAllChildren()
        self.clear()


class BaseObject(BaseRunnable, MetaDriveType, ABC):
    """
    BaseObject is something interacting with game engine. If something is expected to have a body in the world or have
    appearance in the world, it must be a subclass of BaseObject.

    It is created with name/config/randomEngine and can make decision in the world. Besides the random engine can help
    sample some special configs for it ,Properties and parameters in PARAMETER_SPACE of the object are fixed after
    calling __init__().
    """
    MASS = None  # if object has a body, the mass will be set automatically
    COLLISION_MASK = None
    SEMANTIC_LABEL = Semantics.UNLABELED.label

    def __init__(self, name=None, random_seed=None, config=None, escape_random_seed_assertion=False):
        """
        Config is a static conception, which specified the parameters of one element.
        There parameters doesn't change, such as length of straight road, max speed of one vehicle, etc.
        """
        config = copy.deepcopy(config)
        BaseRunnable.__init__(self, name, random_seed, config)
        MetaDriveType.__init__(self)
        if not escape_random_seed_assertion:
            assert random_seed is not None, "Please assign a random seed for {} class.".format(self.class_name)

        # Following properties are available when this object needs visualization and physics property
        self._body = None

        # each element has its node_path to render, physics node are child nodes of it
        self.origin = NodePath(self.name)

        # semantic color
        self.origin.setTag(CameraTagStateKey.Semantic, self.SEMANTIC_LABEL)

        # Temporally store bullet nodes that have to place in bullet world (not NodePath)
        self.dynamic_nodes = PhysicsNodeList()

        # Nodes in this tuple didn't interact with other nodes! they only used to do rayTest or sweepTest
        self.static_nodes = PhysicsNodeList()

        # render or not
        self.render = False if AssetLoader.loader is None else True
        if self.render:
            self.loader = AssetLoader.get_loader()

            if not hasattr(self.loader, "loader"):
                # It is closed before!
                self.loader.__init__()

        # add color setting for visualization
        color = get_color_palette()
        color.pop(2)  # Remove the green and leave it for special vehicle
        idx = get_np_random().randint(len(color))
        rand_c = color[idx]
        self._panda_color = rand_c

        # store all NodePath reparented to this node
        self._node_path_list = []

        # debug
        self.coordinates_debug_np = None
        self.need_show_coordinates = False

    def disable_gravity(self):
        self._body.setGravity(LVector3(0, 0, 0))

    @property
    def height(self):
        return self.origin.getPos()[-1]

    @property
    def panda_color(self):
        return self._panda_color

    def add_body(self, physics_body, add_to_static_world=False):
        if self._body is None:
            # add it to physics world, in which this object will interact with other object (like collision)
            if not isinstance(physics_body, BulletBodyNode):
                raise ValueError("The physics body is not BulletBodyNode type")
            self._body = physics_body
            new_origin = NodePath(self._body)
            new_origin.setH(self.origin.getH())
            new_origin.setPos(self.origin.getPos())
            self.origin.getChildren().reparentTo(new_origin)

            # TODO(PZH): We don't call this sentence:. It might cause problem since we don't remove old origin?
            # self.origin.removeNode()

            if self.COLLISION_MASK is not None:
                self._body.setIntoCollideMask(self.COLLISION_MASK)

            self._node_path_list.append(self.origin)
            self.origin = new_origin
            self.origin.setTag(CameraTagStateKey.Semantic, self.SEMANTIC_LABEL)
            if add_to_static_world:
                self.static_nodes.append(physics_body)
            else:
                self.dynamic_nodes.append(physics_body)
            if self.MASS is not None:
                assert isinstance(self.MASS,
                                  int) or isinstance(self.MASS, float), "MASS should be a float or an integer"
                self._body.setMass(self.MASS)

            if self.engine is not None and self.engine.global_config["show_coordinates"]:
                self.need_show_coordinates = True
                self.show_coordinates()
        else:
            raise AttributeError("You can not set the object body for twice")

    @property
    def body(self):
        if self._body.hasPythonTag(self._body.getName()):
            return self._body.getPythonTag(self._body.getName())
        else:
            return self._body

    def attach_to_world(self, parent_node_path: NodePath, physics_world: PhysicsWorld):
        """
        Load the object to the world from memory, attach the object to the scene graph.
        Args:
            parent_node_path: which parent node to attach
            physics_world: PhysicsWorld, engine.physics_world

        Returns: None

        """
        if not self.is_attached():
            assert isinstance(self.origin, NodePath), "No render model on node_path in this Element"
            self.origin.reparentTo(parent_node_path)
            self.dynamic_nodes.attach_to_physics_world(physics_world.dynamic_world)
            self.static_nodes.attach_to_physics_world(physics_world.static_world)
            logger.debug("{} is attached to the world.".format(self.class_name))
        else:
            logger.debug("Can not attach object {} to world, as it is already attached!".format(self.class_name))

    def detach_from_world(self, physics_world: PhysicsWorld):
        """
        It is not fully remove, it will be left in memory. if this element is useless in the future, call Func destroy()
        Detach the object from the scene graph but store it in the memory
        Args:
            physics_world: PhysicsWorld, engine.physics_world

        Returns: None

        """
        if self.is_attached():
            self.origin.detachNode()
            self.dynamic_nodes.detach_from_physics_world(physics_world.dynamic_world)
            self.static_nodes.detach_from_physics_world(physics_world.static_world)
            logger.debug("{} is detached from the world.".format(self.class_name))
        else:
            logger.debug("Object {} is already detached from the world. Can not detach again".format(self.class_name))

    def is_attached(self):
        return self.origin is not None and self.origin.hasParent()

    def destroy(self):
        """
        Fully delete this element and release the memory
        """
        super(BaseObject, self).destroy()
        try:
            from metadrive.engine.engine_utils import get_engine
        except ImportError:
            pass
        else:
            engine = get_engine()
            if engine is not None:
                if self.is_attached():
                    self.detach_from_world(engine.physics_world)
                if self._body is not None and hasattr(self.body, "object"):
                    self.body.generated_object = None
                if self.origin is not None:
                    self.origin.removeNode()

                self.dynamic_nodes.destroy_node_list()
                self.static_nodes.destroy_node_list()

            clear_node_list(self._node_path_list)

            logger.debug("Finish cleaning {} node path for {}.".format(len(self._node_path_list), self.class_name))
            self._node_path_list.clear()
            self._node_path_list = []

            self.dynamic_nodes.clear()
            self.static_nodes.clear()

    def set_position(self, position, height=None):
        """
        Set this object to a place, the default value is the regular height for red car
        :param position: 2d array or list
        :param height: give a fixed height
        """
        assert len(position) == 2 or len(position) == 3
        if len(position) == 3:
            height = position[-1]
            position = position[:-1]
        else:
            if height is None:
                height = self.origin.getPos()[-1]
        self.origin.setPos(panda_vector(position, height))

    @property
    def position(self):
        return metadrive_vector(self.origin.getPos())

    def set_velocity(self, direction: np.array, value=None, in_local_frame=False):
        """
        Set velocity for object including the direction of velocity and the value (speed)
        The direction of velocity will be normalized automatically, value decided its scale
        :param direction: 2d array or list
        :param value: speed [m/s]
        :param in_local_frame: True, apply speed to local fram
        """
        if in_local_frame:
            direction = self.convert_to_world_coordinates(direction, [0, 0])

        if value is not None:
            norm_ratio = value / (norm(direction[0], direction[1]) + 1e-6)
        else:
            norm_ratio = 1
        self._body.setLinearVelocity(
            LVector3(direction[0] * norm_ratio, direction[1] * norm_ratio,
                     self._body.getLinearVelocity()[-1])
        )

    def set_velocity_km_h(self, direction: list, value=None, in_local_frame=False):
        direction = np.array(direction)
        if value is None:
            direction /= 3.6
        else:
            value /= 3.6
        return self.set_velocity(direction, value, in_local_frame)

    @property
    def velocity(self):
        """
        Velocity, unit: m/s
        """
        if isinstance(self.body, BaseGhostBodyNode):
            return np.array([0, 0])
        else:
            velocity = self.body.get_linear_velocity()
            return np.asarray([velocity[0], velocity[1]])

    @property
    def velocity_km_h(self):
        """
        Velocity, unit: m/s
        """
        return self.velocity * 3.6

    @property
    def speed(self):
        """
        return the speed in m/s
        """
        velocity = self.body.get_linear_velocity()
        speed = norm(velocity[0], velocity[1])
        return clip(speed, 0.0, 100000.0)

    @property
    def speed_km_h(self):
        """
        km/h
        """
        velocity = self.body.get_linear_velocity()
        speed = norm(velocity[0], velocity[1]) * 3.6
        return clip(speed, 0.0, 100000.0)

    def set_heading_theta(self, heading_theta, in_rad=True) -> None:
        """
        Set heading theta for this object
        :param heading_theta: float
        :param in_rad: when set to True, heading theta should be in rad, otherwise, in degree
        """
        h = panda_heading(heading_theta)
        if in_rad:
            h = h * 180 / np.pi
        self.origin.setH(h)

    @property
    def heading_theta(self):
        """
        Get the heading theta of this object, unit [rad]
        :return:  heading in rad
        """
        return wrap_to_pi(self.origin.getH() / 180 * math.pi)

    @property
    def heading(self):
        """
        Heading is a vector = [cos(heading_theta), sin(heading_theta)]
        """
        real_heading = self.heading_theta
        # heading = np.array([math.cos(real_heading), math.sin(real_heading)])
        heading = Vector((math.cos(real_heading), math.sin(real_heading)))
        return heading

    @property
    def roll(self):
        """
        Return the roll of this object. As it is facing to x, so roll is pitch
        """
        return np.deg2rad(self.origin.getP())

    def set_roll(self, roll):
        """
        As it is facing to x, so roll is pitch
        """
        self.origin.setP(roll)

    @property
    def pitch(self):
        """
        Return the pitch of this object, as it is facing to x, so pitch is roll
        """
        return np.deg2rad(self.origin.getR())

    def set_pitch(self, pitch):
        """As it is facing to x, so pitch is roll"""
        self.origin.setR(pitch)

    def set_static(self, flag):
        self.body.setStatic(flag)

    def get_panda_pos(self):
        raise DeprecationWarning("It is not allowed to access Panda Pos!")
        return self.origin.getPos()

    def set_panda_pos(self, pos):
        raise DeprecationWarning("It is not allowed to access Panda Pos!")
        self.origin.setPos(pos)

    def get_state(self) -> Dict:
        pos = self.position
        state = {
            ObjectState.POSITION: [pos[0], pos[1], self.get_z()],
            ObjectState.HEADING_THETA: self.heading_theta,
            ObjectState.ROLL: self.roll,
            ObjectState.PITCH: self.pitch,
            ObjectState.VELOCITY: self.velocity,
            ObjectState.TYPE: type(self)
        }
        return state

    def set_state(self, state: Dict):
        self.set_position(state[ObjectState.POSITION])
        self.set_heading_theta(state[ObjectState.HEADING_THETA])
        self.set_pitch(state[ObjectState.PITCH])
        self.set_roll(state[ObjectState.ROLL])
        self.set_velocity(state[ObjectState.VELOCITY])

    @property
    def top_down_color(self):
        rand_c = self.panda_color
        return rand_c[0] * 255, rand_c[1] * 255, rand_c[2] * 255

    @property
    def top_down_width(self):
        raise NotImplementedError(
            "Implement this func for rendering class {} in top down renderer".format(self.class_name)
        )

    @property
    def top_down_length(self):
        raise NotImplementedError(
            "Implement this func for rendering class {} in top down renderer".format(self.class_name)
        )

    def show_coordinates(self):
        pass

    def get_z(self):
        return self.origin.getZ()

    def set_angular_velocity(self, angular_velocity, in_rad=True):
        if not in_rad:
            angular_velocity = angular_velocity / 180 * np.pi
        self._body.setAngularVelocity(LVector3(0, 0, angular_velocity))

    def rename(self, new_name):
        super(BaseObject, self).rename(new_name)
        physics_node = self._body.getPythonTag(self._body.getName())
        if isinstance(physics_node, BaseGhostBodyNode) or isinstance(physics_node, BaseRigidBodyNode):
            physics_node.rename(new_name)

    def random_rename(self):
        self.rename(random_string())

    def convert_to_local_coordinates(self, vector, origin):
        """
        Give vector in world coordinates, and convert it to object coordinates. For example, vector can be other vehicle
        position, origin could be this vehicles position. In this case, vector-origin will be transformed to ego car
        coordinates. If origin is set to 0, then no offset is applied and this API only calculates relative direction.

        In a word, for calculating **points transformation** in different coordinates, origin is required. This is
        because vectors have no origin but origin is required to define a point.
        """
        vector = np.asarray(vector) - np.asarray(origin)
        vector = LVector3(*vector, 0.)
        vector = self.origin.getRelativeVector(self.engine.origin, vector)
        project_on_x = vector[0]
        project_on_y = vector[1]
        return np.array([project_on_x, project_on_y])

    def convert_to_world_coordinates(self, vector, origin):
        """
        Give a vector in local coordinates, and convert it to world coordinates. The origin should be added as offset.
        For example, vector could be a relative position in local coordinates and origin could be ego car's position.
        If origin is set to 0, then no offset is applied and this API only calculates relative direction.

        In a word, for calculating **points transformation** in different coordinates, origin is required. This is
        because vectors have no origin but origin is required to define a point.
        """
        assert len(vector) == 2 or len(vector) == 3, "the vector should be in shape (2,) or (3,)"
        vector = vector[:2]
        vector = LVector3(*vector, 0.)
        vector = self.engine.origin.getRelativeVector(self.origin, vector)
        project_on_x = vector[0]
        project_on_y = vector[1]
        return np.array([project_on_x, project_on_y]) + np.asarray(origin)

    @property
    def WIDTH(self):
        raise NotImplementedError()

    @property
    def LENGTH(self):
        raise NotImplementedError()

    @property
    def bounding_box(self):
        """
        This function will return the 2D bounding box of vehicle. Points are in clockwise sequence, first point is the
        top-left point.
        """
        p1 = self.convert_to_world_coordinates([self.LENGTH / 2, self.WIDTH / 2], self.position)
        p2 = self.convert_to_world_coordinates([self.LENGTH / 2, -self.WIDTH / 2], self.position)
        p3 = self.convert_to_world_coordinates([-self.LENGTH / 2, -self.WIDTH / 2], self.position)
        p4 = self.convert_to_world_coordinates([-self.LENGTH / 2, self.WIDTH / 2], self.position)
        return [p1, p2, p3, p4]

    @property
    def use_render_pipeline(self):
        """
        Return if we are using render_pipeline
        Returns: Boolean

        """
        return self.engine is not None and self.engine.use_render_pipeline
