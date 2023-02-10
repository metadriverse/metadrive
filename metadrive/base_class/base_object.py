import logging
from typing import Dict

import math
import numpy as np
import seaborn as sns
from panda3d.bullet import BulletWorld, BulletBodyNode
from panda3d.core import LVector3, NodePath, PandaNode

from metadrive.base_class.base_runnable import BaseRunnable
from metadrive.constants import ObjectState
from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.core.physics_world import PhysicsWorld
from metadrive.engine.physics_node import BaseRigidBodyNode
from metadrive.utils import Vector
from metadrive.utils import get_np_random
from metadrive.utils.coordinates_shift import panda_position, metadrive_position, panda_heading, metadrive_heading
from metadrive.utils.math_utils import clip
from metadrive.utils.math_utils import norm

logger = logging.getLogger(__name__)


def _clean_a_node_path(node_path):
    if not node_path.isEmpty():
        for sub_node_path in node_path.getChildren():
            _clean_a_node_path(sub_node_path)
    node_path.detachNode()
    node_path.removeNode()


def clear_node_list(node_path_list):
    from metadrive.engine.engine_utils import get_engine
    engine = get_engine()
    for node_path in node_path_list:
        if isinstance(node_path, NodePath):
            _clean_a_node_path(node_path)
            continue

        elif isinstance(node_path, BaseRigidBodyNode):
            # PZH: Note that this line is extremely important!!!
            # It breaks the cycle reference thus we can release nodes!!!
            # It saves Waymo env!!!
            node_path.destroy()

        elif isinstance(node_path, BulletBodyNode):
            pass

        elif isinstance(node_path, PandaNode):
            node_path.removeAllChildren()
            node_path.clearPythonTag(node_path.getName())

        else:
            raise ValueError(node_path)

        if engine is not None:
            engine.physics_world.static_world.remove(node_path)
            engine.physics_world.dynamic_world.remove(node_path)


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
        self.attached = False

    def destroy_node_list(self, bullet_world: BulletWorld):
        for node in self:
            bullet_world.remove(node)
        self.clear()


class BaseObject(BaseRunnable):
    """
    BaseObject is something interacting with game engine. If something is expected to have a body in the world or have
    appearance in the world, it must be a subclass of BaseObject.

    It is created with name/config/randomEngine and can make decision in the world. Besides the random engine can help
    sample some special configs for it ,Properties and parameters in PARAMETER_SPACE of the object are fixed after
    calling __init__().
    """
    MASS = None  # if object has a body, the mass will be set automatically

    def __init__(self, name=None, random_seed=None, config=None, escape_random_seed_assertion=False):
        """
        Config is a static conception, which specified the parameters of one element.
        There parameters doesn't change, such as length of straight road, max speed of one vehicle, etc.
        """
        super(BaseObject, self).__init__(name, random_seed, config)
        if not escape_random_seed_assertion:
            assert random_seed is not None, "Please assign a random seed for {} class.".format(self.class_name)

        # Following properties are available when this object needs visualization and physics property
        self._body = None

        # each element has its node_path to render, physics node are child nodes of it
        self.origin = NodePath(self.name)

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
        color = sns.color_palette("colorblind")
        color.remove(color[2])  # Remove the green and leave it for special vehicle
        idx = get_np_random().randint(len(color))
        rand_c = color[idx]
        self._panda_color = rand_c

        self._node_path_list = []

    def disable_gravity(self):
        self._body.setGravity(LVector3(0, 0, 0))

    @property
    def height(self):
        return self.origin.getPos()[-1]

    @property
    def panda_color(self):
        return self._panda_color

    def add_body(self, physics_body):
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

            self._node_path_list.append(self.origin)

            self.origin = new_origin
            self.dynamic_nodes.append(physics_body)
            if self.MASS is not None:
                assert isinstance(self.MASS,
                                  int) or isinstance(self.MASS, float), "MASS should be a float or an integer"
                self._body.setMass(self.MASS)
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
        Load to world from memory
        """
        if self.render:
            # double check :-)
            assert isinstance(self.origin, NodePath), "No render model on node_path in this Element"
            self.origin.reparentTo(parent_node_path)
        self.dynamic_nodes.attach_to_physics_world(physics_world.dynamic_world)
        self.static_nodes.attach_to_physics_world(physics_world.static_world)
        logger.debug("{} is attached to the world.".format(type(self)))

    def detach_from_world(self, physics_world: PhysicsWorld):
        """
        It is not fully remove, it will be left in memory. if this element is useless in the future, call Func delete()
        """
        if self.origin is not None:
            self.origin.detachNode()
        self.dynamic_nodes.detach_from_physics_world(physics_world.dynamic_world)
        self.static_nodes.detach_from_physics_world(physics_world.static_world)
        logger.debug("{} is detached from the world.".format(type(self)))

    def destroy(self):
        """
        Fully delete this element and release the memory
        """
        try:
            from metadrive.engine.engine_utils import get_engine
        except ImportError:
            pass
        else:
            engine = get_engine()

            if engine is not None:
                self.detach_from_world(engine.physics_world)
                if self._body is not None and hasattr(self.body, "object"):
                    self.body.generated_object = None
                if self.origin is not None:
                    self.origin.removeNode()

                self.dynamic_nodes.destroy_node_list(bullet_world=engine.physics_world.dynamic_world)
                self.static_nodes.destroy_node_list(bullet_world=engine.physics_world.static_world)

            clear_node_list(self._node_path_list)

            logger.debug("Finish cleaning {} node path.".format(len(self._node_path_list)))
            self._node_path_list.clear()
            self._node_path_list = []

            self.dynamic_nodes.clear()
            self.static_nodes.clear()
            self._config.clear()

    def set_position(self, position, height=0.543):
        """
        Set this object to a place, the default value is the regular height for red car
        :param position: 2d array or list
        """
        self.origin.setPos(panda_position(position, height))

    @property
    def position(self):
        return metadrive_position(self.origin.getPos())

    def set_velocity(self, direction: list, value=None, in_local_frame=False):
        """
        Set velocity for object including the direction of velocity and the value (speed)
        The direction of velocity will be normalized automatically, value decided its scale
        :param direction: 2d array or list
        :param value: speed [m/s]
        :param in_local_frame: True, apply speed to local fram
        """
        if in_local_frame:
            from metadrive.engine.engine_utils import get_engine
            engine = get_engine()
            direction = LVector3(*direction, 0.)
            direction[1] *= -1
            ret = engine.worldNP.getRelativeVector(self.origin, direction)
            direction = [-ret[1], -ret[0]]
        if value is not None:
            norm_ratio = value / (norm(direction[0], direction[1]) + 1e-6)
        else:
            norm_ratio = 1
        self._body.setLinearVelocity(
            LVector3(direction[0] * norm_ratio, -direction[1] * norm_ratio,
                     self._body.getLinearVelocity()[-1])
        )

    @property
    def velocity(self):
        """
        Velocity, unit: m/s
        """
        velocity = self.body.get_linear_velocity()
        return np.asarray([velocity[0], -velocity[1]])

    @property
    def speed(self):
        """
        return the speed in m/s
        """
        velocity = self.body.get_linear_velocity()
        speed = norm(velocity[0], velocity[1])
        return clip(speed, 0.0, 100000.0)

    def set_heading_theta(self, heading_theta, rad_to_degree=True) -> None:
        """
        Set heading theta for this object
        :param heading_theta: float
        :param in_rad: when set to True transfer to degree automatically
        """
        h = panda_heading(heading_theta)
        if rad_to_degree:
            h = h * 180 / np.pi
        self.origin.setH(h)

    @property
    def heading_theta(self):
        """
        Get the heading theta of this object, unit [rad]
        :return:  heading in rad
        """
        return metadrive_heading(self.origin.getH()) / 180 * math.pi

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
        Return the roll of this object
        """
        return self.origin.getR()

    def set_roll(self, roll):
        self.origin.setR(roll)

    @property
    def pitch(self):
        """
        Return the pitch of this object
        """
        return self.origin.getP()

    def set_pitch(self, pitch):
        self.origin.setP(pitch)

    def set_static(self, flag):
        self.body.setStatic(flag)

    def get_panda_pos(self):
        return self.origin.getPos()

    def set_panda_pos(self, pos):
        self.origin.setPos(pos)

    def get_state(self) -> Dict:
        state = {
            ObjectState.POSITION: self.get_panda_pos(),
            ObjectState.HEADING_THETA: self.heading_theta,
            ObjectState.ROLL: self.roll,
            ObjectState.PITCH: self.pitch,
            ObjectState.VELOCITY: self.velocity,
        }
        return state

    def set_state(self, state: Dict):
        self.set_panda_pos(state[ObjectState.POSITION])
        self.set_heading_theta(state[ObjectState.HEADING_THETA])
        self.set_pitch(state[ObjectState.PITCH])
        self.set_roll(state[ObjectState.ROLL])
        self.set_velocity(state[ObjectState.VELOCITY] / 3.6)

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
