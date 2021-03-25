import copy
import logging
from typing import Dict

from panda3d.bullet import BulletWorld
from panda3d.core import NodePath
from pgdrive.utils import PGConfig
from pgdrive.utils.asset_loader import AssetLoader
from pgdrive.utils.pg_space import PGSpace
from pgdrive.world.pg_physics_world import PGPhysicsWorld


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


class Element:
    """
    Element class are base class of all static objects, whose properties are fixed after calling init().
    They have no any desire to change its state, e.g. moving, bouncing, changing color.
    Instead, only other Elements or DynamicElements can affect them and change their states.
    """

    PARAMETER_SPACE = PGSpace({})

    def __init__(self, random_seed=None):
        """
        Config is a static conception, which specified the parameters of one element.
        There parameters doesn't change, such as length of straight road, max speed of one vehicle, etc.
        """
        assert isinstance(
            self.PARAMETER_SPACE, PGSpace
        ) or random_seed is None, "Using PGSpace to define parameter spaces of " + self.class_name
        self._config = PGConfig({k: None for k in self.PARAMETER_SPACE.parameters})
        self.random_seed = 0 if random_seed is None else random_seed
        if self.PARAMETER_SPACE is not None:
            self.PARAMETER_SPACE.seed(self.random_seed)
        self.render = False if AssetLoader.loader is None else True

        # each element has its node_path to render, physics node are child nodes of it
        self.node_path = None

        # Temporally store bullet nodes that have to place in bullet world (not NodePath)
        self.dynamic_nodes = PhysicsNodeList()

        # Nodes in this tuple didn't interact with other nodes! they only used to do rayTest or sweepTest
        self.static_nodes = PhysicsNodeList()

        if self.render:
            self.loader = AssetLoader.get_loader()

    @property
    def class_name(self):
        return self.__class__.__name__

    def get_config(self):
        assert self._config is not None, "config of " + self.class_name + " is None, can not be read !"
        return copy.copy(self._config)

    def set_config(self, config: dict):
        # logging.debug("Read config to " + self.class_name)
        self._config.update(copy.copy(config))

    def attach_to_pg_world(self, parent_node_path: NodePath, pg_physics_world: PGPhysicsWorld):
        if self.render:
            # double check :-)
            assert isinstance(self.node_path, NodePath), "No render model on node_path in this Element"
            self.node_path.reparentTo(parent_node_path)
        self.dynamic_nodes.attach_to_physics_world(pg_physics_world.dynamic_world)
        self.static_nodes.attach_to_physics_world(pg_physics_world.static_world)

    def detach_from_pg_world(self, pg_physics_world: PGPhysicsWorld):
        """
        It is not fully remove, if this element is useless in the future, call Func delete()
        """
        self.node_path.detachNode()
        self.dynamic_nodes.detach_from_physics_world(pg_physics_world.dynamic_world)
        self.static_nodes.detach_from_physics_world(pg_physics_world.static_world)

    def destroy(self, pg_world):
        """
        Fully delete this element and release the memory
        """
        self.detach_from_pg_world(pg_world.physics_world)
        self.node_path.removeNode()
        self.dynamic_nodes.clear()
        self.static_nodes.clear()
        self._config.clear()

    def __del__(self):
        logging.debug("{} is destroyed".format(self.class_name))


class DynamicElement(Element):
    def __init__(self, np_random=None):
        """
        State is a runtime conception used to create a snapshot of scenario at one moment.
        A scenario can be saved to file and recovered to debug or something else.
        The difference between config and state is that state is changed as time goes by.
        Don't mix the two conception.

        It's interesting that sometimes config == state when time-step=1, such as current-vehicle.
        And sometimes config == state in the whole simulation episode, such as radius of a curve block.
        To avoid this, only derive from this class for elements who can do step().
        """
        super(DynamicElement, self).__init__(np_random)

    def get_state(self):
        raise NotImplementedError

    def set_state(self, state: Dict):
        """
        Override it in Dynamical element
        :param state: dict
        :return: None
        """
        raise NotImplementedError

    def prepare_step(self, *args, **kwargs):
        """
        Do Information fusion and then analyze and make decision
        """

    def step(self, *args, **kwargs):
        """
        Implement decision and advance the PGWorld
        Although some elements won't step, please still state this function in it :)
        """
        raise NotImplementedError

    def update_state(self, *args, **kwargs):
        """
        After advancing all elements for a time period, their state should be updated for statistic or other purpose
        """

    def reset(self, *args, **kwargs):
        """
        Although some elements won't reset, please still state this function in it :)
        """
        raise NotImplementedError
