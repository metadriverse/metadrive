import copy
import logging
from typing import Dict

from panda3d.bullet import BulletWorld
from panda3d.core import NodePath

from pg_drive.pg_config.pg_config import PgConfig
from pg_drive.pg_config.pg_space import PgSpace
from pg_drive.utils.visualization_loader import VisLoader


class Element:
    PARAMETER_SPACE = PgSpace({})

    def __init__(self, random_seed=None):
        """
        Config is a static conception, which specified the parameters of one element.
        There parameters doesn't change, such as length of straight road, max speed of one vehicle, etc.
        """
        assert isinstance(
            self.PARAMETER_SPACE, PgSpace
        ) or random_seed is None, "Using PgSpace to define parameter spaces of " + self.class_name
        self._config = PgConfig({k: None for k in self.PARAMETER_SPACE.parameters})
        self.random_seed = 0 if random_seed is None else random_seed
        if self.PARAMETER_SPACE is not None:
            self.PARAMETER_SPACE.seed(self.random_seed)
        self.bullet_nodes = []  # Temporally store bullet nodes that have to place in bullet world (not NodePath)
        self.render = False if VisLoader.loader is None else True
        self.node_path = None  # each element has its node_path to render, physics node are child nodes of it
        if self.render:
            self.loader = VisLoader.get_loader()

    @property
    def class_name(self):
        return self.__class__.__name__

    def get_config(self):
        assert self._config is not None, "config of " + self.class_name + " is None, can not be read !"
        return copy.copy(self._config)

    def set_config(self, config: dict):
        # logging.debug("Read config to " + self.class_name)
        self._config.update(copy.copy(config))

    def add_to_render_module(self, parent_render_np: NodePath):
        """
        Call this to render bodies
        """
        if self.render:
            # double check :-)
            assert isinstance(self.node_path, NodePath), "No render model on node_path in this Element"
            self.node_path.reparentTo(parent_render_np)

    def add_to_physics_world(self, bt_world: BulletWorld):
        """
        Call this to add body to physics world
        """
        for node in self.bullet_nodes:
            bt_world.attach(node)

    def remove_from_physics_world(self, bt_physics_world: BulletWorld):
        """
        Call me to remove element from bullet physics world
        """
        for node in self.bullet_nodes:
            bt_physics_world.remove(node)

    def remove_from_render_module(self):
        """
        It is not fully remove, if this element is useless in the future, call Func delete()
        """
        self.node_path.detachNode()

    def destroy(self, bt_physics_world: BulletWorld):
        """
        Fully delete this element and release the memory
        """
        self.node_path.removeNode()
        self.remove_from_physics_world(bt_physics_world)
        self.bullet_nodes.clear()
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
        self._state = None

    def get_state(self):
        assert self._state is not None, "state of " + self.class_name + " is None, can not be read !"
        return copy.copy(self._state)

    def set_state(self, state: Dict):
        logging.debug("Read state to " + self.class_name)
        self._state = copy.copy(state)

    def step(self, *args, **kwargs):
        """
        Although some elements won't step, please still state this function in it :)
        """
        raise NotImplementedError

    def reset(self, *args, **kwargs):
        """
        Although some elements won't reset, please still state this function in it :)
        """
        raise NotImplementedError
