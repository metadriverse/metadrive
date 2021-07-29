import logging
from typing import Dict, Union

from panda3d.bullet import BulletWorld
from panda3d.core import NodePath

from pgdrive.engine.asset_loader import AssetLoader
from pgdrive.engine.core.pg_physics_world import PGPhysicsWorld
from pgdrive.utils.pg_config import PGConfig
from pgdrive.utils.pg_space import PGSpace
from pgdrive.utils.random import random_string, RandomEngine


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


class BaseObject(RandomEngine):
    """
    Properties and parameters in PARAMETER_SPACE of the object are fixed after calling init().

    They have no any desire to change its state, e.g. moving, bouncing, changing color.
    Instead, only other Elements or DynamicElements can affect them and change their states.
    """

    PARAMETER_SPACE = PGSpace({})

    def __init__(self, name=None, random_seed=None):
        """
        Config is a static conception, which specified the parameters of one element.
        There parameters doesn't change, such as length of straight road, max speed of one vehicle, etc.
        """

        assert random_seed is not None, "Please assign a random seed for {} class in super().__init__()".format(
            self.class_name
        )
        super(BaseObject, self).__init__(random_seed)

        # ID for object
        self.name = random_string() if name is None else name
        self.id = self.name  # name = id

        # Parameter check
        assert isinstance(
            self.PARAMETER_SPACE, PGSpace
        ), "Using PGSpace to define parameter spaces of " + self.class_name

        # initialize and specify the value in config
        self._config = PGConfig({k: None for k in self.PARAMETER_SPACE.parameters})
        self.sample_parameters()

        # each element has its node_path to render, physics node are child nodes of it
        self.node_path = None

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

    def get_config(self, copy=True) -> Union[PGConfig, Dict]:
        """
        Return self._config
        :param copy:
        :return: a copy of config dict
        """
        if copy:
            return self._config.copy()
        return self._config

    def set_config(self, config: dict):
        """
        Merge config and self._config
        """
        self._config.update(config)

    def sample_parameters(self):
        """
        Fix a value of the random parameters in PARAMETER_SPACE
        """
        random_seed = self.np_random.randint(low=0, high=int(1e6))
        self.PARAMETER_SPACE.seed(random_seed)
        ret = self.PARAMETER_SPACE.sample()
        self.set_config(ret)

    def get_state(self) -> Dict:
        """
        Store current state, such as heading, position, etc. to store the movement and change history trajectory
        :return: state dict
        """
        raise NotImplementedError

    def set_state(self, state: Dict):
        """
        Set the position, heading and so on to restore or load state
        :param state: dict
        """
        raise NotImplementedError

    def before_step(self, *args, **kwargs):
        """
        Do Information fusion and then analyze and wait for decision
        """
        pass

    def set_action(self, *args, **kwargs):
        """
        Set action for this object, and the action will last for the minimal simulation interval
        """
        raise NotImplementedError

    def after_step(self, *args, **kwargs):
        """
        After advancing all elements for a time period, their state should be updated for statistic or other purpose
        """

    def reset(self, *args, **kwargs):
        """
        Although some elements won't reset, please still state this function in it :)
        """
        raise NotImplementedError

    def attach_to_world(self, parent_node_path: NodePath, pg_physics_world: PGPhysicsWorld):
        if self.render:
            # double check :-)
            assert isinstance(self.node_path, NodePath), "No render model on node_path in this Element"
            self.node_path.reparentTo(parent_node_path)
        self.dynamic_nodes.attach_to_physics_world(pg_physics_world.dynamic_world)
        self.static_nodes.attach_to_physics_world(pg_physics_world.static_world)

    def detach_from_world(self, pg_physics_world: PGPhysicsWorld):
        """
        It is not fully remove, if this element is useless in the future, call Func delete()
        """
        if self.node_path is not None:
            self.node_path.detachNode()
        self.dynamic_nodes.detach_from_physics_world(pg_physics_world.dynamic_world)
        self.static_nodes.detach_from_physics_world(pg_physics_world.static_world)

    def destroy(self):
        """
        Fully delete this element and release the memory
        """
        from pgdrive.utils.engine_utils import get_pgdrive_engine
        engine = get_pgdrive_engine()
        self.detach_from_world(engine.physics_world)
        if self.node_path is not None:
            self.node_path.removeNode()
        self.dynamic_nodes.clear()
        self.static_nodes.clear()
        self._config.clear()

    @property
    def class_name(self):
        return self.__class__.__name__

    def __del__(self):
        try:
            str(self)
        except AttributeError:
            pass
        else:
            logging.debug("{} is destroyed".format(str(self)))

    def __repr__(self):
        return "{}".format(str(self))

    def __str__(self):
        return "{}, ID:{}".format(self.class_name, self.name)

    @property
    def config(self):
        return self.get_config(True)
