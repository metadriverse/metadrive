from panda3d.bullet import BulletWorld, BulletBodyNode
from panda3d.core import NodePath

from pgdrive.base_class.base_runnable import BaseRunnable
from pgdrive.engine.asset_loader import AssetLoader
from pgdrive.engine.core.physics_world import PhysicsWorld


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


class BaseObject(BaseRunnable):
    """
    BaseObject is something interacting with game engine. If something is expected to have a body in the world or have
    appearance in the world, it must be a subclass of BaseObject.

    It is created with name/config/randomEngine and can make decision in the world. Besides the random engine can help
    sample some special configs for it ,Properties and parameters in PARAMETER_SPACE of the object are fixed after
    calling __init__().
    """
    def __init__(self, name=None, random_seed=None, config=None):
        """
        Config is a static conception, which specified the parameters of one element.
        There parameters doesn't change, such as length of straight road, max speed of one vehicle, etc.
        """
        super(BaseObject, self).__init__(name, random_seed, config)
        assert random_seed is not None, "Please assign a random seed for {} class in super().__init__()".format(
            self.class_name
        )

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

    def add_body(self, physics_body):
        if self._body is None:
            # add it to physics world, in which this object will interact with other object (like collision)
            if not isinstance(physics_body, BulletBodyNode):
                raise ValueError("The physics body is not BulletBodyNode type")
            self._body = physics_body
            new_origin = NodePath(self._body)
            self.origin.getChildren().reparentTo(new_origin)
            self.origin = new_origin
            self.dynamic_nodes.append(physics_body)
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

    def detach_from_world(self, physics_world: PhysicsWorld):
        """
        It is not fully remove, it will be left in memory. if this element is useless in the future, call Func delete()
        """
        if self.origin is not None:
            self.origin.detachNode()
        self.dynamic_nodes.detach_from_physics_world(physics_world.dynamic_world)
        self.static_nodes.detach_from_physics_world(physics_world.static_world)

    def destroy(self):
        """
        Fully delete this element and release the memory
        """
        from pgdrive.engine.engine_utils import get_engine
        engine = get_engine()
        self.detach_from_world(engine.physics_world)
        if self._body is not None and hasattr(self.body, "object"):
            self.body.object = None
        if self.origin is not None:
            self.origin.removeNode()
        self.dynamic_nodes.clear()
        self.static_nodes.clear()
        self._config.clear()
