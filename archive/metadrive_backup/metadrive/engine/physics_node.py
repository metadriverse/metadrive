"""
Physics Node is the subclass of BulletNode (BulletRigidBBodyNode/BulletGhostNode and so on)
Since callback method in BulletPhysicsEngine returns PhysicsNode class and sometimes we need to do some custom
calculation and tell Object about these results, inheriting from these BulletNode class will help communicate between
Physics Callbacks and Object class
"""

from panda3d.bullet import BulletRigidBodyNode, BulletGhostNode


class BaseRigidBodyNode(BulletRigidBodyNode):
    def __init__(self, base_object_name, type_name):
        self.type_name = type_name
        super(BaseRigidBodyNode, self).__init__(type_name)
        self.setPythonTag(type_name, self)
        self.base_object_name = base_object_name
        self._clear_python_tag = False

    def rename(self, new_name):
        self.base_object_name = new_name

    def destroy(self):
        # This sentence is extremely important!
        self.base_object_name = None
        self.clearPythonTag(self.getName())
        self._clear_python_tag = True

    def __del__(self):
        assert self._clear_python_tag, "You should call destroy() of BaseRigidBodyNode!"


class BaseGhostBodyNode(BulletGhostNode):
    """
    Ghost node will not collide with any bodies, while contact information can still be accessed
    """
    def __init__(self, base_object_name, type_name):
        self.type_name = type_name
        super(BaseGhostBodyNode, self).__init__(type_name)
        self.setPythonTag(type_name, self)
        self.base_object_name = base_object_name
        self._clear_python_tag = False

    def rename(self, new_name):
        self.base_object_name = new_name

    def destroy(self):
        # This sentence is extremely important!
        self.base_object_name = None
        self.clearPythonTag(self.getName())
        self._clear_python_tag = True

    def __del__(self):
        assert self._clear_python_tag, "You should call destroy() of BaseRigidBodyNode!"
