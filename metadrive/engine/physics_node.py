"""
Physics Node is the subclass of BulletNode (BulletRigidBBodyNode/BulletGhostNode and so on)
Since callback method in BulletPhysicsEngine returns PhysicsNode class and sometimes we need to do some custom
calculation and tell Object about these results, inheriting from these BulletNode class will help communicate between
Physics Callbacks and Object class
"""

from panda3d.bullet import BulletRigidBodyNode


class BaseRigidBodyNode(BulletRigidBodyNode):
    def __init__(self, base_object_name, type_name=None):
        node_name = base_object_name if type_name is None else type_name
        super(BaseRigidBodyNode, self).__init__(node_name)
        self.setPythonTag(node_name, self)
        self.base_object_name = base_object_name

    def destroy(self):
        self.base_object_name = None
        self.clearPythonTag(self.getName())
