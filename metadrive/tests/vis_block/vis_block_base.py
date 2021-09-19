from typing import Union, Tuple

from direct.showbase import ShowBase
from panda3d.bullet import BulletPlaneShape, BulletRigidBodyNode, BulletDebugNode
from panda3d.core import Vec3, NodePath, LineSegs

from metadrive.component.algorithm.BIG import NextStep
from metadrive.component.map.base_map import BaseMap
from metadrive.constants import BKG_COLOR
from metadrive.constants import CollisionGroup
from metadrive.engine.core.physics_world import PhysicsWorld


class TestBlock(ShowBase.ShowBase):
    def __init__(self, debug=False):
        self.debug = debug
        super(TestBlock, self).__init__(windowType="onscreen")
        self.setBackgroundColor(BKG_COLOR)
        self.setFrameRateMeter(True)
        self.cam.setPos(0, 0, 300)
        self.cam.lookAt(0, 0, 0)
        self.map = None
        self.worldNP = None
        self.world = None
        self.physics_world = None
        self.debugNP = None
        self.groundNP = None
        self.setup()
        self.taskMgr.add(self.update, 'updateWorld')
        self.taskMgr.add(self.analyze, "analyze geom node")
        self.add_block_func = None  # function pointer
        self.last_block = None
        self.block_index = 1
        self.big = None
        self.accept("f4", self.render.analyze)
        self.vehicle = None
        self.inputs = None

    def vis_big(self, big):
        # self.cam.setPos(200, 700, 1000)
        # self.cam.lookAt(200, 700, 0)
        self.big = big
        self.big.next_step = NextStep.forward
        self.task_mgr.add(self.big_algorithm, "start_big")

    def big_algorithm(self, task):
        if self.big.big_helper_func():
            return task.done
        else:
            return task.cont

    def setup(self):
        self.worldNP = self.render.attachNewNode('World')
        self.world = PhysicsWorld()
        self.physics_world = self.world

        # World
        if self.debug:
            self.debugNP = self.worldNP.attachNewNode(BulletDebugNode('Debug'))
            self.debugNP.show()
            self.debugNP.node().showWireframe(True)
            self.debugNP.node().showConstraints(True)
            self.debugNP.node().showBoundingBoxes(True)
            self.debugNP.node().showNormals(True)

            self.debugNP.showTightBounds()
            self.debugNP.showBounds()
            self.world.dynamic_world.setDebugNode(self.debugNP.node())

        # Ground (static)
        shape = BulletPlaneShape(Vec3(0, 0, 1), 0)
        self.groundNP = self.worldNP.attachNewNode(BulletRigidBodyNode('Ground'))
        self.groundNP.node().addShape(shape)
        self.groundNP.setPos(0, 0, 0)
        self.groundNP.setCollideMask(CollisionGroup.AllOn)
        self.world.dynamic_world.attachRigidBody(self.groundNP.node())

    def update(self, task):
        dt = 1 / 60
        # self.world.doPhysics(dt)
        self.world.dynamic_world.doPhysics(dt, 1, dt)
        return task.cont

    def analyze(self, task):
        self.render.analyze()
        return task.done

    def test_reset(self, big_type, para):
        """
        For reset test and map storing
        """
        self.map = BaseMap(self, {"type": big_type, "config": para})
        self.accept("c", self.clear)
        self.accept("a", self.re_add)

    def clear(self):
        self.map.unload_map(self)

    def re_add(self):
        self.map.load_map(self)

    def add_line(self, start_p: Union[Vec3, Tuple], end_p: Union[Vec3, Tuple], color, thickness: float):
        line_seg = LineSegs("interface")
        line_seg.setColor(*color)
        line_seg.moveTo(start_p)
        line_seg.drawTo(end_p)
        line_seg.setThickness(thickness)
        NodePath(line_seg.create(False)).reparentTo(self.render)

    def show_bounding_box(self, road_network):
        bound_box = road_network.get_bounding_box()
        points = [(x, -y) for x in bound_box[:2] for y in bound_box[2:]]
        for k, p in enumerate(points[:-1]):
            for p_ in points[k + 1:]:
                self.add_line((*p, 2), (*p_, 2), (1, 0., 0., 1), 2)


if __name__ == "__main__":
    TestBlock = TestBlock()
    TestBlock.run()
