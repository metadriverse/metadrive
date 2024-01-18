"""
This file create a TestBlock class, which resembles the ``engine'' in normal code. By building this TestBlock,
we can directly pop up a Panda3D window and visualize the content, e.g. the road blocks that are constructed via
block.construct_block(test_block.render, test_block.world).
"""
from typing import Union, Tuple
from metadrive.engine.asset_loader import close_asset_loader
from direct.showbase import ShowBase
from panda3d.bullet import BulletPlaneShape, BulletRigidBodyNode, BulletDebugNode
from panda3d.core import Vec3, NodePath, LineSegs

from metadrive.component.algorithm.BIG import NextStep
from metadrive.component.map.base_map import BaseMap
from metadrive.constants import BKG_COLOR
from metadrive.constants import CollisionGroup
from metadrive.engine.core.physics_world import PhysicsWorld


class TestBlock(ShowBase.ShowBase):
    def __init__(self, debug=False, window_type="onscreen"):
        self.debug = True
        super(TestBlock, self).__init__(windowType=window_type)
        self.mode = "onscreen"
        self.setBackgroundColor(BKG_COLOR)
        if window_type != "none":
            self.setFrameRateMeter(True)
            self.cam.setPos(0, 0, 300)
            self.cam.lookAt(0, 0, 0)
        self.use_render_pipeline = False
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
        self.accept("4", self.render.analyze)
        self.accept("f1", self.toggleDebug)
        self.accept("1", self.toggleDebug)
        self.agent = None
        self.inputs = None

    def toggleDebug(self):
        if self.debugNP is None:
            debugNode = BulletDebugNode('Debug')
            debugNode.showWireframe(True)
            debugNode.showConstraints(True)
            debugNode.showBoundingBoxes(False)
            debugNode.showNormals(True)
            debugNP = self.render.attachNewNode(debugNode)
            self.physics_world.static_world.setDebugNode(debugNP.node())
            self.debugNP = debugNP
        if self.debugNP.isHidden():
            self.debugNP.show()

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
        self.world = PhysicsWorld(debug=self.debug)
        self.physics_world = self.world

        # Ground (static)
        shape = BulletPlaneShape(Vec3(0, 0, 1), 0)
        self.groundNP = self.worldNP.attachNewNode(BulletRigidBodyNode('Ground'))
        self.groundNP.node().addShape(shape)
        self.groundNP.setPos(0, 0, 0)
        self.groundNP.setCollideMask(CollisionGroup.AllOn)
        self.world.dynamic_world.attachRigidBody(self.groundNP.node())
        self.toggleDebug()

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

    def _draw_line_3d(self, start_p: Union[Vec3, Tuple], end_p: Union[Vec3, Tuple], color, thickness: float):
        line_seg = LineSegs("interface")
        line_seg.setColor(*color)
        line_seg.moveTo(start_p)
        line_seg.drawTo(end_p)
        line_seg.setThickness(thickness)
        np = NodePath(line_seg.create(False))
        np.reparentTo(self.render)

    def show_bounding_box(self, road_network):
        bound_box = road_network.get_bounding_box()
        points = [(x, y) for x in bound_box[:2] for y in bound_box[2:]]
        for k, p in enumerate(points[:-1]):
            for p_ in points[k + 1:]:
                self._draw_line_3d((*p, 2), (*p_, 2), (1, 0., 0., 1), 2)

    def close(self):
        """
        Close the showbase
        Returns: None

        """
        self.taskMgr.stop()
        # It will report a warning said AsynTaskChain is created when taskMgr.destroy() is called but a new showbase is
        # created.
        self.taskMgr.destroy()
        self.physics_world.dynamic_world.clearContactAddedCallback()
        self.physics_world.destroy()
        self.destroy()
        close_asset_loader()

        import sys
        if sys.version_info >= (3, 0):
            import builtins
        else:
            import __builtin__ as builtins
        if hasattr(builtins, "base"):
            del builtins.base


if __name__ == "__main__":
    TestBlock = TestBlock()
    TestBlock.run()
