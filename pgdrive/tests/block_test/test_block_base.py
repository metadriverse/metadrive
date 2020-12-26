from direct.controls.InputState import InputState
from direct.showbase import ShowBase
from panda3d.bullet import BulletWorld, BulletPlaneShape, BulletRigidBodyNode, BulletDebugNode
from panda3d.core import Vec3, BitMask32

from pgdrive.scene_creator.algorithm.BIG import NextStep
from pgdrive.scene_creator.map import Map


class TestBlock(ShowBase.ShowBase):
    asset_path = "../../"

    def __init__(self, debug=False):
        self.debug = debug
        super(TestBlock, self).__init__(windowType="onscreen")
        self.setBackgroundColor(38 / 255, 58 / 255, 102 / 255, 1)
        self.setFrameRateMeter(True)
        self.cam.setPos(0, 0, 300)
        self.cam.lookAt(0, 0, 0)
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
        self.world = BulletWorld()
        self.physics_world = self.world
        self.world.setGravity(Vec3(0, 0, -9.81))

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
            self.world.setDebugNode(self.debugNP.node())

        # Ground (static)
        shape = BulletPlaneShape(Vec3(0, 0, 1), 0)
        self.groundNP = self.worldNP.attachNewNode(BulletRigidBodyNode('Ground'))
        self.groundNP.node().addShape(shape)
        self.groundNP.setPos(0, 0, 0)
        self.groundNP.setCollideMask(BitMask32.allOn())
        self.world.attachRigidBody(self.groundNP.node())

    def update(self, task):
        dt = 1 / 60
        # self.world.doPhysics(dt)
        self.world.doPhysics(dt, 1, dt)
        if self.vehicle is not None:
            self.processInput(dt)
        return task.cont

    def analyze(self, task):
        self.render.analyze()
        return task.done

    def test_reset(self, big_type, para):
        """
        For reset test and map storing
        """
        self.map = Map(self, {"type": big_type, "config": para})
        self.accept("c", self.clear)
        self.accept("a", self.re_add)

    def clear(self):
        self.map.unload_from_pg_world(self)

    def re_add(self):
        self.map.load_to_pg_world(self)

    def add_vehicle(self, vehicle):
        self.vehicle = vehicle
        vehicle.attach_to_pg_world(self.worldNP, self.world)
        self.inputs = InputState()
        self.inputs.watchWithModifiers('forward', 'w')
        self.inputs.watchWithModifiers('reverse', 's')
        self.inputs.watchWithModifiers('turnLeft', 'a')
        self.inputs.watchWithModifiers('turnRight', 'd')

    def processInput(self, dt):
        if not self.inputs.isSet('turnLeft') and not self.inputs.isSet('turnRight'):
            steering = 0.0
            # self.bt_vehicle.steering = 0.0
        else:
            if self.inputs.isSet('turnLeft'):
                # steering = dt
                steering = 1

            if self.inputs.isSet('turnRight'):
                # steering = -dt
                steering = -1

        if not self.inputs.isSet('forward') and not self.inputs.isSet("reverse"):
            throttle_brake = 0
        else:
            if self.inputs.isSet('forward'):
                throttle_brake = 1.0
            if self.inputs.isSet('reverse'):
                throttle_brake = -1.0

        # self.bt_vehicle.set_incremental_action(numpy.array([steering, throttle_brake]))
        self.vehicle.step([steering, throttle_brake])


if __name__ == "__main__":
    TestBlock = TestBlock()
    TestBlock.run()
