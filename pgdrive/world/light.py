from panda3d.core import LVector4, NodePath, DirectionalLight, AmbientLight

from pgdrive.pg_config import PgConfig
from pgdrive.pg_config.cam_mask import CamMask
from pgdrive.utils.element import DynamicElement


class Light(DynamicElement):
    """
    It is dynamic element since it will follow the camera
    """
    def __init__(self, config: PgConfig):
        super(Light, self).__init__()
        self.node_path = NodePath("Light")
        self.direction_np = NodePath(DirectionalLight("direction light"))
        # self.light.node().setScene(self.render)

        # Too large will cause the graphics card out of memory.
        # self.direction_np.node().setShadowCaster(True, 8192, 8192)
        # self.direction_np.node().setShadowCaster(True, 4096, 4096)
        self.direction_np.node().setShadowCaster(True, 128, 128)

        # self.direction_np.node().showFrustum()
        # self.light.node().getLens().setNearFar(10, 100)

        self.direction_np.node().setColor(LVector4(1, 1, 0.8, 1))
        self.direction_np.node().setCameraMask(CamMask.Shadow)

        dlens = self.direction_np.node().getLens()
        dlens.setFilmSize(8, 8)
        # dlens.setFocalLength(1)
        # dlens.setNear(3)

        self.direction_np.node().setColorTemperature(4000)
        self.direction_np.reparentTo(self.node_path)

        self.ambient_np = NodePath(AmbientLight("Ambient"))
        self.ambient_np.node().setColor(LVector4(0.8, 0.8, 0.8, 1))
        self.ambient_np.reparentTo(self.node_path)

    def step(self, pos):
        self.direction_np.setPos(pos[0] - 200, pos[1] + 100, 150)
        self.direction_np.lookAt(pos[0], pos[1], 0)

    def reset(self):
        if self.direction_np is not None:
            self.direction_np.setHpr(-90, -120, 0)
            self.direction_np.setY(20)
            self.direction_np.setX(0)
            self.direction_np.setZ(50)
