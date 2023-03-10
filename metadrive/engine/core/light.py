from panda3d.core import LVector4, NodePath, DirectionalLight, AmbientLight

from metadrive.base_class.base_object import BaseObject
from metadrive.constants import CamMask


class Light(BaseObject):
    """
    It is dynamic element since it will follow the camera
    """

    def __init__(self, config):
        super(Light, self).__init__(random_seed=0)
        self.global_light = config["global_light"]
        self.direction_np = NodePath(DirectionalLight("direction light"))
        # self.light.node().setScene(self.render)

        self._node_path_list.append(self.direction_np)

        # Too large will cause the graphics card out of memory.
        # self.direction_np.node().setShadowCaster(True, 8192, 8192)
        # self.direction_np.node().setShadowCaster(True, 4096, 4096)
        if self.global_light:
            shadow = (16384, 16384)
        else:
            shadow = (512, 512)

        self.direction_np.node().setShadowCaster(True, *shadow)
        self.direction_np.setPos(-1000, 300, 1000)
        self.direction_np.lookAt(0, 0, 0)

        # self.direction_np.node().showFrustum()
        # self.light.node().getLens().setNearFar(10, 100)

        self.direction_np.node().setColor(LVector4(0.6, 0.6, 0.6, 1))
        self.direction_np.node().setCameraMask(CamMask.Shadow)

        dlens = self.direction_np.node().getLens()
        if self.global_light:
            dlens.setFilmSize(256, 256)
        else:
            dlens.setFilmSize(128, 128)
        # dlens.setFocalLength(1)
        # dlens.setNear(3)

        self.direction_np.node().setColorTemperature(6500)
        self.direction_np.reparentTo(self.origin)

        self.ambient_np = NodePath(AmbientLight("Ambient"))
        self.ambient_np.node().setColor(LVector4(0.1, 0.1, 0.1, 1))
        self.ambient_np.reparentTo(self.origin)

        self._node_path_list.append(self.ambient_np)

    # def step(self, pos):
    #     # Disabled now
    #     return
    #     if not self.global_light:
    #         self.direction_np.setPos(pos[0] - 200, pos[1] + 100, 150)
    #         self.direction_np.lookAt(pos[0], pos[1], 0)
    #
    # def reset(self, random_seed=None, *args, **kwargs):
    #     # Disabled now
    #     return
    #     if self.direction_np is not None and not self.global_light:
    #         self.direction_np.setHpr(-90, -120, 0)
    #         self.direction_np.setY(20)
    #         self.direction_np.setX(0)
    #         self.direction_np.setZ(50)
