from pg_drive.world.bt_world import BtWorld
from pg_drive.world.image_buffer import ImageBuffer
from panda3d.core import Vec3, NodePath
from pg_drive.pg_config.cam_mask import CamMask
from typing import Tuple


class MiniMap(ImageBuffer):
    # shape(1200, 600) default
    BUFFER_L = 1200
    BUFFER_W = 600
    CAM_MASK = CamMask.MiniMap
    display_top = 1.13

    def __init__(self, para: Tuple, chassis_np: NodePath, bt_world: BtWorld):
        self.BUFFER_L = para[0]
        self.BUFFER_W = para[1]
        height = para[2]
        super(MiniMap, self).__init__(
            self.BUFFER_L, self.BUFFER_W, Vec3(0, 20, height), self.BKG_COLOR, bt_world.win.makeTextureBuffer,
            bt_world.makeCamera, chassis_np
        )
        self.cam.lookAt(Vec3(0, 20, 0))
        # lens = self.cam.node().getLens()
        # lens.setFilmOffset(1.0, 2.0)
        bt_world.add_to_console(self, [0., 0.33, self.display_bottom, self.display_top])
        self.buffer.setSort(0)
