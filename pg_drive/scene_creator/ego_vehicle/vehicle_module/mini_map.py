from pg_drive.world.bt_world import BtWorld
from pg_drive.world.ImageBuffer import ImageBuffer
from panda3d.core import Vec3
from copy import copy
from pg_drive.pg_config.cam_mask import CamMask


class MiniMap(ImageBuffer):
    # shape(1200, 600)
    BUFFER_L = 1200
    BUFFER_W = 600
    TOP_CAM_DIST = 10
    CAM_MASK = CamMask.MiniMap

    def __init__(self, height: float, bt_world: BtWorld):
        super(MiniMap, self).__init__(
            self.BUFFER_L, self.BUFFER_W, Vec3(0, 0, height), self.BKG_COLOR, bt_world.win.makeTextureBuffer,
            bt_world.makeCamera, bt_world.render
        )
        self.height = height
        bt_world.add_to_console(self, [0., 0.33, self.display_bottom, self.display_top])
        self.buffer.setSort(0)

    def renew_position(self, current_pos, forward_dir):
        center_pos = copy(list(current_pos))
        center_pos[0] += forward_dir[0] * self.TOP_CAM_DIST
        center_pos[1] += forward_dir[1] * self.TOP_CAM_DIST
        center_pos[2] += self.height
        self.cam.setPos(*center_pos)

        current_pos[0] += forward_dir[0] * (self.TOP_CAM_DIST + 1)
        current_pos[1] += forward_dir[1] * (self.TOP_CAM_DIST + 1)
        self.cam.lookAt(current_pos)
