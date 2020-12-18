from panda3d.core import NodePath, PGTop, TextNode, Vec3

from pgdrive.pg_config.cam_mask import CamMask
from pgdrive.world.image_buffer import ImageBuffer
from pgdrive.world.pg_world import PgWorld


class VehiclePanel(ImageBuffer):
    PARA_VIS_LENGTH = 12
    MAX_SPEED = 120
    BUFFER_W = 2
    BUFFER_H = 1
    CAM_MASK = CamMask.PARA_VIS
    GAP = 4.1
    TASK_NAME = "update panel"

    def __init__(self, vehicle, pg_world: PgWorld):
        if pg_world.win is None:
            return
        self.aspect2d_np = NodePath(PGTop("aspect2d"))
        self.aspect2d_np.show(self.CAM_MASK)
        self.para_vis_np = []
        # make_buffer_func, make_camera_func = pg_world.win.makeTextureBuffer, pg_world.makeCamera

        # don't delete the space in word, it is used to set a proper position
        for i, np_name in enumerate(["Steering", " Throttle", "     Brake", "    Speed"]):
            text = TextNode(np_name)
            text.setText(np_name)
            text.setSlant(0.1)
            textNodePath = self.aspect2d_np.attachNewNode(text)
            textNodePath.setScale(0.052)
            text.setFrameColor(0, 0, 0, 1)
            text.setTextColor(0, 0, 0, 1)
            text.setFrameAsMargin(-self.GAP, self.PARA_VIS_LENGTH, 0, 0)

            text.setCardColor(0.8, 0.8, 0.8, 0.9)
            text.setCardAsMargin(-self.GAP, 0, 0, 0)
            text.setCardDecal(True)
            text.setAlign(TextNode.ARight)
            textNodePath.setPos(-1.125111, 0, 0.9 - i * 0.08)
            self.para_vis_np.append(textNodePath)
        super(VehiclePanel, self).__init__(
            self.BUFFER_W,
            self.BUFFER_H,
            Vec3(-0.9, -1.01, 0.78),
            self.BKG_COLOR,
            pg_world=pg_world,
            parent_node=self.aspect2d_np
        )
        self.add_to_display(pg_world, [2 / 3, 1, self.display_bottom, self.display_top])

    def renew_2d_car_para_visualization(self, vehicle):
        steering, throttle_brake, speed = vehicle.steering, vehicle.throttle_brake, vehicle.speed
        if throttle_brake < 0:
            self.para_vis_np[2].node().setCardAsMargin(-self.GAP, self.PARA_VIS_LENGTH * abs(throttle_brake), 0, 0)
            self.para_vis_np[1].node().setCardAsMargin(-self.GAP, 0, 0, 0)
        elif throttle_brake > 0:
            self.para_vis_np[2].node().setCardAsMargin(-self.GAP, 0., 0, 0)
            self.para_vis_np[1].node().setCardAsMargin(-self.GAP, 0. + self.PARA_VIS_LENGTH * throttle_brake, 0, 0)
        else:
            self.para_vis_np[2].node().setCardAsMargin(-self.GAP, 0., 0, 0)
            self.para_vis_np[1].node().setCardAsMargin(-self.GAP, 0., 0, 0)

        steering_value = abs(steering) * self.PARA_VIS_LENGTH / 2
        if steering < 0:
            self.para_vis_np[0].node().setCardAsMargin(
                -self.GAP - self.PARA_VIS_LENGTH / 2, self.PARA_VIS_LENGTH / 2 + steering_value, 0, 0
            )
        elif steering > 0:
            left = self.PARA_VIS_LENGTH / 2 - steering_value
            self.para_vis_np[0].node().setCardAsMargin(-self.GAP - left, self.PARA_VIS_LENGTH / 2, 0, 0)
        else:
            self.para_vis_np[0].node().setCardAsMargin(-self.GAP, 0, 0, 0)
        speed_value = speed / self.MAX_SPEED * self.PARA_VIS_LENGTH
        self.para_vis_np[3].node().setCardAsMargin(-self.GAP, speed_value + 0.09, 0, 0)

    def destroy(self, pg_world=None):
        super(VehiclePanel, self).destroy(pg_world)
        for para in self.para_vis_np:
            para.removeNode()
        self.aspect2d_np.removeNode()
