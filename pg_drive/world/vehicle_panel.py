from panda3d.core import NodePath, PGTop, TextNode, Vec3, TextFont
from pg_drive.world.ImageBuffer import ImageBuffer
from pg_drive.pg_config.cam_mask import CamMask


class VehiclePanel(ImageBuffer):
    PARA_VIS_LENGTH = 12
    MAX_SPEED = 120
    BUFFER_L = 800
    BUFFER_W = 400
    CAM_MASK = CamMask.PARA_VIS
    GAP = 4.1

    def __init__(self, make_buffer_func, make_camera_func):
        self.aspect2d_np = NodePath(PGTop("aspect2d"))
        self.aspect2d_np.show(self.CAM_MASK)
        self.para_vis_np = []

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
            self.BUFFER_L, self.BUFFER_W, Vec3(-0.9, -1.01, 0.78), self.BKG_COLOR, make_buffer_func, make_camera_func,
            self.aspect2d_np
        )

    def renew_2d_car_para_visualization(self, steering, throttle_brake, speed):
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
