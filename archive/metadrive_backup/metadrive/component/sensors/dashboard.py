from panda3d.core import NodePath, PGTop, TextNode, CardMaker, Vec3, OrthographicLens

from metadrive.component.sensors.base_sensor import BaseSensor
from metadrive.constants import CamMask
from metadrive.engine.core.image_buffer import ImageBuffer


class DashBoard(ImageBuffer, BaseSensor):
    """
    Dashboard for showing the speed and brake/throttle/steering
    """
    def perceive(self, *args, **kwargs):
        """
        This is only used for GUI and won't provide any observation result
        """
        raise NotImplementedError

    PARA_VIS_LENGTH = 12
    PARA_VIS_HEIGHT = 1
    MAX_SPEED = 120
    BUFFER_W = 2
    BUFFER_H = 1
    CAM_MASK = CamMask.PARA_VIS
    GAP = 4.1
    TASK_NAME = "update panel"

    def __init__(self, engine, *, cuda):
        if engine.win is None:
            return
        self.aspect2d_np = NodePath(PGTop("aspect2d"))
        self.aspect2d_np.show(self.CAM_MASK)
        self.para_vis_np = {}

        tmp_node_path_list = []
        # make_buffer_func, make_camera_func = engine.win.makeTextureBuffer, engine.makeCamera

        # don't delete the space in word, it is used to set a proper position
        for i, np_name in enumerate(["Steering", " Throttle", "     Brake", "    Speed"]):
            text = TextNode(np_name)
            text.setText(np_name)
            text.setSlant(0.1)
            textNodePath = self.aspect2d_np.attachNewNode(text)
            tmp_node_path_list.append(textNodePath)

            textNodePath.setScale(0.052)
            text.setFrameColor(0, 0, 0, 1)
            text.setTextColor(0, 0, 0, 1)
            text.setFrameAsMargin(-self.GAP, self.PARA_VIS_LENGTH, 0, 0)
            text.setAlign(TextNode.ARight)
            textNodePath.setPos(-1.125111, 0, 0.9 - i * 0.08)
            if i != 0:
                cm = CardMaker(np_name)
                cm.setFrame(0, self.PARA_VIS_LENGTH - 0.21, -self.PARA_VIS_HEIGHT / 2 + 0.1, self.PARA_VIS_HEIGHT / 2)
                cm.setHasNormals(True)
                card = textNodePath.attachNewNode(cm.generate())
                tmp_node_path_list.append(card)

                card.setPos(0.21, 0, 0.22)
                self.para_vis_np[np_name.lstrip()] = card
            else:
                # left
                name = "Left"
                cm = CardMaker(name)
                cm.setFrame(
                    0, (self.PARA_VIS_LENGTH - 0.4) / 2, -self.PARA_VIS_HEIGHT / 2 + 0.1, self.PARA_VIS_HEIGHT / 2
                )
                cm.setHasNormals(True)
                card = textNodePath.attachNewNode(cm.generate())
                tmp_node_path_list.append(card)

                card.setPos(0.2 + self.PARA_VIS_LENGTH / 2, 0, 0.22)
                self.para_vis_np[name] = card
                # right
                name = "Right"
                cm = CardMaker(np_name)
                cm.setFrame(
                    -(self.PARA_VIS_LENGTH - 0.1) / 2, 0, -self.PARA_VIS_HEIGHT / 2 + 0.1, self.PARA_VIS_HEIGHT / 2
                )
                cm.setHasNormals(True)
                card = textNodePath.attachNewNode(cm.generate())
                tmp_node_path_list.append(card)

                card.setPos(0.2 + self.PARA_VIS_LENGTH / 2, 0, 0.22)
                self.para_vis_np[name] = card
        super(DashBoard, self).__init__(
            self.BUFFER_W, self.BUFFER_H, self.BKG_COLOR, parent_node=self.aspect2d_np, engine=engine
        )
        self.origin = NodePath("DashBoard")
        self._node_path_list.extend(tmp_node_path_list)

    def _create_camera(self, parent_node, bkg_color):
        """
        Create orthogonal camera for the buffer
        """
        self.cam = cam = self.engine.makeCamera(self.buffer, clearColor=bkg_color)
        cam.node().setCameraMask(self.CAM_MASK)

        self.cam.reparentTo(parent_node)
        self.cam.setPos(Vec3(-0.9, -1.01, 0.78))

    def update_vehicle_state(self, vehicle):
        """
        Update the dashboard result given a vehicle
        """
        steering, throttle_brake, speed = vehicle.steering, vehicle.throttle_brake, vehicle.speed_km_h
        if throttle_brake < 0:
            self.para_vis_np["Throttle"].setScale(0, 1, 1)
            self.para_vis_np["Brake"].setScale(-throttle_brake, 1, 1)
        elif throttle_brake > 0:
            self.para_vis_np["Throttle"].setScale(throttle_brake, 1, 1)
            self.para_vis_np["Brake"].setScale(0, 1, 1)
        else:
            self.para_vis_np["Throttle"].setScale(0, 1, 1)
            self.para_vis_np["Brake"].setScale(0, 1, 1)

        steering_value = abs(steering)
        if steering < 0:
            self.para_vis_np["Left"].setScale(steering_value, 1, 1)
            self.para_vis_np["Right"].setScale(0, 1, 1)
        elif steering > 0:
            self.para_vis_np["Right"].setScale(steering_value, 1, 1)
            self.para_vis_np["Left"].setScale(0, 1, 1)
        else:
            self.para_vis_np["Right"].setScale(0, 1, 1)
            self.para_vis_np["Left"].setScale(0, 1, 1)
        speed_value = speed / self.MAX_SPEED
        self.para_vis_np["Speed"].setScale(speed_value, 1, 1)

    def remove_display_region(self):
        self.buffer.set_active(False)
        super(DashBoard, self).remove_display_region()

    def add_display_region(self, display_region, keep_height=False):
        super(DashBoard, self).add_display_region(display_region, False)
        self.buffer.set_active(True)
        self.origin.reparentTo(self.aspect2d_np)

    def destroy(self):
        super(DashBoard, self).destroy()
        for para in self.para_vis_np.values():
            para.removeNode()
        self.aspect2d_np.removeNode()

    def track(self, *args, **kwargs):
        # for compatibility
        pass
