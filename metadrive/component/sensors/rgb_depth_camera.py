from metadrive.component.sensors.depth_camera import DepthCamera


class RGBDepthCamera(DepthCamera):
    frame_buffer_rgb_bits = (8, 8, 8, 8)
    shader_name = "rgb_depth_cam"
    VIEW_GROUND = False
