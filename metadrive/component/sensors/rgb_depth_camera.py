from metadrive.component.sensors.depth_camera import DepthCamera


class RGBDepthCamera(DepthCamera):
    """
    (Deprecated) Same as RGBCamera, while the forth channel is for storing depth information
    """
    raise DeprecationWarning("This one won't work currently")
    shader_name = "rgb_depth_cam"
    VIEW_GROUND = False
