import numpy as np

from metadrive.engine.engine_utils import get_engine


class InterfaceCamera:
    def __init__(self):
        engine = get_engine()
        if engine.global_config["vehicle_config"]["image_source"] == "main_camera":
            assert engine.global_config["show_interface"
                                        ] is False, "Turn off interface by [show_interface=False] for using main camera"
            assert engine.global_config["show_logo"
                                        ] is False, "Turn off logo-showing by [show_logo=False] for using main camera"
            assert engine.global_config["show_fps"
                                        ] is False, "Turn off fps-showing by [show_fps=False] for using main camera"

    @staticmethod
    def get_pixels_array(vehicle, clip):
        engine = get_engine()
        assert engine.main_camera.current_track_vehicle is vehicle, "Tracked vehicle mismatch"
        if engine.episode_step <= 1:
            engine.graphicsEngine.renderFrame()
        origin_img = engine.win.getDisplayRegion(0).getScreenshot()
        v = memoryview(origin_img.getRamImage()).tolist()
        img = np.array(v, dtype=np.uint8)
        img = img.reshape((origin_img.getYSize(), origin_img.getXSize(), 4))
        img = img[::-1]
        img = img[..., :-1]
        if not clip:
            return img.astype(np.uint8)
        else:
            return img / 255

    def destroy(self, *args, **kwargs):
        pass
