import logging
from typing import Union

from direct.showbase.Loader import Loader


class VisLoader:
    """
    Load model for each element when render is needed.
    """
    loader = None
    asset_path = None

    # pre load lane line model to save memory
    strip_lane_line = None
    circular_lane_line = None
    side_walk = None

    @staticmethod
    def init_loader(show_base_loader: Union[Loader, bool], pg_path: str):
        """
        Due to the feature of Panda3d, keep reference of loader in static variable
        """
        VisLoader.asset_path = VisLoader.file_path(pg_path, "asset")
        if not show_base_loader:
            logging.debug("Offscreen mode")
            return
        logging.debug("Onscreen mode, Render Elements")
        VisLoader.loader = show_base_loader
        # VisLoader.pre_load_lane_line_model()

    @classmethod
    def get_loader(cls):
        assert VisLoader.loader, "Initialize VisLoader before getting it"
        return cls.loader

    @staticmethod
    def windows_style2unix_style(path):
        u_path = "/" + path[0].lower() + path[2:]
        u_path.replace("\\", "/")
        return u_path

    @staticmethod
    def file_path(*args):
        import os, sys
        path = os.path.join(*args)
        if sys.platform.startswith("win"):
            path = path.replace("\\", "/")
        return path
