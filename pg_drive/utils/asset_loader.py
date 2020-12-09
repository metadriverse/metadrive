import logging
from typing import Union

from direct.showbase.Loader import Loader


class AssetLoader:
    """
    Load model for each element when render is needed.
    """
    loader = None
    asset_path = None

    @staticmethod
    def init_loader(show_base_loader: Union[Loader, bool], pg_path: str):
        """
        Due to the feature of Panda3d, keep reference of loader in static variable
        """
        AssetLoader.asset_path = AssetLoader.file_path(pg_path, "assets")
        if not show_base_loader:
            logging.debug("Offscreen mode")
            return
        logging.debug("Onscreen mode, Render Elements")
        AssetLoader.loader = show_base_loader

    @classmethod
    def get_loader(cls):
        assert AssetLoader.loader, "Initialize AssetLoader before getting it"
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
