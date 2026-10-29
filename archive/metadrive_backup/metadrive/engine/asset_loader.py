import os
import pathlib
import sys

from metadrive.constants import RENDER_MODE_NONE
from metadrive.engine.logger import get_logger
from metadrive.utils.utils import is_win
from metadrive.version import VERSION


class AssetLoader:
    """
    Load model for each element when Online/Offline render is needed. It will load all assets in initialization.
    """
    logger = get_logger()
    loader = None
    asset_path = pathlib.PurePosixPath(__file__).parent.parent.joinpath("assets") if not is_win(
    ) else pathlib.Path(__file__).resolve().parent.parent.joinpath("assets")

    @staticmethod
    def init_loader(engine):
        """
        Due to the feature of Panda3d, keep reference of loader in static variable
        """
        if engine.win is None:
            AssetLoader.logger.debug("Physics world mode")
            return
        assert engine.mode != RENDER_MODE_NONE
        AssetLoader.logger.debug("Onscreen/Offscreen mode, Render/Load Elements")
        AssetLoader.loader = engine.loader

    @property
    def asset_version(self):
        """
        Read the asset version
        Returns: str Asset version

        """
        return asset_version()

    @classmethod
    def get_loader(cls):
        """
        Return asset loader. It equals to engine.loader and AssetLoader.loader
        Returns: asset loader

        """
        assert AssetLoader.loader, "Please initialize AssetLoader before getting it!"
        return cls.loader

    @staticmethod
    def windows_style2unix_style(win_path):
        """
        Panda uses unix style path even on Windows, we can use this API to convert Windows path to Unix style
        Args:
            win_path: Path in windows style like C://my//file.txt

        Returns: path in unix style like /c/my/file.txt

        """
        path = win_path.as_posix()
        panda_path = "/" + path[0].lower() + path[2:]
        return panda_path

    @staticmethod
    def file_path(*path_string, unix_style=True):
        """
        Usage is the same as path.join(dir_1,dir_2,file_name)
        :param path_string: a tuple
        :param unix_style: it will convert windows path style to unix style. This is because panda uses unix style path
        to find assets.
        :return: file path used to load asset
        """
        path = AssetLoader.asset_path.joinpath(*path_string)
        return AssetLoader.windows_style2unix_style(path
                                                    ) if sys.platform.startswith("win") and unix_style else str(path)

    @classmethod
    def load_model(cls, file_path):
        """
        A quick load method
        :param file_path: path in string, usually use the return value of AssetLoader.file_path()
        :return: model node path
        """
        assert cls.loader is not None
        return cls.loader.loadModel(file_path)

    @classmethod
    def initialized(cls):
        return cls.loader is not None

    @classmethod
    def should_update_asset(cls):
        """Return should pull the asset or not."""
        asset_version_match = asset_version() == VERSION

        # In PR #531, we introduced new assets. For the user that installed MetaDrive before
        # this PR, we need to pull the latest texture again.
        grass_texture_exists = os.path.exists(
            AssetLoader.file_path("textures", "grass1", "GroundGrassGreen002_COL_1K.jpg", unix_style=False)
        )

        return (not asset_version_match) or (not grass_texture_exists)


def initialize_asset_loader(engine):
    """
    Initialize asset loader
    Args:
        engine: baseEngine

    Returns: None

    """
    # load model file in utf-8
    os.environ["PYTHONUTF8"] = "on"
    if AssetLoader.initialized():
        AssetLoader.logger.warning(
            "AssetLoader is initialize to root path: {}! But you are initializing again!".format(
                AssetLoader.asset_path
            )
        )
        return
    AssetLoader.init_loader(engine)


def close_asset_loader():
    cls = AssetLoader
    cls.loader = None


def randomize_cover():
    background_folder_name = "background"
    files = os.listdir(AssetLoader.asset_path.joinpath(background_folder_name))
    files = [f for f in files if f.startswith("logo") and f.endswith("png")]
    from metadrive.utils import get_np_random
    selected = get_np_random().choice(files)
    selected_file = AssetLoader.file_path("{}/{}".format(background_folder_name, selected))
    return selected_file


def get_logo_file():
    file = AssetLoader.file_path("logo-tiny.png")
    # assert os.path.exists(file)
    return file


def asset_version():
    from metadrive.version import asset_version
    return asset_version()
