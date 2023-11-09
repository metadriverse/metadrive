# import numpy
#
#
import numpy as np
from panda3d.core import Texture
import cv2
from metadrive.engine.asset_loader import AssetLoader
from metadrive.engine.engine_utils import initialize_engine
from metadrive.envs.base_env import BASE_DEFAULT_CONFIG

if __name__ == '__main__':
    engine = initialize_engine(BASE_DEFAULT_CONFIG)
    tex_origin = engine.loader.loadTexture(AssetLoader.file_path("textures", "asphalt", "diff_2k.png"))
    tex = np.frombuffer(tex_origin.getRamImage().getData(), dtype=np.uint8)
    tex = tex.copy()
    tex = tex.reshape((tex_origin.getYSize(), tex_origin.getXSize(), 3))

    for x in range(0, 2048, 512):
        tex[x:x + 256, ...] = 220

    cv2.imwrite("test_tex.png", tex)

    # semantic_tex = Texture()
    # semantic_tex.setup2dTexture(*semantics.shape[:2], Texture.TFloat, Texture.F_red)
    # semantic_tex.setRamImage(semantics)
