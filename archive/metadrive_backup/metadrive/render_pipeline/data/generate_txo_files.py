"""

Converts pipeline resources to TXO files to speed up loading.
This is called during the setup.

"""

from __future__ import print_function

import os

from panda3d.core import Filename, Texture, load_prc_file_data
load_prc_file_data("", "window-type none")
load_prc_file_data("", "notify-level-pnmimage error")
load_prc_file_data("", "textures-power-2 none")

files_to_convert = [
    "data/gui/loading_screen_bg.png", "rpplugins/bloom/resources/lens_dirt.png", "data/builtin_models/skybox/skybox.jpg"
]

this_dir = os.path.realpath(os.path.dirname(__file__))
os.chdir(this_dir)

pipeline_dir = "../"

import direct.directbase.DirectStart

for filename in files_to_convert:
    src_path = os.path.abspath(os.path.join(pipeline_dir, filename))
    fullpath = Filename.from_os_specific(src_path).get_fullpath()
    dest_path = fullpath.replace(".png", ".txo.pz")
    dest_path = dest_path.replace(".jpg", ".txo.pz")

    print(src_path, "->", dest_path)
    loader.load_texture(fullpath).write(dest_path)
