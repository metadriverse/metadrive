"""

RenderPipeline

Copyright (c) 2014-2016 tobspr <tobias.springer1@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

"""

# This script precomputes the noise for the film grain to improve performance

import os
curr_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(curr_dir)

import sys
sys.path.insert(0, "../../rpcore")
sys.path.insert(0, "../../")

from panda3d.core import load_prc_file_data, NodePath, Shader, Texture, ShaderAttrib
from panda3d.core import PNMImage

from direct.showbase.ShowBase import ShowBase


class Application(ShowBase):
    def __init__(self):
        load_prc_file_data(
            "", """
            textures-power-2 none
            window-type offscreen
            win-size 100 100
            gl-coordinate-system default
            notify-level-display error
            print-pipe-types #f
            gl-version 4 3
        """
        )

        ShowBase.__init__(self)

        dest_tex = Texture()
        dest_tex.setup_2d_texture(2048, 2048, Texture.T_unsigned_byte, Texture.F_rgba8)
        cshader = Shader.load_compute(Shader.SL_GLSL, "grain.compute.glsl")
        node = NodePath("")
        node.set_shader(cshader)
        node.set_shader_input("DestTex", dest_tex)
        attr = node.get_attrib(ShaderAttrib)
        self.graphicsEngine.dispatch_compute((2048 // 16, 2048 // 16, 1), attr, self.win.gsg)

        base.graphicsEngine.extract_texture_data(dest_tex, base.win.gsg)

        # Convert to single channel
        img = PNMImage(2048, 2048, 1, 255)
        dest_tex.store(img)
        img.set_num_channels(1)

        tex = Texture()
        tex.load(img)
        tex.write("grain.txo.pz")


Application()
