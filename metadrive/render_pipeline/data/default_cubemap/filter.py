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

from __future__ import print_function, division

import os
import shutil
from os.path import dirname, realpath
from panda3d.core import *
from direct.stdpy.file import isdir, isfile, join
from direct.showbase.ShowBase import ShowBase

class Application(ShowBase):
    def __init__(self):
        load_prc_file_data("", """
            textures-power-2 none
            window-type offscreen
            win-size 100 100
            gl-coordinate-system default
            notify-level-display error
            print-pipe-types #f
            gl-version 4 3
        """)

        ShowBase.__init__(self)

        base_path = realpath(dirname(__file__))
        os.chdir(base_path)
        filter_dir = join(base_path, "tmp/")
        if isdir(filter_dir):
            shutil.rmtree(filter_dir)
        os.makedirs(filter_dir)

        source_path = join(base_path, "source")
        extension = ".jpg"
        if isfile(join(source_path, "1.png")):
            extension = ".png"

        cubemap = self.loader.loadCubeMap(
            Filename.from_os_specific(join(source_path, "#" + extension)))
        mipmap, size = -1, 1024

        cshader = Shader.load_compute(Shader.SL_GLSL, "filter.compute.glsl")

        while size > 1:
            size = size // 2
            mipmap += 1
            print("Filtering mipmap", mipmap)

            dest_cubemap = Texture("Dest")
            dest_cubemap.setup_cube_map(size, Texture.T_float, Texture.F_rgba16)
            node = NodePath("")

            for i in range(6):
                node.set_shader(cshader)
                node.set_shader_inputs(
                    SourceTex=cubemap,
                    DestTex=dest_cubemap,
                    currentSize=size,
                    currentMip=mipmap,
                    currentFace=i)
                attr = node.get_attrib(ShaderAttrib)
                self.graphicsEngine.dispatch_compute(
                    ( (size + 15) // 16, (size+15) // 16, 1), attr, self.win.gsg)

            print(" Extracting data ..")

            self.graphicsEngine.extract_texture_data(dest_cubemap, self.win.gsg)

            print(" Writing data ..")
            dest_cubemap.write(join(filter_dir, "{}-#.png".format(mipmap)), 0, 0, True, False)


        print("Reading in data back in ..")
        tex = self.loader.loadCubeMap(Filename.from_os_specific(join(base_path, "tmp/#-#.png")), readMipmaps="True")

        print("Writing txo ..")
        tex.write("cubemap.txo.pz")

        shutil.rmtree(join(base_path, "tmp"))


Application()
