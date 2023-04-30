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

import math

from metadrive.render_pipeline.rplibs.six.moves import range  # pylint: disable=import-error
from metadrive.render_pipeline.rplibs.six import iteritems, itervalues

from direct.stdpy.file import listdir, isfile, join
from panda3d.core import SamplerState, ShaderAttrib, NodePath

from metadrive.render_pipeline.rpcore.globals import Globals
from metadrive.render_pipeline.rpcore.rpobject import RPObject
from metadrive.render_pipeline.rpcore.image import Image
from metadrive.render_pipeline.rpcore.loader import RPLoader


class ScatteringMethod(RPObject):
    """ Base class for all scattering methods """
    def __init__(self, plugin_handle):
        RPObject.__init__(self)
        self.handle = plugin_handle

    def load(self):
        """ Should load all required resources here """
        raise NotImplementedError()

    def compute(self):
        """ Should compute the model here if necessary """
        raise NotImplementedError()


class ScatteringMethodHosekWilkie(ScatteringMethod):
    """ Scattering as suggested by Hosek and Wilkie """
    def load(self):
        """ Loads the scattering method """
        lut_src = self.handle.get_resource("hosek_wilkie_scattering/scattering_lut.txo")

        if not isfile(lut_src):
            self.error(
                "Could not find precompiled LUT for the Hosek Wilkie "
                "Scattering! Make sure you compiled the algorithm code!"
            )
            return

        lut_tex = RPLoader.load_sliced_3d_texture(lut_src, 512, 128, 100)
        lut_tex.set_wrap_u(SamplerState.WM_repeat)
        lut_tex.set_wrap_v(SamplerState.WM_clamp)
        lut_tex.set_wrap_w(SamplerState.WM_clamp)
        lut_tex.set_minfilter(SamplerState.FT_linear)
        lut_tex.set_magfilter(SamplerState.FT_linear)

        # Setting the format explicitely shouldn't be necessary
        # lut_tex.set_format(Image.F_rgb16)

        self.handle.display_stage.set_shader_input("ScatteringLUT", lut_tex)
        self.handle.envmap_stage.set_shader_input("ScatteringLUT", lut_tex)

    def compute(self):
        """ Computes the scattering method, not required since we use a precomputed
        LUT """
        pass


class ScatteringMethodEricBruneton(ScatteringMethod):
    """ Precomputed atmospheric scattering by Eric Bruneton """
    def load(self):
        """ Inits parameters, those should match with the ones specified in common.glsl """
        self.use_32_bit = False
        self.trans_w, self.trans_h = 256 * 4, 64 * 4
        self.sky_w, self.sky_h = 64 * 4, 16 * 4
        self.res_r, self.res_mu, self.res_mu_s, self.res_nu = 32, 128, 32, 8
        self.res_mu_s_nu = self.res_mu_s * self.res_nu

        self.create_shaders()
        self.create_textures()

    def create_textures(self):
        """ Creates all textures required for the scattering """

        tex_format = "RGBA32" if self.use_32_bit else "RGBA16"
        img_2d, img_3d = Image.create_2d, Image.create_3d

        self.textures = {
            "transmittance": img_2d("scat-trans", self.trans_w, self.trans_h, tex_format),
            "irradiance": img_2d("scat-irrad", self.sky_w, self.sky_h, tex_format),
            "inscatter": img_3d("scat-inscat", self.res_mu_s_nu, self.res_mu, self.res_r, tex_format),  # noqa # pylint: disable=line-too-long
            "delta_e": img_2d("scat-dx-e", self.sky_w, self.sky_h, tex_format),
            "delta_sr": img_3d("scat-dx-sr", self.res_mu_s_nu, self.res_mu, self.res_r, tex_format),
            "delta_sm": img_3d("scat-dx-sm", self.res_mu_s_nu, self.res_mu, self.res_r, tex_format),
            "delta_j": img_3d("scat-dx-j", self.res_mu_s_nu, self.res_mu, self.res_r, tex_format),
        }

        for img in itervalues(self.textures):
            img.set_minfilter(SamplerState.FT_linear)
            img.set_magfilter(SamplerState.FT_linear)
            img.set_wrap_u(SamplerState.WM_clamp)
            img.set_wrap_v(SamplerState.WM_clamp)
            img.set_wrap_w(SamplerState.WM_clamp)

    def create_shaders(self):
        """ Creates all the shaders used for precomputing """
        self.shaders = {}
        resource_path = self.handle.get_shader_resource("eric_bruneton")
        for fname in listdir(resource_path):
            fpath = join(resource_path, fname)
            if isfile(fpath) and fname.endswith(".compute.glsl"):
                shader_name = fname.split(".")[0]
                shader_obj = RPLoader.load_shader(fpath)
                self.shaders[shader_name] = shader_obj

    def exec_compute_shader(self, shader_obj, shader_inputs, exec_size, workgroup_size=(16, 16, 1)):
        """ Executes a compute shader. The shader object should be a shader
        loaded with Shader.load_compute, the shader inputs should be a dict where
        the keys are the names of the shader inputs and the values are the
        inputs. The workgroup_size has to match the size defined in the
        compute shader """
        ntx = int(math.ceil(exec_size[0] / workgroup_size[0]))
        nty = int(math.ceil(exec_size[1] / workgroup_size[1]))
        ntz = int(math.ceil(exec_size[2] / workgroup_size[2]))

        nodepath = NodePath("shader")
        nodepath.set_shader(shader_obj)
        nodepath.set_shader_inputs(**shader_inputs)

        attr = nodepath.get_attrib(ShaderAttrib)
        Globals.base.graphicsEngine.dispatch_compute((ntx, nty, ntz), attr, Globals.base.win.gsg)

    def compute(self):
        """ Precomputes the scattering """

        self.debug("Precomputing ...")
        exec_cshader = self.exec_compute_shader

        # Transmittance
        exec_cshader(
            self.shaders["transmittance"], {"dest": self.textures["transmittance"]}, (self.trans_w, self.trans_h, 1)
        )

        # Delta E
        exec_cshader(
            self.shaders["delta_e"], {
                "transmittanceSampler": self.textures["transmittance"],
                "dest": self.textures["delta_e"]
            }, (self.sky_w, self.sky_h, 1)
        )

        # Delta S
        exec_cshader(
            self.shaders["delta_sm_sr"], {
                "transmittanceSampler": self.textures["transmittance"],
                "destDeltaSR": self.textures["delta_sr"],
                "destDeltaSM": self.textures["delta_sm"]
            }, (self.res_mu_s_nu, self.res_mu, self.res_r), (8, 8, 8)
        )

        # Copy deltaE to irradiance texture
        exec_cshader(
            self.shaders["copy_irradiance"], {
                "k": 0.0,
                "deltaESampler": self.textures["delta_e"],
                "dest": self.textures["irradiance"]
            }, (self.sky_w, self.sky_h, 1)
        )

        # Copy delta s into inscatter texture
        exec_cshader(
            self.shaders["copy_inscatter"], {
                "deltaSRSampler": self.textures["delta_sr"],
                "deltaSMSampler": self.textures["delta_sm"],
                "dest": self.textures["inscatter"]
            }, (self.res_mu_s_nu, self.res_mu, self.res_r), (8, 8, 8)
        )

        for order in range(2, 5):
            first = order == 2

            # Delta J
            exec_cshader(
                self.shaders["delta_j"], {
                    "transmittanceSampler": self.textures["transmittance"],
                    "deltaSRSampler": self.textures["delta_sr"],
                    "deltaSMSampler": self.textures["delta_sm"],
                    "deltaESampler": self.textures["delta_e"],
                    "dest": self.textures["delta_j"],
                    "first": first
                }, (self.res_mu_s_nu, self.res_mu, self.res_r), (8, 8, 8)
            )

            # Delta E
            exec_cshader(
                self.shaders["irradiance_n"], {
                    "transmittanceSampler": self.textures["transmittance"],
                    "deltaSRSampler": self.textures["delta_sr"],
                    "deltaSMSampler": self.textures["delta_sm"],
                    "dest": self.textures["delta_e"],
                    "first": first
                }, (self.sky_w, self.sky_h, 1)
            )

            # Delta Sr
            exec_cshader(
                self.shaders["delta_sr"], {
                    "transmittanceSampler": self.textures["transmittance"],
                    "deltaJSampler": self.textures["delta_j"],
                    "dest": self.textures["delta_sr"],
                    "first": first
                }, (self.res_mu_s_nu, self.res_mu, self.res_r), (8, 8, 8)
            )

            # Add delta E to irradiance
            exec_cshader(
                self.shaders["add_delta_e"], {
                    "deltaESampler": self.textures["delta_e"],
                    "dest": self.textures["irradiance"],
                }, (self.sky_w, self.sky_h, 1)
            )

            # Add deltaSr to inscatter texture
            exec_cshader(
                self.shaders["add_delta_sr"], {
                    "deltaSSampler": self.textures["delta_sr"],
                    "dest": self.textures["inscatter"]
                }, (self.res_mu_s_nu, self.res_mu, self.res_r), (8, 8, 8)
            )

        # Make stages available
        for stage in [self.handle.display_stage, self.handle.envmap_stage]:
            stage.set_shader_inputs(
                InscatterSampler=self.textures["inscatter"],
                transmittanceSampler=self.textures["transmittance"],
                IrradianceSampler=self.textures["irradiance"]
            )
