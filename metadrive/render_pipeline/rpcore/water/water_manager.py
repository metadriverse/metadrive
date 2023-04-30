from panda3d.core import Texture, NodePath, ShaderAttrib, LVecBase2i, PTAFloat
from panda3d.core import Vec2, PNMImage, Shader

from metadrive.render_pipeline.rpcore.globals import Globals
from metadrive.render_pipeline.rpcore.water.gpu_fft import GPUFFT

from random import random as generateRandom
from random import seed as setRandomSeed

from math import sqrt, log, cos, pi


class WaterManager:
    """ Simple wrapper around WaterDisplacement which combines 3 displacement
    maps into one, and also generates a normal map """
    def __init__(self, water_options):
        self.options = water_options
        self.options.size = 512
        self.options.wind_dir.normalize()
        self.options.wave_amplitude *= 1e-7

        self.displacement_tex = Texture("Displacement")
        self.displacement_tex.setup_2d_texture(self.options.size, self.options.size, Texture.TFloat, Texture.FRgba16)

        self.normal_tex = Texture("Normal")
        self.normal_tex.setup_2d_texture(self.options.size, self.options.size, Texture.TFloat, Texture.FRgba16)

        self.combine_shader = Shader.load_compute(Shader.SLGLSL, "/$$rp/rpcore/water/shader/combine.compute")

        self.pta_time = PTAFloat.emptyArray(1)

        # Create a gaussian random texture, as shaders aren't well suited
        # for that
        setRandomSeed(523)
        self.random_storage = PNMImage(self.options.size, self.options.size, 4)
        self.random_storage.setMaxval((2**16) - 1)

        for x in range(self.options.size):
            for y in range(self.options.size):
                rand1 = self._get_gaussian_random() / 10.0 + 0.5
                rand2 = self._get_gaussian_random() / 10.0 + 0.5
                self.random_storage.setXel(x, y, float(rand1), float(rand2), 0)
                self.random_storage.setAlpha(x, y, 1.0)

        self.random_storage_tex = Texture("RandomStorage")
        self.random_storage_tex.load(self.random_storage)
        self.random_storage_tex.set_format(Texture.FRgba16)
        self.random_storage_tex.set_minfilter(Texture.FTNearest)
        self.random_storage_tex.set_magfilter(Texture.FTNearest)

        # Create the texture wwhere the intial height (H0 + Omega0) is stored.
        self.tex_initial_height = Texture("InitialHeight")
        self.tex_initial_height.setup_2d_texture(self.options.size, self.options.size, Texture.TFloat, Texture.FRgba16)
        self.tex_initial_height.set_minfilter(Texture.FTNearest)
        self.tex_initial_height.set_magfilter(Texture.FTNearest)

        # Create the shader which populates the initial height texture
        self.shader_initial_height = Shader.load_compute(
            Shader.SLGLSL, "/$$rp/rpcore/water/shader/initial_height.compute"
        )
        self.node_initial_height = NodePath("initialHeight")
        self.node_initial_height.set_shader(self.shader_initial_height)
        self.node_initial_height.set_shader_input("dest", self.tex_initial_height)
        self.node_initial_height.set_shader_input("N", LVecBase2i(self.options.size))
        self.node_initial_height.set_shader_input("patchLength", self.options.patch_length)
        self.node_initial_height.set_shader_input("windDir", self.options.wind_dir)
        self.node_initial_height.set_shader_input("windSpeed", self.options.wind_speed)
        self.node_initial_height.set_shader_input("waveAmplitude", self.options.wave_amplitude)
        self.node_initial_height.set_shader_input("windDependency", self.options.wind_dependency)
        self.node_initial_height.set_shader_input("randomTex", self.random_storage_tex)

        self.attr_initial_height = self.node_initial_height.get_attrib(ShaderAttrib)

        self.height_textures = []
        for i in range(3):
            tex = Texture("Height")
            tex.setup_2d_texture(self.options.size, self.options.size, Texture.TFloat, Texture.FRgba16)
            tex.set_minfilter(Texture.FTNearest)
            tex.set_magfilter(Texture.FTNearest)
            tex.set_wrap_u(Texture.WMClamp)
            tex.set_wrap_v(Texture.WMClamp)
            self.height_textures.append(tex)

        # Also create the shader which updates the spectrum
        self.shader_update = Shader.load_compute(Shader.SLGLSL, "/$$rp/rpcore/water/shader/update.compute")
        self.node_update = NodePath("update")
        self.node_update.set_shader(self.shader_update)
        self.node_update.set_shader_input("outH0x", self.height_textures[0])
        self.node_update.set_shader_input("outH0y", self.height_textures[1])
        self.node_update.set_shader_input("outH0z", self.height_textures[2])
        self.node_update.set_shader_input("initialHeight", self.tex_initial_height)
        self.node_update.set_shader_input("N", LVecBase2i(self.options.size))
        self.node_update.set_shader_input("time", self.pta_time)
        self.attr_update = self.node_update.get_attrib(ShaderAttrib)

        # Create 3 FFTs
        self.fftX = GPUFFT(self.options.size, self.height_textures[0], self.options.normalization_factor)
        self.fftY = GPUFFT(self.options.size, self.height_textures[1], self.options.normalization_factor)
        self.fftZ = GPUFFT(self.options.size, self.height_textures[2], self.options.normalization_factor)

        self.combine_node = NodePath("Combine")
        self.combine_node.set_shader(self.combine_shader)
        self.combine_node.set_shader_input("displacementX", self.fftX.get_result_texture())
        self.combine_node.set_shader_input("displacementY", self.fftY.get_result_texture())
        self.combine_node.set_shader_input("displacementZ", self.fftZ.get_result_texture())
        self.combine_node.set_shader_input("normalDest", self.normal_tex)
        self.combine_node.set_shader_input("displacementDest", self.displacement_tex)
        self.combine_node.set_shader_input("N", LVecBase2i(self.options.size))
        self.combine_node.set_shader_input("choppyScale", self.options.choppy_scale)
        self.combine_node.set_shader_input("gridLength", self.options.patch_length)
        # Store only the shader attrib as this is way faster
        self.attr_combine = self.combine_node.get_attrib(ShaderAttrib)

    def _get_gaussian_random(self):
        """ Returns a gaussian random number """
        u1 = generateRandom()
        u2 = generateRandom()
        if u1 < 1e-6:
            u1 = 1e-6
        return sqrt(-2 * log(u1)) * cos(2 * pi * u2)

    def setup(self):
        """ Setups the manager """

        Globals.base.graphicsEngine.dispatch_compute(
            (self.options.size // 16, self.options.size // 16, 1), self.attr_initial_height, Globals.base.win.get_gsg()
        )

    def get_displacement_texture(self):
        """ Returns the displacement texture, storing the 3D Displacement in
        the RGB channels """
        return self.displacement_tex

    def get_normal_texture(self):
        """ Returns the normal texture, storing the normal in world space in
        the RGB channels """
        return self.normal_tex

    def update(self):
        """ Updates the displacement / normal map """

        self.pta_time[0] = 1 + Globals.clock.get_frame_time() * self.options.time_scale

        Globals.base.graphicsEngine.dispatch_compute(
            (self.options.size // 16, self.options.size // 16, 1), self.attr_update, Globals.base.win.get_gsg()
        )

        self.fftX.execute()
        self.fftY.execute()
        self.fftZ.execute()

        # Execute the shader which combines the 3 displacement maps into
        # 1 displacement texture and 1 normal texture. We could use dFdx in
        # the fragment shader, however that gives no accurate results as
        # dFdx returns the same value for a 2x2 pixel block
        Globals.base.graphicsEngine.dispatch_compute(
            (self.options.size // 16, self.options.size // 16, 1), self.attr_combine, Globals.base.win.get_gsg()
        )
