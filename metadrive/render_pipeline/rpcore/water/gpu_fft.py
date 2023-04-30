from panda3d.core import PNMImage, Texture, LVecBase3d, NodePath, Shader, LVecBase3i
from panda3d.core import ShaderAttrib, LVecBase2i, Vec2

from metadrive.render_pipeline.rpcore.globals import Globals

import math


class GPUFFT:
    """ This is a collection of compute shaders to generate the inverse
    fft efficiently on the gpu, with butterfly FFT and precomputed weights """
    def __init__(self, size, source_tex, normalization_factor):
        """ Creates a new fft instance. The source texture has to specified
        from the begining, as the shaderAttributes are pregenerated for
        performance reasons """

        self.size = size
        self.log2_size = int(math.log(size, 2))
        self.normalization_factor = normalization_factor

        # Create a ping and a pong texture, because we can't write to the
        # same texture while reading to it (that would lead to unexpected
        # behaviour, we could solve that by using an appropriate thread size,
        # but it works fine so far)
        self.ping_texture = Texture("FFTPing")
        self.ping_texture.setup_2d_texture(self.size, self.size, Texture.TFloat, Texture.FRgba32)
        self.pong_texture = Texture("FFTPong")
        self.pong_texture.setup_2d_texture(self.size, self.size, Texture.TFloat, Texture.FRgba32)
        self.source_tex = source_tex

        for tex in [self.ping_texture, self.pong_texture, source_tex]:
            tex.set_minfilter(Texture.FTNearest)
            tex.set_magfilter(Texture.FTNearest)
            tex.set_wrap_u(Texture.WMClamp)
            tex.set_wrap_v(Texture.WMClamp)

        # Pregenerate weights & indices for the shaders
        self._compute_weighting()

        # Pre generate the shaders, we have 2 passes: Horizontal and Vertical
        # which both execute log2(N) times with varying radii
        self.horizontal_fft_shader = Shader.load_compute(
            Shader.SLGLSL, "/$$rp/rpcore/water/shader/horizontal_fft.compute"
        )
        self.horizontal_fft = NodePath("HorizontalFFT")
        self.horizontal_fft.set_shader(self.horizontal_fft_shader)
        self.horizontal_fft.set_shader_input("precomputedWeights", self.weights_lookup_tex)
        self.horizontal_fft.set_shader_input("N", LVecBase2i(self.size))

        self.vertical_fft_shader = Shader.load_compute(Shader.SLGLSL, "/$$rp/rpcore/water/shader/vertical_fft.compute")
        self.vertical_fft = NodePath("VerticalFFT")
        self.vertical_fft.set_shader(self.vertical_fft_shader)
        self.vertical_fft.set_shader_input("precomputedWeights", self.weights_lookup_tex)
        self.vertical_fft.set_shader_input("N", LVecBase2i(self.size))

        # Create a texture where the result is stored
        self.result_texture = Texture("Result")
        self.result_texture.setup2dTexture(self.size, self.size, Texture.TFloat, Texture.FRgba16)
        self.result_texture.set_minfilter(Texture.FTLinear)
        self.result_texture.set_magfilter(Texture.FTLinear)

        # Prepare the shader attributes, so we don't have to regenerate them
        # every frame -> That is VERY slow (3ms per fft instance)
        self._prepare_attributes()

    def get_result_texture(self):
        """ Returns the result texture, only contains valid data after execute
        was called at least once """
        return self.result_texture

    def _generate_indices(self, storage_a, storage_b):
        """ This method generates the precompute indices, see
        http://cnx.org/content/m12012/latest/image1.png """
        num_iter = self.size
        offset = 1
        step = 0
        for i in range(self.log2_size):
            num_iter = num_iter >> 1
            step = offset
            for j in range(self.size):
                goLeft = (j // step) % 2 == 1
                index_a, index_b = 0, 0
                if goLeft:
                    index_a, index_b = j - step, j
                else:
                    index_a, index_b = j, j + step

                storage_a[i][j] = index_a
                storage_b[i][j] = index_b
            offset = offset << 1

    def _generate_weights(self, storage):
        """ This method generates the precomputed weights """

        # Using a custom pi variable should force the calculations to use
        # high precision (I hope so)
        pi = 3.141592653589793238462643383
        num_iter = self.size // 2
        num_k = 1
        resolution_float = float(self.size)
        for i in range(self.log2_size):
            start = 0
            end = 2 * num_k
            for b in range(num_iter):
                K = 0
                for k in range(start, end, 2):
                    fK = float(K)
                    f_num_iter = float(num_iter)
                    weight_a = Vec2(
                        math.cos(2.0 * pi * fK * f_num_iter / resolution_float),
                        -math.sin(2.0 * pi * fK * f_num_iter / resolution_float)
                    )
                    weight_b = Vec2(
                        -math.cos(2.0 * pi * fK * f_num_iter / resolution_float),
                        math.sin(2.0 * pi * fK * f_num_iter / resolution_float)
                    )
                    storage[i][k // 2] = weight_a
                    storage[i][k // 2 + num_k] = weight_b
                    K += 1
                start += 4 * num_k
                end = start + 2 * num_k

            num_iter = num_iter >> 1
            num_k = num_k << 1

    def _reverse_row(self, indices):
        """ Reverses the bits in the given row. This is required for inverse
        fft (actually we perform a normal fft, but reversing the bits gives
        us an inverse fft) """
        mask = 0x1
        for j in range(self.size):
            val = 0x0
            temp = int(indices[j])  # Int is required, for making a copy
            for i in range(self.log2_size):
                t = mask & temp
                val = (val << 1) | t
                temp = temp >> 1
            indices[j] = val

    def _compute_weighting(self):
        """ Precomputes the weights & indices, and stores them in a texture """
        indices_a = [[0 for i in range(self.size)] for k in range(self.log2_size)]
        indices_b = [[0 for i in range(self.size)] for k in range(self.log2_size)]
        weights = [[Vec2(0.0) for i in range(self.size)] for k in range(self.log2_size)]

        # Pre-Generating indices ..
        self._generate_indices(indices_a, indices_b)
        self._reverse_row(indices_a[0])
        self._reverse_row(indices_b[0])

        # Pre-Generating weights .."
        self._generate_weights(weights)

        # Create storage for the weights & indices
        self.weights_lookup = PNMImage(self.size, self.log2_size, 4)
        self.weights_lookup.setMaxval((2**16) - 1)
        self.weights_lookup.fill(0.0)

        # Populate storage
        for x in range(self.size):
            for y in range(self.log2_size):
                index_a = indices_a[y][x]
                index_b = indices_b[y][x]
                weight = weights[y][x]

                self.weights_lookup.set_red(x, y, index_a / float(self.size))
                self.weights_lookup.set_green(x, y, index_b / float(self.size))
                self.weights_lookup.set_blue(x, y, weight.x * 0.5 + 0.5)
                self.weights_lookup.set_alpha(x, y, weight.y * 0.5 + 0.5)

        # Convert storage to texture so we can use it in a shader
        self.weights_lookup_tex = Texture("Weights Lookup")
        self.weights_lookup_tex.load(self.weights_lookup)
        self.weights_lookup_tex.set_format(Texture.FRgba16)
        self.weights_lookup_tex.set_minfilter(Texture.FTNearest)
        self.weights_lookup_tex.set_magfilter(Texture.FTNearest)
        self.weights_lookup_tex.set_wrap_u(Texture.WMClamp)
        self.weights_lookup_tex.set_wrap_v(Texture.WMClamp)

    def _prepare_attributes(self):
        """ Prepares all shaderAttributes, so that we have a list of
        ShaderAttributes we can simply walk through in the update method,
        that is MUCH faster than using set_shader_input, as each call to
        set_shader_input forces the generation of a new ShaderAttrib """
        self.attributes = []
        textures = [self.ping_texture, self.pong_texture]

        current_index = 0
        firstPass = True

        # Horizontal
        for step in range(self.log2_size):
            source = textures[current_index]
            dest = textures[1 - current_index]

            if firstPass:
                source = self.source_tex
                firstPass = False

            index = self.log2_size - step - 1
            self.horizontal_fft.set_shader_input("source", source)
            self.horizontal_fft.set_shader_input("dest", dest)
            self.horizontal_fft.set_shader_input("butterflyIndex", LVecBase2i(index))
            self._queue_shader(self.horizontal_fft)
            current_index = 1 - current_index

        # Vertical
        for step in range(self.log2_size):
            source = textures[current_index]
            dest = textures[1 - current_index]
            is_last_pass = step == self.log2_size - 1
            if is_last_pass:
                dest = self.result_texture
            index = self.log2_size - step - 1
            self.vertical_fft.set_shader_input("source", source)
            self.vertical_fft.set_shader_input("dest", dest)
            self.vertical_fft.set_shader_input("isLastPass", is_last_pass)
            self.vertical_fft.set_shader_input("normalizationFactor", self.normalization_factor)
            self.vertical_fft.set_shader_input("butterflyIndex", LVecBase2i(index))
            self._queue_shader(self.vertical_fft)

            current_index = 1 - current_index

    def execute(self):
        """ Executes the inverse fft once """
        for attr in self.attributes:
            self._execute_shader(attr)

    def _queue_shader(self, node):
        """ Internal method to fetch the ShaderAttrib of a node and store it
        in the update queue """
        sattr = node.getAttrib(ShaderAttrib)
        self.attributes.append(sattr)

    def _execute_shader(self, sattr):
        """ Internal method to execute a shader by a given ShaderAttrib """
        Globals.base.graphicsEngine.dispatch_compute(
            (self.size // 16, self.size // 16, 1), sattr, Globals.base.win.get_gsg()
        )
