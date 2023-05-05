import colorsys
import math

from panda3d.core import CS_linear, PNMImage, PNMFileTypeRegistry


class Palette2LUT(object):
    def __init__(self):
        self.tile_width = 64  # cube width -> red-green tile width
        self.atlas_width = int(math.sqrt(self.tile_width))  # altas width in number of tiles
        lut_size = self.tile_width * self.atlas_width

        reg = PNMFileTypeRegistry.get_global_ptr()
        ftype = reg.get_type_from_extension('a.png')

        self.lut = PNMImage(lut_size, lut_size, 3, 255, ftype, CS_linear)

    def set_color(self, color_in, color_out):
        r, g, b = color_in

        tile_x = int(round(r * (self.tile_width - 1)))
        tile_y = int(round((1 - g) * (self.tile_width - 1)))

        tile_row = int(round(b * (self.tile_width - 1))) // self.atlas_width
        tile_col = int(round(b * (self.tile_width - 1))) % self.atlas_width

        x = tile_col * self.tile_width + tile_x
        y = tile_row * self.tile_width + tile_y

        r, g, b = color_out
        self.lut.set_xel(x, y, r, g, b)

    def convert(self, ipath):
        if ipath is None:
            for r in range(256):
                for g in range(256):
                    for b in range(256):
                        color = (r / 255, g / 255, b / 255)
                        self.set_color(color, color)
            return

        palette = PNMImage()
        palette.read(ipath)

        quality = 256
        for h in range(quality + 1):
            x = int(round(h / quality * (palette.get_read_x_size() - 1)))
            h0, s0, v0 = colorsys.rgb_to_hsv(*palette.get_xel(x, 0))

            for s in range(quality + 1):
                for v in range(quality + 1):
                    color_in = colorsys.hsv_to_rgb(h / quality, s / quality, v / quality)
                    color_out = colorsys.hsv_to_rgb(h0, s / quality, v / quality)
                    self.set_color(color_in, color_out)

    def save(self, opath):
        self.lut.write(opath)
