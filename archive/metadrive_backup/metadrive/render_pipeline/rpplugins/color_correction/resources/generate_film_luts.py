"""

Script to convert spi1d files to luts

"""

from __future__ import print_function

import os
from panda3d.core import PNMImage

for f in os.listdir("film_luts_raw"):
    if not f.endswith(".spi1d"):
        continue

    input_file = "film_luts_raw/" + f

    print("Processing", input_file)
    output_name = input_file.replace("\\", "/").split("/")[-1].replace(".spi1d", "")

    with open(input_file, "r") as handle:
        content = [i.strip() for i in handle.readlines()]

    header = [
        "Version 1",
        "From 0.0 1.0",
        "Length 1024",
        "Components 3",
        "{",
    ]

    for i, line in enumerate(header):
        if content[i] != line:
            print("Invalid header! Expected", line, "but got", content[i])
            break
    else:

        content = content[len(header):-1]

        values = [[], [], []]

        def srgb_to_linear(channel):
            if channel <= 0.04045:
                return channel / 12.92
            else:
                return pow((channel + 0.055) / (1.0 + 0.055), 2.4)

        for i in xrange(1024):
            line = [float(k) for k in content[i].split()]

            for k in xrange(3):
                values[k].append(srgb_to_linear(line[k]))

        def to_linear(v):
            return float(v) / float(64 - 1)

        def to_linear_inv(v):
            return 1 - to_linear(v)

        def lookup_value(v, values):
            return values[int(v * (len(values) - 1))]

        # Generate lut
        img = PNMImage(64 * 8, 64 * 8, 3, 2**16 - 1)
        for r in xrange(64):
            for g in xrange(64):
                for b in xrange(64):
                    slice_offset_x = (b % 8) * 64
                    slice_offset_y = (b // 8) * 64

                    fr, fg, fb = to_linear(r), to_linear_inv(g), to_linear(b)

                    fr = lookup_value(fr, values[0])
                    fg = lookup_value(fg, values[1])
                    fb = lookup_value(fb, values[2])

                    img.set_xel(r + slice_offset_x, g + slice_offset_y, fr, fg, fb)

        img.write("film_luts/" + output_name + ".png")
