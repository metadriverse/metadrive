"""

Uses mitsuba to generate the environment brdf

"""

from __future__ import print_function

import os
import sys
import math
from panda3d.core import PNMImage, load_prc_file_data, Vec3

load_prc_file_data("", "notify-level error")
load_prc_file_data("", "notify-level-pnmimage error")

configs = {
    "normal": {
        "out_dir": "slices",
        "out_name": "env_brdf_{}.png",
        "template_suffix": "",
        "sequence": xrange(7),
        "samples": 32,
    },
    "metallic": {
        "out_dir": "slices_metal",
        "out_name": "env_brdf.png",
        "template_suffix": "-metal",
        "sequence": [1],
        "samples": 32,
    },
    "clearcoat": {
        "out_dir": "slices_coat",
        "out_name": "env_brdf.png",
        "template_suffix": "-coat",
        "sequence": [1],
        "samples": 2048,
    }
}

# configs_to_run = ["normal", "metallic", "clearcoat"]
configs_to_run = ["clearcoat"]

for config_name in configs_to_run:

    config = configs[config_name]

    if not os.path.isdir(config["out_dir"]):
        os.makedirs(config["out_dir"])


    for pass_index in config["sequence"]:

        ior = 1.01 + 0.2 * pass_index

        dest_size = 512
        dest_h = 32
        dest = PNMImage(dest_size, dest_h)

        # run mitsuba
        print("Running mitsuba for ior =", ior, "( index =", pass_index,")")
        with open("res/scene" + config["template_suffix"] + ".templ.xml", "r") as handle:
            content = handle.read()

        content = content.replace("%IOR%", str(ior))
        content = content.replace("%SAMPLES%", str(config["samples"]))

        with open("res/scene.xml", "w") as handle:
            handle.write(content)

        os.system("run_mitsuba.bat")

        img = PNMImage("scene.png")
        source_w = img.get_x_size()

        print("Converting ..")

        indices = []
        nxv_values = []

        # Generate nonlinear NxV sequence
        for i in xrange(source_w):
            v = 1 - i / float(source_w)
            NxV = math.sqrt(1 - v*v)
            nxv_values.append(NxV)

        # Generate lerp indices and weights
        for x in xrange(dest_size):
            NxV = (x) / float(dest_size)
            index = 0
            for i, s_nxv in enumerate(reversed(nxv_values)):
                if NxV >= s_nxv:
                    index = i
                    break
            index = len(nxv_values) - index - 1
            next_index = index + 1 if index < dest_size - 1 else index

            curr_nxv = nxv_values[index]
            next_nxv = nxv_values[next_index]

            lerp_factor = (NxV - curr_nxv) / max(1e-10, abs(next_nxv - curr_nxv))
            lerp_factor = max(0.0, min(1.0, lerp_factor))
            indices.append((index, next_index, lerp_factor))

        # Generate the final linear lut using the lerp weights
        for y in xrange(dest_h):
            for x in xrange(dest_size):
                curr_i, next_i, lerp = indices[x]
                curr_v = img.get_xel(curr_i, y)
                next_v = img.get_xel(next_i, y)
                dest.set_xel(x, y, curr_v * (1 - lerp) + next_v * lerp)


        out_name = config["out_name"].replace("{}", str(pass_index))
        dest.write(config["out_dir"] + "/" + out_name)

try:
    os.remove("scene.png")
except:
    pass
