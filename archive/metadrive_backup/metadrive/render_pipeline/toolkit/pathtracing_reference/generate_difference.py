"""

Compares both renders

"""

from __future__ import print_function

from panda3d.core import PNMImage

source_a = "scene.png"
source_b = "scene-rp.png"

write_diff_img = False

img_a = PNMImage(source_a)
img_b = PNMImage(source_b)

w, h = img_a.get_x_size(), img_a.get_y_size()
img_dest = PNMImage(w, h, 3)

error_scale = 10.0

total_diff = 0.0

for x in xrange(w):
    for y in xrange(h):
        val_a = img_a.get_xel(x, y)
        val_b = img_b.get_xel(x, y)

        abs_diff = (val_a - val_b) * error_scale
        r, g, b = abs(abs_diff.x), abs(abs_diff.y), abs(abs_diff.z)

        img_dest.set_xel(x, y, r, g, b)
        total_diff += r + g + b

total_diff /= float(w * h)
total_diff /= error_scale

print("Average difference: ", total_diff, " in RGB: ", total_diff * 255)

img_dest.write("difference.png")
