from __future__ import division, print_function

from panda3d.core import PNMImage, Vec3


def approx(roughness):
    return 1.0 - 0.5 * roughness


for i in range(11):
    r = i / 10.0
    fname = "batch_compare/Gold-R" + str(r) + ".png"

    img = PNMImage(fname)
    # color = (img.get_xel(img.get_x_size() // 2 + 3, img.get_y_size() // 2))
    color = (img.get_xel(256, 47))

    color.x = pow(color.x, 2.2)
    color.y = pow(color.y, 2.2)
    color.z = pow(color.z, 2.2)

    # print(color)

    basecolor = Vec3(1, 0.867136, 0.358654)

    ref_r, apprx_r = color.x, approx(r) * (0.5 + 0.5 * basecolor.x)
    ref_g, apprx_g = color.y, approx(r) * (0.5 + 0.5 * basecolor.y)
    ref_b, apprx_b = color.z, approx(r) * (0.5 + 0.5 * basecolor.z)
    print("Roughness:", r, ", color = ", ref_r, ref_g, ref_b, "vs", apprx_r, apprx_g, apprx_b)
    l = abs(ref_r - apprx_r) + abs(ref_g - apprx_g) + abs(ref_b - apprx_b)
    print("  --> ", l)
    # print("Roughness:", r, ", color = ", ref, "vs", apprx, " =\t", abs(ref - apprx) * 100.0)
