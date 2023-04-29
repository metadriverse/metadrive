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

# pylint: skip-file
# flake8: noqa

# Jitters from
# https://github.com/playdeadgames/temporal/blob/master/Assets/Scripts/FrustumJitter.cs

from __future__ import division
from panda3d.core import Vec2

__all__ = ("JITTERS",)

JITTERS = {

    "still": [
        Vec2(0.0, 0.0),
    ],

    "uniform2": [
        Vec2(-0.25, -0.25), # ll
        Vec2( 0.25,  0.25), # ur
    ],

    "uniform4": [
        Vec2(-0.25, -0.25), # ll
        Vec2( 0.25, -0.25), # lr
        Vec2( 0.25,  0.25), # ur
        Vec2(-0.25,  0.25), # ul
    ],

    "uniform4_helix": [
        Vec2(-0.25, -0.25), # ll  3  1
        Vec2( 0.25,  0.25), # ur   \/|
        Vec2( 0.25, -0.25), # lr   /\|
        Vec2(-0.25,  0.25), # ul  0  2
    ],

    "uniform4_double_helix": [
        Vec2(-0.25, -0.25), # ll  3  1
        Vec2( 0.25,  0.25), # ur   \/|
        Vec2( 0.25, -0.25), # lr   /\|
        Vec2(-0.25,  0.25), # ul  0  2
        Vec2(-0.25, -0.25), # ll  6--7
        Vec2( 0.25, -0.25), # lr   \
        Vec2(-0.25,  0.25), # ul    \
        Vec2( 0.25,  0.25), # ur  4--5
    ],

    "skew_butterly": [
        Vec2(-0.250, -0.250),
        Vec2( 0.250,  0.250),
        Vec2( 0.125, -0.125),
        Vec2(-0.125,  0.125),
    ],

    "rotated4": [
        Vec2(-0.125, -0.375), # ll
        Vec2( 0.375, -0.125), # lr
        Vec2( 0.125,  0.375), # ur
        Vec2(-0.375,  0.125), # ul
    ],

    "rotated4_helix": [
        Vec2(-0.125, -0.375), # ll  3  1
        Vec2( 0.125,  0.375), # ur   \/|
        Vec2( 0.375, -0.125), # lr   /\|
        Vec2(-0.375,  0.125), # ul  0  2
    ],

    "rotated4_helix2": [
        Vec2(-0.125, -0.375), # ll  2--1
        Vec2( 0.125,  0.375), # ur   \/
        Vec2(-0.375,  0.125), # ul   /\
        Vec2( 0.375, -0.125), # lr  0  3
    ],

    "poisson10": [
        Vec2(-0.16795960 * 0.25,  0.65544910 * 0.25),
        Vec2(-0.69096030 * 0.25,  0.59015970 * 0.25),
        Vec2( 0.49843820 * 0.25,  0.83099720 * 0.25),
        Vec2( 0.17230150 * 0.25, -0.03882703 * 0.25),
        Vec2(-0.60772670 * 0.25, -0.06013587 * 0.25),
        Vec2( 0.65606390 * 0.25,  0.24007600 * 0.25),
        Vec2( 0.80348370 * 0.25, -0.48096900 * 0.25),
        Vec2( 0.33436540 * 0.25, -0.73007030 * 0.25),
        Vec2(-0.47839520 * 0.25, -0.56005300 * 0.25),
        Vec2(-0.12388120 * 0.25, -0.96633990 * 0.25),
    ],

    "pentagram": [
        Vec2( 0.000000 * 0.5,  0.525731 * 0.5), # head
        Vec2(-0.309017 * 0.5, -0.425325 * 0.5), # lleg
        Vec2( 0.500000 * 0.5,  0.162460 * 0.5), # rarm
        Vec2(-0.500000 * 0.5,  0.162460 * 0.5), # larm
        Vec2( 0.309017 * 0.5, -0.425325 * 0.5), # rleg
    ],
}


def halton_seq(prime, index=1):
    """ Low discrepancy halton sequence with a given prime """
    r, f, i = 0.0, 1.0, index
    while i > 0:
        f /= prime
        r += f * (i % prime)
        i = int(i / float(prime))
    return r

# Initialize halton sequences
for seq_num in [8, 16, 32, 256]:
    sequence = []
    for seq_iter in range(seq_num):
        point = Vec2(halton_seq(2, seq_iter + 1) - 0.5, halton_seq(3, seq_iter + 1) - 0.5)
        sequence.append(point)
    JITTERS["halton" + str(seq_num)] = sequence
