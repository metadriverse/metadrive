#!/usr/bin/env python3
# Copyright (c) 2022 kitsune.ONE team.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse

from metadrive.third_party.kitsunetsuki.lut import Palette2LUT

from panda3d.core import Filename, VirtualFileSystem, get_model_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, help='output file')
    parser.add_argument('--input', type=str, help='input file', default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    pl = Palette2LUT()
    pl.convert(args.input)
    pl.save(args.output)


if __name__ == '__main__':
    main()
