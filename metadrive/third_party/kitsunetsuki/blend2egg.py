#!/usr/bin/env python3
# Copyright (c) 2020 kitsune.ONE team.

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


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('inputs', type=str, help='Input .blend file path.', nargs='*')
    parser.add_argument(
        '-b', '--background', action='store_true', required=False, help="Blender's argument placeholder."
    )
    parser.add_argument('-P', '--python', type=str, required=False, help="Blender's argument placeholder.")
    parser.add_argument('-o', '--output', type=str, required=False, help='Output .egg file path.')
    parser.add_argument(
        '-e', '--export', type=str, required=False, default='scene', help='Export type: scene/animation/collision/all'
    )
    parser.add_argument('-r', '--render', type=str, required=False, default='default', help='Render type: default/rp.')
    parser.add_argument('-x', '--exec', type=str, required=False, help='Internal script name to execute.')
    parser.add_argument('-a', '--action', type=str, required=False, help='Action name to export.')
    parser.add_argument('-sp', '--speed', type=float, required=False, help='Animations speed scale.')
    parser.add_argument('-sc', '--scale', type=float, required=False, help='Geom scale.')
    parser.add_argument('-m', '--merge', action='store_true', help='Merge objects and meshes inside the collection.')
    parser.add_argument(
        '-k', '--keep', action='store_true', help='Keep the original objects and meshes before merging.'
    )
    parser.add_argument('-nuv', '--no-extra-uv', action='store_true', help="Don't export extra non-primary UV.")
    parser.add_argument(
        '-nmat', '--no-materials', action='store_true', help="Don't export materials (skips textures aswell)."
    )
    parser.add_argument('-ntex', '--no-textures', action='store_true', help="Don't export textures.")
    parser.add_argument(
        '-etex', '--empty-textures', action='store_true', help="Use placeholder images for empty texture slots."
    )
    parser.add_argument(
        '-sorg', '--set-origin', action='store_true', help="Set origin to center of bounds for collisions."
    )

    return parser.parse_args()


def main():
    args = parse_args()
    from panda3d.egg import EggData
    from metadrive.third_party.kitsunetsuki.exporter.egg import EggExporter
    e = EggExporter(args)
    out = e.convert()

    if args.output:
        if isinstance(out, EggData):
            out.write_egg(args.output)
        else:
            out.write(args.output)
    else:
        print(out)


if __name__ == '__main__':
    main()
