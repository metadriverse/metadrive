# Copyright (c) 2021 kitsune.ONE team.

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

bl_info = {
    'name': 'KITSUNETSUKI Asset Tools',
    'author': 'kitsune.ONE team',
    'version': (0, 6, 9),
    'blender': (2, 92, 0),
    'location': 'File > Import-Export',
    'description': 'Exports: glTF, VRM',
    'warning': '',
    'support': 'COMMUNITY',
    'wiki_url': '',
    'tracker_url': 'https://github.com/kitsune-ONE-team/KITSUNETSUKI-Asset-Tools/issues',
    'category': 'Import-Export',
}


def reload_package(module_dict_main):
    import importlib
    from pathlib import Path
    from typing import Any, Dict

    def reload_package_recursive(current_dir: Path, module_dict: Dict[str, Any]) -> None:
        for path in current_dir.iterdir():
            if '__init__' in str(path) or path.stem not in module_dict:
                continue

            if path.is_file() and path.suffix == '.py':
                importlib.reload(module_dict[path.stem])
            elif path.is_dir():
                reload_package_recursive(path, module_dict[path.stem].__dict__)

    reload_package_recursive(Path(__file__).parent, module_dict_main)


if 'bpy' in locals():
    reload_package(locals())


def register():
    import bpy

    if bpy.app.version[:2] < bl_info['blender'][:2]:
        cur_ver = '{}.{}'.format(*bpy.app.version[:2])
        req_ver = '{}.{}'.format(*bl_info['blender'][:2])
        raise Exception(
            f"This add-on doesn't support Blender version less than {req_ver}. "
            f'Blender version {req_ver} or greater is recommended, '
            f'but the current version is {cur_ver}'
        )

    from . import blend2gltf, blend2vrm
    blend2gltf.register(bl_info['version'])
    blend2vrm.register(bl_info['version'])


def unregister():
    import bpy

    if bpy.app.version < bl_info['blender'][:2]:
        return

    from . import blend2gltf, blend2vrm
    blend2gltf.unregister()
    blend2vrm.unregister()


if __name__ == '__main__':
    register()
