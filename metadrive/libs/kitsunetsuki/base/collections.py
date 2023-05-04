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

import bpy


def get_object_collections(obj):
    results = []

    for collection in bpy.data.collections:
        if collection.name == 'RigidBodyWorld':
            continue

        for obj2 in collection.objects:
            if obj2.name == obj.name:
                results.append(collection)

    return results


def get_object_collection(obj):
    collections = get_object_collections(obj)
    if collections:
        return collections[0]
