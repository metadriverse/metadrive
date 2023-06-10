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


def get_root_node(node_tree, type_):
    for node in node_tree.nodes:
        if node.type == type_:
            if type_ == 'OUTPUT_MATERIAL':
                if not node.is_active_output:
                    continue
            return node


def get_from_node(node_tree, type, to_node, from_socket_name, to_socket_name):
    for link in node_tree.links:
        # if (link.from_node.type == type and link.to_node == to_node and
        #         link.from_socket.name == from_socket_name and
        #         link.to_socket.name == to_socket_name):
        #     return link.from_node

        # right side matched:
        # -> [To Socket] To Node
        if (link.to_node == to_node and link.to_socket.name == to_socket_name):
            # left side matched:
            # From Node [From Socket] ->
            if (link.from_node.type == type and link.from_socket.name == from_socket_name):
                return link.from_node

            # through mix shader:
            # Mix Shader [Shader] ->
            elif (link.from_node.type == 'MIX_SHADER' and link.from_socket.name == 'Shader'):
                mix = link.from_node
                for input_ in mix.inputs:
                    if input_.name == 'Shader':
                        node = get_from_node(node_tree, type, mix, from_socket_name, 'Shader')
                        if node:
                            return node

            # through math:
            # Math [Value] ->
            elif (link.from_node.type == 'MATH' and link.from_socket.name == 'Value'):
                math = link.from_node
                for input_ in math.inputs:
                    if input_.name == 'Value':
                        node = get_from_node(node_tree, type, math, from_socket_name, 'Value')
                        if node:
                            return node

            # through separate RGB:
            # Separate RGB [Value] ->
            elif (link.from_node.type == 'SEPRGB' and link.from_socket.name in ('R', 'G', 'B')):
                seprgb = link.from_node
                node = get_from_node(node_tree, type, seprgb, from_socket_name, 'Image')
                if node:
                    return node
