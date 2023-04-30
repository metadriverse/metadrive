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
from metadrive.render_pipeline.rplibs.six import iteritems

from panda3d.core import PTAFloat, PTALVecBase3f, PTALMatrix4f, PTALVecBase2f
from panda3d.core import PTALVecBase4f, PTALMatrix3f, PTAInt, TypeRegistry, PTALVecBase2i

from metadrive.render_pipeline.rpcore.rpobject import RPObject


class SimpleInputBlock(RPObject):
    """ Simplest possible uniform buffer which just stores a set of values.
    This does not use any fancy uniform buffer objects under the hood, and
    instead just sets every value as a shader input. """
    def __init__(self, name):
        """ Creates the ubo with the given name """
        RPObject.__init__(self)
        self._inputs = {}
        self.name = name

    def add_input(self, name, value):
        """ Adds a new input to the UBO """
        self._inputs[self.name + "." + name] = value

    def bind_to(self, target):
        """ Binds the UBO to a target """
        target.set_shader_inputs(**self._inputs)


class GroupedInputBlock(RPObject):
    """ Grouped uniform buffer which either uses PointerToArray's to efficiently
    store and update the shader inputs, or in case of uniform buffer object (UBO)
    support, uses these to pass the inputs to the shaders. """

    # Keeps track of the global allocated input blocks to be able to assign
    # a unique binding to all of them
    UBO_BINDING_INDEX = 0

    # Mapping of the pta-types to glsl types, and vice versa
    PTA_MAPPINGS = {
        PTAInt: "int",
        PTAFloat: "float",
        PTALVecBase2f: "vec2",
        PTALVecBase2i: "ivec2",
        PTALVecBase3f: "vec3",
        PTALVecBase4f: "vec4",
        PTALMatrix3f: "mat3",
        PTALMatrix4f: "mat4",
    }

    def __init__(self, name):
        """ Constructs the input block with a given name """
        RPObject.__init__(self)
        self.ptas = {}
        self._inputs = {}
        self.name = name
        self.use_ubo = bool(TypeRegistry.ptr().find_type("GLUniformBufferContext"))

        # Acquire a unique index for each UBO to store its binding
        self.bind_id = GroupedInputBlock.UBO_BINDING_INDEX
        GroupedInputBlock.UBO_BINDING_INDEX += 1

        if self.bind_id == 0:
            # Only output the bind support debug output once (for the first ubo)
            self.debug("Native UBO support =", self.use_ubo)

    def register_pta(self, name, input_type):
        """ Registers a new input, type should be a glsl type """
        pta = self.glsl_type_to_pta(input_type).empty_array(1)
        self.ptas[name] = pta
        if self.use_ubo:
            self._inputs[self.name + "_UBO." + name] = pta
        else:
            self._inputs[self.name + "." + name] = pta

    def pta_to_glsl_type(self, pta_handle):
        """ Converts a PtaXXX to a glsl type """
        for pta_type, glsl_type in iteritems(GroupedInputBlock.PTA_MAPPINGS):
            if isinstance(pta_handle, pta_type):
                return glsl_type
        self.error("Unrecognized PTA type:", pta_handle)

    def glsl_type_to_pta(self, glsl_type):
        """ Converts a glsl type to a PtaXXX type """
        for key, val in iteritems(GroupedInputBlock.PTA_MAPPINGS):
            if val == glsl_type:
                return key
        self.error("Could not resolve GLSL type:", glsl_type)

    def bind_to(self, target):
        """ Binds all inputs of this UBO to the given target, which may be
        either a RenderTarget or a NodePath """

        target.set_shader_inputs(**self._inputs)

    def update_input(self, name, value):
        """ Updates an existing input """
        self.ptas[name][0] = value

    def get_input(self, name):
        """ Returns the value of an existing input """
        return self.ptas[name][0]

    def generate_shader_code(self):  # pylint: disable=too-many-branches
        """ Generates the GLSL shader code to use the UBO """

        content = "#pragma once\n\n"
        content += "// Autogenerated by the render pipeline\n"
        content += "// Do not edit! Your changes will be lost.\n\n"

        structs = {}
        inputs = []

        for input_name, handle in iteritems(self.ptas):
            parts = input_name.split(".")

            # Single input, simply add it to the input list
            if len(parts) == 1:
                inputs.append(self.pta_to_glsl_type(handle) + " " + input_name + ";")

            # Nested input, like scattering.sun_color
            elif len(parts) == 2:
                struct_name = parts[0]
                actual_input_name = parts[1]
                if struct_name in structs:
                    # Struct is already defined, add member definition
                    structs[struct_name].append(self.pta_to_glsl_type(handle) + " " + actual_input_name + ";")
                else:
                    # Construct a new struct and add it to the list of inputs
                    inputs.append(struct_name + "_UBOSTRUCT " + struct_name + ";")
                    structs[struct_name] = [self.pta_to_glsl_type(handle) + " " + actual_input_name + ";"]

            # Nested input, like scattering.some_setting.sun_color, not supported yet
            else:
                self.warn("Structure definition too nested, not supported (yet):", input_name)

        # Add structures
        for struct_name, members in iteritems(structs):
            content += "struct " + struct_name + "_UBOSTRUCT {\n"
            for member in members:
                content += " " * 4 + member + "\n"
            content += "};\n\n"

        # Add actual inputs
        if len(inputs) < 1:
            self.debug("No UBO inputs present for", self.name)
        else:
            if self.use_ubo:

                content += "layout(shared, binding={}) uniform {}_UBO {{\n".format(self.bind_id, self.name)
                for ipt in inputs:
                    content += " " * 4 + ipt + "\n"
                content += "} " + self.name + ";\n"
            else:
                content += "uniform struct {\n"
                for ipt in inputs:
                    content += " " * 4 + ipt + "\n"
                content += "} " + self.name + ";\n"

        content += "\n"
        return content
