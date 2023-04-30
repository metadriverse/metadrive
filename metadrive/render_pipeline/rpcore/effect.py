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

from metadrive.render_pipeline.rplibs.six import iteritems, iterkeys
from metadrive.render_pipeline.rplibs.yaml import load_yaml_file

from panda3d.core import Filename
from direct.stdpy.file import open

from metadrive.render_pipeline.rpcore.rpobject import RPObject
from metadrive.render_pipeline.rpcore.loader import RPLoader


class Effect(RPObject):
    """ This class represents an instance of a compiled effect. It can be loaded
    from a file. """

    # Configuration options which can be set per effect instance. These control
    # which features are available in the effect, and which passes to render.
    _DEFAULT_OPTIONS = {
        "render_gbuffer": True,
        "render_shadow": True,
        "render_voxelize": True,
        "render_envmap": True,
        "render_forward": False,
        "alpha_testing": True,
        "normal_mapping": True,
        "parallax_mapping": False,
    }

    # All supported render passes, should match the available passes in the
    # TagStateManager class.
    _PASSES = ("gbuffer", "shadow", "voxelize", "envmap", "forward")

    # Effects are cached based on their source filename and options, this is
    # the cache where compiled are effects stored.
    _GLOBAL_CACHE = {}

    # Global counter to store the amount of generated effects, used to create
    # a unique id used for writing temporary files.
    _EFFECT_ID = 0

    @classmethod
    def load(cls, filename, options):
        """ Loads an effect from a given filename with the specified options.
        This lookups in the global effect cache, and checks if a similar effect
        (i.e. with the same hash) was already loaded, and in that case returns it.
        Otherwise a new effect with the given options is created. """
        effect_hash = cls._generate_hash(filename, options)
        if effect_hash in cls._GLOBAL_CACHE:
            return cls._GLOBAL_CACHE[effect_hash]
        effect = cls()
        effect.set_options(options)
        if not effect.do_load(filename):
            RPObject.global_error("Effect", "Could not load effect!")
            return None
        return effect

    @classmethod
    def _generate_hash(cls, filename, options):
        """ Generates an unique hash for the effect. The effect hash is based
        on the filename and the configured options, and is ensured to make the
        effect unique. This is important to make sure the caching works as
        intended. All options not present in options are set to the default value"""

        # Set all options which are not present in the dict to its defaults
        options = {k: options.get(k, v) for k, v in iteritems(cls._DEFAULT_OPTIONS)}

        # Hash filename, make sure it has the right format and also resolve
        # it to an absolute path, to make sure that relative paths are cached
        # correctly (otherwise, specifying a different path to the same file
        # will cause a cache miss)
        filename = Filename(filename)
        filename.make_absolute()
        file_hash = str(hash(filename.to_os_generic()))

        # Hash the options, that is, sort the keys to make sure the values
        # are always in the same order, and then convert the flags to strings using
        # '1' for a set flag, and '0' for a unset flag
        options_hash = "".join(["1" if options[key] else "0" for key in sorted(iterkeys(options))])
        return file_hash + "-" + options_hash

    def __init__(self):
        """ Constructs a new empty effect, this is a private constructor and
        should not be called. Instead, use Effect.load() """
        RPObject.__init__(self)
        self.effect_id = Effect._EFFECT_ID
        Effect._EFFECT_ID += 1
        self.filename = None
        self._options = self._DEFAULT_OPTIONS.copy()
        self._generated_shader_paths = {}
        self._shader_objs = {}

    def get_option(self, name):
        """ Returns a given option value by name """
        return self._options[name]

    def set_options(self, options):
        """ Sets the effect options, overriding the default options """
        for key, val in iteritems(options):
            if key not in self._options:
                self.error("Unkown option:", key)
                continue
            self._options[key] = val

    def do_load(self, filename):
        """ Internal method to load the effect from the given filename, do
        not use this directly, instead use load(). """
        self.filename = filename
        self.effect_name = self._convert_filename_to_name(filename)
        self.effect_hash = self._generate_hash(filename, self._options)

        # Load the YAML file
        parsed_yaml = load_yaml_file(filename) or {}
        self._parse_content(parsed_yaml)

        # Construct a shader object for each pass
        for pass_id in self._PASSES:
            vertex_src = self._generated_shader_paths["vertex-" + pass_id]
            fragment_src = self._generated_shader_paths["fragment-" + pass_id]
            self._shader_objs[pass_id] = RPLoader.load_shader(vertex_src, fragment_src)
        return True

    def get_shader_obj(self, pass_id):
        """ Returns a handle to the compiled shader object for a given render
        pass. """
        if pass_id not in self._shader_objs:
            self.warn("Pass '" + pass_id + "' not found!")
            return False
        return self._shader_objs[pass_id]

    def _convert_filename_to_name(self, filename):
        """ Constructs an effect name from a filename, this is used for writing
        out temporary files """
        return filename.replace(".yaml", "").replace("effects/", "")\
            .replace("/", "_").replace("\\", "_").replace(".", "-")

    def _parse_content(self, parsed_yaml):
        """ Internal method to construct the effect from a yaml object """
        vtx_data = parsed_yaml.get("vertex", None) or {}
        frag_data = parsed_yaml.get("fragment", None) or {}

        for pass_id in self._PASSES:
            self._parse_shader_template(pass_id, "vertex", vtx_data)
            self._parse_shader_template(pass_id, "fragment", frag_data)

    def _parse_shader_template(self, pass_id, stage, data):
        """ Parses a fragment template. This just finds the default template
        for the shader, and redirects that to construct_shader_from_data """
        if stage == "fragment":
            shader_ext = {"vertex": "vert", "fragment": "frag"}[stage]
            template_src = "/$$rp/shader/templates/{}.{}.glsl".format(pass_id, shader_ext)
        elif stage == "vertex":
            # Using a shared vertex shader
            template_src = "/$$rp/shader/templates/vertex.vert.glsl"

        shader_path = self._construct_shader_from_data(pass_id, stage, template_src, data)
        self._generated_shader_paths[stage + "-" + pass_id] = shader_path

    def _construct_shader_from_data(self, pass_id, stage, template_src, data):  # noqa # pylint: disable=too-many-branches
        """ Constructs a shader from a given dataset """
        injects = {"defines": []}

        for key, val in iteritems(self._options):
            if isinstance(val, bool):
                val_str = "1" if val else "0"
            else:
                val_str = str(val)
            injects["defines"].append("#define OPT_{} {}".format(key.upper(), val_str))

        injects["defines"].append("#define IN_" + stage.upper() + "_SHADER 1")
        injects["defines"].append("#define IN_" + pass_id.upper() + "_SHADER 1")
        injects["defines"].append("#define IN_RENDERING_PASS 1")

        # Parse dependencies
        if "dependencies" in data:
            injects["includes"] = []
            for dependency in data["dependencies"]:
                include_str = "#pragma include \"{}\"".format(dependency)
                injects["includes"].append(include_str)
            del data["dependencies"]

        # Append aditional injects
        for key, val in iteritems(data):
            if val is None:
                self.warn("Empty insertion: '" + key + "'")
                continue

            if isinstance(val, (list, tuple)):
                self.warn("Invalid syntax, you used a list but you should have used a string:")
                self.warn(val)
                continue
            injects[key] = injects.get(key, []) + [i for i in val.split("\n")]

        cache_key = self.effect_name + "@" + stage + "-" + pass_id + "@" + self.effect_hash
        return self._process_shader_template(template_src, cache_key, injects)

    def _process_shader_template(self, template_src, cache_key, injections):  # noqa # pylint: disable=too-many-branches
        """ Generates a compiled shader object from a given shader
        source location and code injection definitions. """
        with open(template_src, "r") as handle:
            shader_lines = handle.readlines()

        parsed_lines = ["\n\n"]
        addline = parsed_lines.append

        addline("/* Compiled Shader Template")
        addline(" * generated from: '" + template_src + "'")
        addline(" * cache key: '" + cache_key + "'")
        addline(" *")
        addline(" * !!! Autogenerated, do not edit! Your changes will be lost. !!!")
        addline(" */\n\n")

        # Store whether we are in the main function already - we need this
        # to properly insert scoped code blocks
        in_main = False

        for line in shader_lines:  # pylint: disable=too-many-nested-blocks
            stripped_line = line.strip().lower()

            # Check if we are already in the main function
            if "void main()" in stripped_line:
                in_main = True

            # Check if the current line is a hook
            if stripped_line.startswith("%") and stripped_line.endswith("%"):

                # If the line is a hook, get the hook name and save the
                # indent so we can indent all injected lines properly.
                hook_name = stripped_line[1:-1]
                indent = " " * (len(line) - len(line.lstrip()))

                # Inject all registered template values into the hook
                if hook_name in injections:

                    # Directly remove the value from the list so we can check which
                    # hooks were not found in the template
                    insertions = injections.pop(hook_name)

                    if len(insertions) > 0:

                        # When we are in the main function, we have to make sure we
                        # use a seperate scope, so there are no conflicts with variable
                        # declarations
                        header = indent + "/* Hook: " + hook_name + " */" + (" {" if in_main else "")  # noqa # pylint: disable=line-too-long
                        addline(header)

                        for line_to_insert in insertions:
                            if line_to_insert is None:
                                self.warn("Empty insertion '" + hook_name + "'")
                                continue

                            if not isinstance(line_to_insert, str):
                                self.warn("Invalid line type: ", line_to_insert)
                                continue

                            # Dont indent defines and pragmas
                            if line_to_insert.startswith("#"):
                                addline(line_to_insert)
                            else:
                                addline(indent + line_to_insert)

                        if in_main:
                            addline(indent + "}")

            else:
                addline(line.rstrip())

        # Add a closing newline to the file
        addline("")

        # Warn the user about all unused hooks
        for key in injections:
            self.warn("Hook '" + key + "' not found in template '" + template_src + "'!")

        # Write the constructed shader and load it back
        shader_content = "\n".join(parsed_lines)
        temp_path = "/$$rptemp/$$effect-" + cache_key + ".glsl"

        with open(temp_path, "w") as handle:
            handle.write(shader_content)

        return temp_path
