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

from metadrive.render_pipeline.rplibs.six import itervalues, iteritems

from metadrive.render_pipeline.rpcore.rpobject import RPObject
from metadrive.render_pipeline.rpcore.render_target import RenderTarget
from metadrive.render_pipeline.rpcore.loader import RPLoader


class RenderStage(RPObject):
    """ This class is the abstract class for all stages used in the pipeline.
    It represents a part of the pipeline render process. Each stage specifies
    which pipes it uses and which pipes it produces. A pipe can be seen as a
    texture, which gets modified. E.g. the gbuffer pass produces the gbuffer
    pipe, the ambient occlusion pass produces the occlusion pipe and so on. The
    lighting pass can then specify which pipes it needs and compute the image.
    Using a pipe system ensures that new techniques can be inserted easily,
    without the other techniques even being aware of them """

    required_inputs = []
    required_pipes = []

    produced_inputs = {}
    produced_pipes = {}
    produced_defines = {}

    disabled = False

    def __init__(self, pipeline):
        """ Creates a new render stage """
        RPObject.__init__(self)
        self.stage_id = self.__class__.__name__
        self._pipeline = pipeline
        self._active = True
        self._targets = {}

    def create(self):
        """ This method should setup the stage and create the pipes """
        raise NotImplementedError()

    def reload_shaders(self):
        """ This method should set all required shaders, there should be no
        shaders set in the create method, because the shader auto config is not
        generated there """
        pass

    def set_shader_input(self, *args):
        """ This method sets a shader input on all stages, which is mainly used
        by the stage manager """
        for target in itervalues(self._targets):
            target.set_shader_input(*args)

    def set_shader_inputs(self, **kwargs):
        """ This method sets shader inputs on all stages, which is mainly used
        by the stage manager """
        for target in itervalues(self._targets):
            target.set_shader_inputs(**kwargs)

    def update(self):
        """ This method gets called every frame, and can be overridden by render
        stages to perform custom updates """
        pass

    @property
    def active(self):
        """ Returns whether *all* targets of the stage are active """
        return self._active

    @active.setter
    def active(self, state):
        """ Enables or disables this stage. In case the stage is disabled, it will
        not get updated anymore, and all stages are distabled """
        if self._active != state:
            self._active = state
            for target in itervalues(self._targets):
                target.active = self._active

    def create_target(self, name):
        """ Creates a new render target and binds it to this stage """
        # Format the name like Plugin:Stage:Name, so it can be easily
        # found in pstats below the plugin cagetory
        name = self._get_plugin_id() + ":" + self.stage_id + ":" + name
        if name in self._targets:
            return self.error("Overriding existing target: " + name)
        self._targets[name] = RenderTarget(name)
        return self._targets[name]

    def remove_target(self, target):
        """ Removes a previously registered target. This unregisters the
        target, as well as removing it from the list of assigned targets. """
        target.remove()
        target_key = None
        for key, value_target in iteritems(self._targets):
            if target == value_target:
                target_key = key
                break
        del self._targets[target_key]

    def _get_shader_handle(self, path, *args):
        """ Returns a handle to a Shader object, containing all sources passed
        as arguments. The path argument will be used to locate shaders if no
        absolute path is given. This is the internal method used in load_shader
        and load_plugin_shader. """
        assert len(args) > 0 and len(args) <= 3
        path_args = []

        for source in args:
            for prefix in ("/$$rpconfig", "/$$rp/shader", "/$$rptemp"):
                if prefix in source:
                    path_args.append(source)
                    break
            else:
                path_args.append(path.format(source))

        # If only one shader is specified, assume its a postprocess fragment shader,
        # and use the default vertex shader
        if len(args) == 1:
            path_args = ["/$$rp/shader/default_post_process.vert.glsl"] + path_args
        return RPLoader.load_shader(*path_args)

    def _get_plugin_id(self):
        """ Returns the id of the plugin which created this stage. This is done
        by extracting the name of the plugin from the module name """
        if "rpcore.stages" in self.__class__.__module__:
            return "render_pipeline_internal"
        return str(self.__class__.__module__).split(".")[-2]

    def load_shader(self, *args):
        """ Loads a shader from the given args. If only one argument is passed,
        the default template for the stage is loaded. If two arguments are
        passed, the first argument should be the vertex shader and the second
        argument should be the fragment shader. If three arguments are passed,
        the order should be vertex, fragment, geometry """
        return self._get_shader_handle("/$$rp/shader/{0}", *args)

    def load_plugin_shader(self, *args):
        """ Loads a shader from the plugin directory. This method is useful
        for RenderStages created by plugins. For a description of the arguments,
        see the load_shader function. """
        shader_path = "rpplugins/" + self._get_plugin_id() + "/shader/{0}"
        return self._get_shader_handle(shader_path, *args)

    def handle_window_resize(self):
        """ This method gets called when the window gets resized. By default,
        this just resizes all render targets. """
        self.set_dimensions()
        for target in itervalues(self._targets):
            target.consider_resize()

    def set_dimensions(self):
        """ This method should set the dimensions on all targets which don't
        have a relative constraint, and also the size of all images. This
        is called after initialization, and when the window resized. """
        pass
