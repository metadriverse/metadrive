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

from __future__ import division

import sys
import math
import time

from panda3d.core import LVecBase2i, TransformState, RenderState, load_prc_file
from panda3d.core import PandaSystem, MaterialAttrib, WindowProperties
from panda3d.core import GeomTristrips, Vec4

from direct.showbase.ShowBase import ShowBase
from direct.stdpy.file import isfile

from metadrive.render_pipeline.rplibs.yaml import load_yaml_file_flat
from metadrive.render_pipeline.rplibs.six.moves import range  # pylint: disable=import-error

from metadrive.render_pipeline.rpcore.globals import Globals
from metadrive.render_pipeline.rpcore.effect import Effect
from metadrive.render_pipeline.rpcore.rpobject import RPObject
from metadrive.render_pipeline.rpcore.common_resources import CommonResources
from metadrive.render_pipeline.rpcore.native import TagStateManager, PointLight, SpotLight
from metadrive.render_pipeline.rpcore.render_target import RenderTarget
from metadrive.render_pipeline.rpcore.pluginbase.manager import PluginManager
from metadrive.render_pipeline.rpcore.pluginbase.day_manager import DayTimeManager

from metadrive.render_pipeline.rpcore.util.task_scheduler import TaskScheduler
from metadrive.render_pipeline.rpcore.util.network_communication import NetworkCommunication
from metadrive.render_pipeline.rpcore.util.ies_profile_loader import IESProfileLoader

from metadrive.render_pipeline.rpcore.gui.debugger import Debugger
from metadrive.render_pipeline.rpcore.gui.loading_screen import LoadingScreen

from metadrive.render_pipeline.rpcore.mount_manager import MountManager
from metadrive.render_pipeline.rpcore.stage_manager import StageManager
from metadrive.render_pipeline.rpcore.light_manager import LightManager

from metadrive.render_pipeline.rpcore.stages.ambient_stage import AmbientStage
from metadrive.render_pipeline.rpcore.stages.gbuffer_stage import GBufferStage
from metadrive.render_pipeline.rpcore.stages.final_stage import FinalStage
from metadrive.render_pipeline.rpcore.stages.downscale_z_stage import DownscaleZStage
from metadrive.render_pipeline.rpcore.stages.combine_velocity_stage import CombineVelocityStage
from metadrive.render_pipeline.rpcore.stages.upscale_stage import UpscaleStage


class RenderPipeline(RPObject):
    """ This is the main render pipeline class, it combines all components of
    the pipeline to form a working system. It does not do much work itself, but
    instead setups all the managers and systems to be able to do their work. """
    def __init__(self):
        """ Creates a new pipeline with a given showbase instance. This should
        be done before intializing the ShowBase, the pipeline will take care of
        that. If the showbase has been initialized before, have a look at
        the alternative initialization of the render pipeline (the first sample)."""
        RPObject.__init__(self)
        self._analyze_system()
        self.mount_mgr = MountManager(self)
        self.settings = {}
        self._applied_effects = []
        self._pre_showbase_initialized = False
        self._first_frame = None
        self.set_loading_screen_image("/$$rp/data/gui/loading_screen_bg.txo")

    def load_settings(self, path):
        """ Loads the pipeline configuration from a given filename. Usually
        this is the 'config/pipeline.ini' file. If you call this more than once,
        only the settings of the last file will be used. """
        self.settings = load_yaml_file_flat(path)

    def reload_shaders(self):
        """ Reloads all shaders. This will reload the shaders of all plugins,
        as well as the pipelines internally used shaders. Because of the
        complexity of some shaders, this operation take might take several
        seconds. Also notice that all applied effects will be lost, and instead
        the default effect will be set on all elements again. Due to this fact,
        this method is primarly useful for fast iterations when developing new
        shaders. """
        if self.settings["pipeline.display_debugger"]:
            self.debug("Reloading shaders ..")
            self.debugger.error_msg_handler.clear_messages()
            self.debugger.set_reload_hint_visible(True)
            self._showbase.graphicsEngine.render_frame()
            self._showbase.graphicsEngine.render_frame()
        self.tag_mgr.cleanup_states()
        self.stage_mgr.reload_shaders()
        self.light_mgr.reload_shaders()
        self._set_default_effect()
        self.plugin_mgr.trigger_hook("shader_reload")
        if self.settings["pipeline.display_debugger"]:
            self.debugger.set_reload_hint_visible(False)
        self._apply_custom_shaders()

    def _apply_custom_shaders(self):
        """ Re-applies all custom shaders the user applied, to avoid them getting
        removed when the shaders are reloaded """
        self.debug("Re-applying", len(self._applied_effects), "custom shaders")
        for args in self._applied_effects:
            self._internal_set_effect(*args)

    def pre_showbase_init(self):
        """ Setups all required pipeline settings and configuration which have
        to be set before the showbase is setup. This is called by create(),
        in case the showbase was not initialized, however you can (and have to)
        call it manually before you init your custom showbase instance.
        See the 00-Loading the pipeline sample for more information. """
        if not self.mount_mgr.is_mounted:
            self.debug("Mount manager was not mounted, mounting now ...")
            self.mount_mgr.mount()

        if not self.settings:
            self.debug("No settings loaded, loading from default location")
            self.load_settings("/$$rpconfig/pipeline.yaml")

        if not isfile("/$$rp/data/install.flag"):
            self.fatal("You didn't setup the pipeline yet! Please run setup.py.")

        load_prc_file("/$$rpconfig/panda3d-config.prc")
        self._pre_showbase_initialized = True

    def create(self, base=None):
        """ This creates the pipeline, and setups all buffers. It also
        constructs the showbase. The settings should have been loaded before
        calling this, and also the base and write path should have been
        initialized properly (see MountManager).

        If base is None, the showbase used in the RenderPipeline constructor
        will be used and initialized. Otherwise it is assumed that base is an
        initialized ShowBase object. In this case, you should call
        pre_showbase_init() before initializing the ShowBase"""

        start_time = time.time()
        self._init_showbase(base)

        if not self._showbase.win.gsg.supports_compute_shaders:
            self.fatal(
                "Sorry, your GPU does not support compute shaders! Make sure\n"
                "you have the latest drivers. If you already have, your gpu might\n"
                "be too old, or you might be using the open source drivers on linux."
            )

        self._init_globals()
        self.loading_screen.create()
        self._adjust_camera_settings()
        self._create_managers()
        self.plugin_mgr.load()
        self.daytime_mgr.load_settings()
        self.common_resources.write_config()
        self._init_debugger()

        self.plugin_mgr.trigger_hook("stage_setup")
        self.plugin_mgr.trigger_hook("post_stage_setup")

        self._create_common_defines()
        self._initialize_managers()
        self._create_default_skybox()

        self.plugin_mgr.trigger_hook("pipeline_created")

        self._listener = NetworkCommunication(self)
        self._set_default_effect()

        # Measure how long it took to initialize everything, and also store
        # when we finished, so we can measure how long it took to render the
        # first frame (where the shaders are actually compiled)
        init_duration = (time.time() - start_time)
        self._first_frame = time.process_time()
        self.debug(
            "Finished initialization in {:3.3f} s, first frame: {}".format(
                init_duration, Globals.clock.get_frame_count()
            )
        )

    def set_loading_screen_image(self, image_source):
        """ Tells the pipeline to use the default loading screen, which consists
        of a simple loading image. The image source should be a fullscreen
        16:9 image, and not too small, to avoid being blurred out. """
        self.loading_screen = LoadingScreen(self, image_source)

    def add_light(self, light):
        """ Adds a new light to the rendered lights, check out the LightManager
        add_light documentation for further information. """
        self.light_mgr.add_light(light)

    def remove_light(self, light):
        """ Removes a previously attached light, check out the LightManager
        remove_light documentation for further information. """
        self.light_mgr.remove_light(light)

    def load_ies_profile(self, filename):
        """ Loads an IES profile from a given filename and returns a handle which
        can be used to set an ies profile on a light """
        return self.ies_loader.load(filename)

    def _internal_set_effect(self, nodepath, effect_src, options=None, sort=30):
        """ Sets an effect to the given object, using the specified options.
        Check out the effect documentation for more information about possible
        options and configurations. The object should be a nodepath, and the
        effect will be applied to that nodepath and all nodepaths below whose
        current effect sort is less than the new effect sort (passed by the
        sort parameter). """
        effect = Effect.load(effect_src, options)
        if effect is None:
            return self.error("Could not apply effect")

        for i, stage in enumerate(("gbuffer", "shadow", "voxelize", "envmap", "forward")):
            if not effect.get_option("render_" + stage):
                nodepath.hide(self.tag_mgr.get_mask(stage))
            else:
                shader = effect.get_shader_obj(stage)
                if stage == "gbuffer":
                    nodepath.set_shader(shader, 25)
                else:
                    self.tag_mgr.apply_state(stage, nodepath, shader, str(effect.effect_id), 25 + 10 * i + sort)
                nodepath.show_through(self.tag_mgr.get_mask(stage))

        if effect.get_option("render_gbuffer") and effect.get_option("render_forward"):
            self.error(
                "You cannot render an object forward and deferred at the "
                "same time! Either use render_gbuffer or use render_forward, "
                "but not both."
            )

    def set_effect(self, nodepath, effect_src, options=None, sort=30):
        """ See _internal_set_effect. """
        args = (nodepath, effect_src, options, sort)
        self._applied_effects.append(args)
        self._internal_set_effect(*args)

    def add_environment_probe(self):
        """ Constructs a new environment probe and returns the handle, so that
        the probe can be modified. In case the env_probes plugin is not activated,
        this returns a dummy object which can be modified but has no impact. """
        if not self.plugin_mgr.is_plugin_enabled("env_probes"):
            self.warn("env_probes plugin is not loaded - cannot add environment probe")

            class DummyEnvironmentProbe(object):  # pylint: disable=too-few-public-methods
                def __getattr__(self, *args, **kwargs):
                    return lambda *args, **kwargs: None

            return DummyEnvironmentProbe()

        # Ugh ..
        from metadrive.render_pipeline.rpplugins.env_probes.environment_probe import EnvironmentProbe
        probe = EnvironmentProbe()
        self.plugin_mgr.instances["env_probes"].probe_mgr.add_probe(probe)
        return probe

    def prepare_scene(self, scene):
        """ Prepares a given scene, by converting panda lights to render pipeline
        lights. This also converts all empties with names starting with 'ENVPROBE'
        to environment probes. Conversion of blender to render pipeline lights
        is done by scaling their intensity by 100 to match lumens.

        Additionally, this finds all materials with the 'TRANSPARENT' shading
        model, and sets the proper effects on them to ensure they are rendered
        properly.

        This method also returns a dictionary with handles to all created
        objects, that is lights, environment probes, and transparent objects.
        This can be used to store them and process them later on, or delete
        them when a newer scene is loaded."""
        lights = []
        for light in scene.find_all_matches("**/+PointLight"):
            light_node = light.node()
            rp_light = PointLight()
            rp_light.pos = light.get_pos(Globals.base.render)
            rp_light.radius = light_node.max_distance
            rp_light.energy = 20.0 * light_node.color.w
            rp_light.color = light_node.color.xyz
            rp_light.casts_shadows = light_node.shadow_caster
            rp_light.shadow_map_resolution = light_node.shadow_buffer_size.x
            rp_light.inner_radius = 0.4

            self.add_light(rp_light)
            light.remove_node()
            lights.append(rp_light)

        for light in scene.find_all_matches("**/+Spotlight"):
            light_node = light.node()
            rp_light = SpotLight()
            rp_light.pos = light.get_pos(Globals.base.render)
            rp_light.radius = light_node.max_distance
            rp_light.energy = 20.0 * light_node.color.w
            rp_light.color = light_node.color.xyz
            rp_light.casts_shadows = light_node.shadow_caster
            rp_light.shadow_map_resolution = light_node.shadow_buffer_size.x
            rp_light.fov = light_node.exponent / math.pi * 180.0
            lpoint = light.get_mat(Globals.base.render).xform_vec((0, 0, -1))
            rp_light.direction = lpoint
            self.add_light(rp_light)
            light.remove_node()
            lights.append(rp_light)

        envprobes = []
        for np in scene.find_all_matches("**/ENVPROBE*"):
            probe = self.add_environment_probe()
            probe.set_mat(np.get_mat())
            probe.border_smoothness = 0.0001
            probe.parallax_correction = True
            np.remove_node()
            envprobes.append(probe)

        tristrips_warning_emitted = False
        transparent_objects = []
        for geom_np in scene.find_all_matches("**/+GeomNode"):
            geom_node = geom_np.node()
            geom_count = geom_node.get_num_geoms()
            for i in range(geom_count):
                state = geom_node.get_geom_state(i)
                geom = geom_node.get_geom(i)

                needs_conversion = False
                for prim in geom.get_primitives():
                    if isinstance(prim, GeomTristrips):
                        needs_conversion = True
                        if not tristrips_warning_emitted:
                            self.warn(
                                "At least one GeomNode (", geom_node.get_name(),
                                "and possible more..) contains tristrips."
                            )
                            self.warn("Due to a NVIDIA Driver bug, we have to convert them to triangles now.")
                            self.warn("Consider exporting your models with the Bam Exporter to avoid this.")
                            tristrips_warning_emitted = True
                            break

                if needs_conversion:
                    geom_node.modify_geom(i).decompose_in_place()

                if not state.has_attrib(MaterialAttrib):
                    self.warn("Geom", geom_node, "has no material! Please fix this.")
                    continue

                material = state.get_attrib(MaterialAttrib).get_material()
                shading_model = material.emission.x

                # SHADING_MODEL_TRANSPARENT
                if shading_model == 3:
                    if geom_count > 1:
                        self.error(
                            "Transparent materials must be on their own geom!\n"
                            "If you are exporting from blender, split them into\n"
                            "seperate meshes, then re-export your scene. The\n"
                            "problematic mesh is: " + geom_np.get_name()
                        )
                        continue
                    self.set_effect(
                        geom_np, "effects/default.yaml", {
                            "render_forward": True,
                            "render_gbuffer": False
                        }, 100
                    )

        return {"lights": lights, "envprobes": envprobes, "transparent_objects": transparent_objects}

    def _create_managers(self):
        """ Internal method to create all managers and instances. This also
        initializes the commonly used render stages, which are always required,
        independently of which plugins are enabled. """
        self.task_scheduler = TaskScheduler(self)
        self.tag_mgr = TagStateManager(Globals.base.cam)
        self.plugin_mgr = PluginManager(self)
        self.stage_mgr = StageManager(self)
        self.light_mgr = LightManager(self)
        self.daytime_mgr = DayTimeManager(self)
        self.ies_loader = IESProfileLoader(self)
        self.common_resources = CommonResources(self)
        self._init_common_stages()

    def _analyze_system(self):
        """ Prints information about the system used, including information
        about the used Panda3D build. Also checks if the Panda3D build is out
        of date. """
        self.debug(
            "Using Python {}.{} with architecture {}".format(
                sys.version_info.major, sys.version_info.minor, PandaSystem.get_platform()
            )
        )

        build_date = getattr(PandaSystem, 'build_date', None)
        if build_date:
            self.debug("Using Panda3D {} built on {}".format(PandaSystem.get_version_string(), build_date))
        else:
            self.debug("Using Panda3D {}".format(PandaSystem.get_version_string()))

        if PandaSystem.get_git_commit():
            self.debug("Using git commit {}".format(PandaSystem.get_git_commit()))
        else:
            self.debug("Using custom Panda3D build")
        if not self._check_version():
            self.fatal(
                "Your Panda3D version is outdated! Please update to the newest \n"
                "git version! Checkout https://github.com/panda3d/panda3d to "
                "compile panda from source, or get a recent buildbot build."
            )

    def _initialize_managers(self):
        """ Internal method to initialize all managers, after they have been
        created earlier in _create_managers. The creation and initialization
        is seperated due to the fact that plugins and various other subprocesses
        have to get initialized inbetween. """
        self.stage_mgr.setup()
        self.stage_mgr.reload_shaders()
        self.light_mgr.reload_shaders()
        self._init_bindings()
        self.light_mgr.init_shadows()

    def _init_debugger(self):
        """ Internal method to initialize the GUI-based debugger. In case debugging
        is disabled, this constructs a dummy debugger, which does nothing.
        The debugger itself handles the various onscreen components. """
        if self.settings["pipeline.display_debugger"]:
            self.debugger = Debugger(self)
        else:
            # Use an empty onscreen debugger in case the debugger is not
            # enabled, which defines all member functions as empty lambdas
            class EmptyDebugger(object):  # pylint: disable=too-few-public-methods
                def __getattr__(self, *args, **kwargs):
                    return lambda *args, **kwargs: None

            self.debugger = EmptyDebugger()  # pylint: disable=redefined-variable-type
            del EmptyDebugger

    def _init_globals(self):
        """ Inits all global bindings. This includes references to the global
        ShowBase instance, as well as the render resolution, the GUI font,
        and various global logging and output methods. """
        Globals.load(self._showbase)
        native_w, native_h = self._showbase.win.get_x_size(), self._showbase.win.get_y_size()
        Globals.native_resolution = LVecBase2i(native_w, native_h)
        self._last_window_dims = LVecBase2i(Globals.native_resolution)
        self._compute_render_resolution()
        RenderTarget.RT_OUTPUT_FUNC = lambda *args: RPObject.global_warn("RenderTarget", *args[1:])
        RenderTarget.USE_R11G11B10 = self.settings["pipeline.use_r11_g11_b10"]

    def _set_default_effect(self):
        """ Sets the default effect used for all objects if not overridden, this
        just calls set_effect with the default effect and options as parameters.
        This uses a very low sort, to make sure that overriding the default
        effect does not require a custom sort parameter to be passed. """
        self.set_effect(Globals.render, "effects/default.yaml", {}, -10)

    def _adjust_camera_settings(self):
        """ Sets the default camera settings, this includes the cameras
        near and far plane, as well as FoV. The reason for this is, that pandas
        default field of view is very small, and thus we increase it. """
        self._showbase.camLens.set_near_far(0.1, 70000)
        self._showbase.camLens.set_fov(40)

    def _compute_render_resolution(self):
        """ Computes the internally used render resolution. This might differ
        from the window dimensions in case a resolution scale is set. """
        scale_factor = self.settings["pipeline.resolution_scale"]
        w = int(float(Globals.native_resolution.x) * scale_factor)
        h = int(float(Globals.native_resolution.y) * scale_factor)
        # Make sure the resolution is a multiple of 4
        w, h = w - w % 4, h - h % 4
        self.debug("Render resolution is", w, "x", h)
        Globals.resolution = LVecBase2i(w, h)

    def _init_showbase(self, base):
        """ Inits the the given showbase object. This is part of an alternative
        method of initializing the showbase. In case base is None, a new
        ShowBase instance will be created and initialized. Otherwise base() is
        expected to either be an uninitialized ShowBase instance, or an
        initialized instance with pre_showbase_init() called inbefore. """
        if not base:
            self.pre_showbase_init()
            self._showbase = ShowBase()
        else:
            if not hasattr(base, "render"):
                self.pre_showbase_init()
                ShowBase.__init__(base)
            else:
                if not self._pre_showbase_initialized:
                    self.fatal(
                        "You constructed your own ShowBase object but you\n"
                        "did not call pre_show_base_init() on the render\n"
                        "pipeline object before! Checkout the 00-Loading the\n"
                        "pipeline sample to see how to initialize the RP."
                    )
            self._showbase = base

        # Now that we have a showbase and a window, we can print out driver info
        self.debug("Driver Version =", self._showbase.win.gsg.driver_version)
        self.debug("Driver Vendor =", self._showbase.win.gsg.driver_vendor)
        self.debug("Driver Renderer =", self._showbase.win.gsg.driver_renderer)

    def _init_bindings(self):
        """ Internal method to init the tasks and keybindings. This constructs
        the tasks to be run on a per-frame basis. """
        self._showbase.addTask(self._manager_update_task, "RP_UpdateManagers", sort=10)
        self._showbase.addTask(self._plugin_pre_render_update, "RP_Plugin_BeforeRender", sort=12)
        self._showbase.addTask(self._plugin_post_render_update, "RP_Plugin_AfterRender", sort=15)
        self._showbase.addTask(self._update_inputs_and_stages, "RP_UpdateInputsAndStages", sort=18)
        self._showbase.taskMgr.doMethodLater(0.5, self._clear_state_cache, "RP_ClearStateCache")
        self._showbase.accept("window-event", self._handle_window_event)

    def _handle_window_event(self, event):
        """ Checks for window events. This mainly handles incoming resizes,
        and calls the required handlers """
        self._showbase.windowEvent(event)
        window_dims = LVecBase2i(self._showbase.win.get_x_size(), self._showbase.win.get_y_size())
        if window_dims != self._last_window_dims and window_dims != Globals.native_resolution:
            self._last_window_dims = LVecBase2i(window_dims)

            # Ensure the dimensions are a multiple of 4, and if not, correct it
            if window_dims.x % 4 != 0 or window_dims.y % 4 != 0:
                self.debug("Correcting non-multiple of 4 window size:", window_dims)
                window_dims.x = window_dims.x - window_dims.x % 4
                window_dims.y = window_dims.y - window_dims.y % 4
                props = WindowProperties.size(window_dims.x, window_dims.y)
                self._showbase.win.request_properties(props)

            self.debug("Resizing to", window_dims.x, "x", window_dims.y)
            Globals.native_resolution = window_dims
            self._compute_render_resolution()
            self.light_mgr.compute_tile_size()
            self.stage_mgr.handle_window_resize()
            self.debugger.handle_window_resize()
            self.plugin_mgr.trigger_hook("window_resized")

    def _clear_state_cache(self, task=None):
        """ Task which repeatedly clears the state cache to avoid storing
        unused states. While running once a while, this task prevents over-polluting
        the state-cache with unused states. This complements Panda3D's internal
        state garbarge collector, which does a great job, but still cannot clear
        up all states. """
        task.delayTime = 2.0
        TransformState.clear_cache()
        RenderState.clear_cache()
        return task.again

    def _manager_update_task(self, task):
        """ Update task which gets called before the rendering, and updates
        all managers."""
        self.task_scheduler.step()
        self._listener.update()
        self.debugger.update()
        self.daytime_mgr.update()
        self.light_mgr.update()

        if Globals.clock.get_frame_count() >= 10 and self.loading_screen.fullscreen_node.hasParent():
            self.debug("Hiding loading screen after 10 pre-rendered frames.")
            self.loading_screen.remove()

        return task.cont

    def _update_inputs_and_stages(self, task):
        """ Updates the commonly used inputs each frame. This is a seperate
        task to be able view detailed performance information in pstats, since
        a lot of matrix calculations are involved here. """
        self.common_resources.update()
        self.stage_mgr.update()
        return task.cont

    def _plugin_pre_render_update(self, task):
        """ Update task which gets called before the rendering, and updates the
        plugins. This is a seperate task to split the work, and be able to do
        better performance analysis in pstats later on. """
        self.plugin_mgr.trigger_hook("pre_render_update")
        return task.cont

    def _plugin_post_render_update(self, task):
        """ Update task which gets called after the rendering, and should cleanup
        all unused states and objects. This also triggers the plugin post-render
        update hook. """
        self.plugin_mgr.trigger_hook("post_render_update")
        if self._first_frame is not None:
            duration = time.process_time() - self._first_frame
            self.debug("Took", round(duration, 3), "s until first frame")
            self._first_frame = None
        return task.cont

    def _create_common_defines(self):
        """ Creates commonly used defines for the shader configuration. """
        defines = self.stage_mgr.defines
        defines["CAMERA_NEAR"] = round(Globals.base.camLens.get_near(), 10)
        defines["CAMERA_FAR"] = round(Globals.base.camLens.get_far(), 10)

        # Work arround buggy nvidia driver, which expects arrays to be const
        if "NVIDIA 361.43" in self._showbase.win.gsg.get_driver_version():
            defines["CONST_ARRAY"] = "const"
        else:
            defines["CONST_ARRAY"] = ""

        # Provide driver vendor as a define
        vendor = self._showbase.win.gsg.get_driver_vendor().lower()
        defines["IS_NVIDIA"] = "nvidia" in vendor
        defines["IS_AMD"] = vendor.startswith("ati")
        defines["IS_INTEL"] = "intel" in vendor

        defines["REFERENCE_MODE"] = self.settings["pipeline.reference_mode"]
        self.light_mgr.init_defines()
        self.plugin_mgr.init_defines()

    def _create_default_skybox(self, size=40000):
        """ Returns the default skybox, with a scale of <size>, and all
        proper effects and shaders already applied. The skybox is already
        parented to render as well. """
        skybox = self.common_resources.load_default_skybox()
        skybox.set_scale(size)
        skybox.reparent_to(Globals.render)
        skybox.set_bin("unsorted", 10000)
        self.set_effect(
            skybox, "effects/skybox.yaml", {
                "render_shadow": False,
                "render_envmap": False,
                "render_voxelize": False,
                "alpha_testing": False,
                "normal_mapping": False,
                "parallax_mapping": False
            }, 1000
        )
        return skybox

    def _check_version(self):
        """ Internal method to check if the required Panda3D version is met. Returns
        True if the version is new enough, and False if the version is outdated. """
        from panda3d.core import Texture
        if not hasattr(Texture, "F_r16i"):
            return False
        return True

    def _init_common_stages(self):
        """ Inits the commonly used stages, which don't belong to any plugin,
        but yet are necessary and widely used. """
        add_stage = self.stage_mgr.add_stage
        self._ambient_stage = AmbientStage(self)
        add_stage(self._ambient_stage)
        self._gbuffer_stage = GBufferStage(self)
        add_stage(self._gbuffer_stage)
        self._final_stage = FinalStage(self)
        add_stage(self._final_stage)
        self._downscale_stage = DownscaleZStage(self)
        add_stage(self._downscale_stage)
        self._combine_velocity_stage = CombineVelocityStage(self)
        add_stage(self._combine_velocity_stage)

        # Add an upscale/downscale stage in case we render at a different resolution
        if abs(1 - self.settings["pipeline.resolution_scale"]) > 0.005:
            self._upscale_stage = UpscaleStage(self)
            add_stage(self._upscale_stage)

    def _get_serialized_material_name(self, material, index=0):
        """ Returns a serializable material name """
        return str(index) + "-" + (material.get_name().replace(" ", "").strip() or "unnamed")

    def export_materials(self, pth):
        """ Exports a list of all materials found in the current scene in a
        serialized format to the given path """

        with open(pth, "w") as handle:
            for i, material in enumerate(Globals.render.find_all_materials()):
                if not material.has_base_color() or not material.has_roughness() or not material.has_refractive_index():
                    print("Skipping non-pbr material:", material.name)
                    continue

                handle.write(
                    ("{} " * 11).format(
                        self._get_serialized_material_name(material, i),
                        material.base_color.x,
                        material.base_color.y,
                        material.base_color.z,
                        material.roughness,
                        material.refractive_index,
                        material.metallic,
                        material.emission.x,  # shading model
                        material.emission.y,  # normal strength
                        material.emission.z,  # arbitrary 0
                        material.emission.w,  # arbitrary 1
                    ) + "\n"
                )

    def update_serialized_material(self, data):
        """ Internal method to update a material from a given serialized material """
        name = data[0]

        for i, material in enumerate(Globals.render.find_all_materials()):
            if self._get_serialized_material_name(material, i) == name:
                material.set_base_color(Vec4(float(data[1]), float(data[2]), float(data[3]), 1.0))
                material.set_roughness(float(data[4]))
                material.set_refractive_index(float(data[5]))
                material.set_metallic(float(data[6]))
                material.set_emission(Vec4(
                    float(data[7]),
                    float(data[8]),
                    float(data[9]),
                    float(data[10]),
                ))

        RenderState.clear_cache()
