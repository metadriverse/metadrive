"""

Precomputed GI using spherical harmonics.

"""

from __future__ import division, print_function

import os
import sys

os.chdir(os.path.realpath(os.path.dirname(__file__)))
sys.path.insert(0, "../../")

# Shouldn't import everything, but saves a lot of code
from panda3d.core import *
from direct.showbase.ShowBase import ShowBase

from rpcore.globals import Globals
from rpcore.render_target import RenderTarget
from rplibs.progressbar import ETA, ProgressBar, Percentage, Bar, Counter, Rate


class Application(ShowBase):
    def __init__(self):

        # Load settings
        load_prc_file_data(
            "", """
            win-size 10 10
            window-type offscreen
            win-title GI Bake
            textures-power-2 none
            gl-coordinate-system default
            sync-video #f
            support-stencil #f
            framebuffer-stencil #f
            framebuffer-multisample #f
            multisamples 0
            gl-cube-map-seamless #t
            gl-force-fbo-color #f
        """
        )

        ShowBase.__init__(self)
        Globals.load(self)
        Globals.resolution = LVecBase2i(1600, 900)
        sun_vector = Vec3(0.2, 0.5, 1.2).normalized()
        capture_resolution = 32
        sun_shadow_map_resolution = 2048
        num_probes = LVecBase3i(64)
        divisor = 128
        num_bakers = 8
        padding = 0.5

        for region in self.win.get_display_regions():
            region.set_active(False)

        print("Loading scene ...")
        # model = loader.load_model("resources/test-scene.bam")
        model = loader.load_model("scene/scene.bam")
        # model = loader.load_model("scene/LivingRoom.egg")
        model.reparent_to(render)
        model.flatten_strong()

        start_point, end_point = model.get_tight_bounds()
        start_point -= padding
        end_point += padding
        diameter = (start_point - end_point).length() * 0.5
        model_center = (start_point + end_point) * 0.5
        model_size = end_point - start_point

        print("Rendering sun shadow map ..")
        sun_shadow_cam = Camera("SunShadowCamera")
        sun_shadow_lens = OrthographicLens()
        sun_shadow_lens.set_film_size(diameter * 2, diameter * 2)
        sun_shadow_lens.set_near_far(0, 2 * diameter)
        sun_shadow_cam.set_lens(sun_shadow_lens)
        sun_shadow_cam_np = render.attach_new_node(sun_shadow_cam)
        sun_shadow_cam_np.set_pos(model_center + sun_vector * diameter)
        sun_shadow_cam_np.look_at(model_center)

        sun_shadow_target = RenderTarget()
        sun_shadow_target.size = sun_shadow_map_resolution
        sun_shadow_target.add_depth_attachment(bits=32)
        sun_shadow_target.prepare_render(sun_shadow_cam_np)

        self.render_frame()
        sun_shadow_target.active = False
        shadow_mvp = self.get_mvp(sun_shadow_cam_np)

        print("Computing first bounce ..")

        # Target to store all results
        max_probes = num_probes.x * num_probes.y * num_probes.z

        if max_probes % divisor != 0:
            print("WARNING: Bad divisor:", divisor, "for", max_probes)

        num_rows = (max_probes + divisor - 1) // divisor
        final_data = Texture("FinalProbeResult")
        final_data.setup_2d_texture(6 * divisor, num_rows, Texture.T_float, Texture.F_rgba16)
        final_data.set_clear_color(Vec4(1.0, 0.6, 0.2, 1.0))

        worker_handles = []

        store_shader = Shader.load(Shader.SL_GLSL, "resources/default.vert.glsl", "resources/copy_cubemap.frag.glsl")
        convolute_shader = Shader.load(Shader.SL_GLSL, "resources/default.vert.glsl", "resources/convolute.frag.glsl")

        for worked_id in range(num_bakers):
            probe_position = Vec3(0, 0, 4)
            capture_target = RenderTarget()
            capture_target.size = capture_resolution * 6, capture_resolution
            capture_target.add_depth_attachment(bits=16)
            capture_target.add_color_attachment(bits=16, alpha=True)
            capture_target.prepare_render(None)

            # Remove all unused display regions
            internal_buffer = capture_target.internal_buffer
            internal_buffer.remove_all_display_regions()
            internal_buffer.disable_clears()
            internal_buffer.get_overlay_display_region().disable_clears()

            # Setup the cubemap capture rig
            directions = (Vec3(1, 0, 0), Vec3(-1, 0, 0), Vec3(0, 1, 0), Vec3(0, -1, 0), Vec3(0, 0, 1), Vec3(0, 0, -1))
            capture_regions = []
            capture_cams = []
            capture_rig = render.attach_new_node("CaptureRig")
            capture_rig.set_pos(probe_position)

            # Prepare the display regions
            for i in range(6):
                region = capture_target.internal_buffer.make_display_region(i / 6, i / 6 + 1 / 6, 0, 1)
                region.set_sort(25 + i)
                region.set_active(True)
                region.disable_clears()

                # Set the correct clears
                region.set_clear_depth_active(True)
                region.set_clear_depth(1.0)
                region.set_clear_color_active(True)
                region.set_clear_color(Vec4(0.0, 0.0, 0.0, 0.0))

                lens = PerspectiveLens()
                lens.set_fov(90)
                lens.set_near_far(0.05, 2 * diameter)
                camera = Camera("CaptureCam-" + str(i), lens)
                camera_np = capture_rig.attach_new_node(camera)
                camera_np.look_at(camera_np, directions[i])
                region.set_camera(camera_np)
                capture_regions.append(region)
                capture_cams.append(camera_np)

            capture_cams[0].set_r(90)
            capture_cams[1].set_r(-90)
            capture_cams[3].set_r(180)
            capture_cams[5].set_r(180)

            destination_cubemap = Texture("TemporaryCubemap")
            destination_cubemap.setup_cube_map(capture_resolution, Texture.T_float, Texture.F_rgba16)

            # Target to convert the FBO to a cubemap
            target_store_cubemap = RenderTarget()
            target_store_cubemap.size = capture_resolution * 6, capture_resolution
            target_store_cubemap.prepare_buffer()
            target_store_cubemap.set_shader_inputs(SourceTex=capture_target.color_tex, DestTex=destination_cubemap)

            target_store_cubemap.shader = store_shader

            # Target to filter the data
            store_pta = PTALVecBase2i.empty_array(1)
            target_convolute = RenderTarget()
            target_convolute.size = 6, 1
            # target_convolute.add_color_attachment(bits=16)
            target_convolute.prepare_buffer()
            target_convolute.set_shader_inputs(SourceTex=destination_cubemap, DestTex=final_data, storeCoord=store_pta)
            target_convolute.shader = convolute_shader

            # Set initial shader
            shader = Shader.load(Shader.SL_GLSL, "resources/first-bounce.vert.glsl", "resources/first-bounce.frag.glsl")
            render.set_shader(shader)
            render.set_shader_inputs(ShadowMap=sun_shadow_target.depth_tex, shadowMVP=shadow_mvp, sunVector=sun_vector)

            worker_handles.append((capture_rig, store_pta))

        print("Preparing to render", max_probes, "probes ..")
        widgets = [Counter(), "  ", Bar(), "  ", Percentage(), "  ", ETA(), " ", Rate()]
        progressbar = ProgressBar(widgets=widgets, maxval=max_probes).start()
        progressbar.update(0)

        work_queue = []

        for z_pos in range(num_probes.z):
            for y_pos in range(num_probes.y):
                for x_pos in range(num_probes.x):
                    index = x_pos + y_pos * num_probes.x + z_pos * num_probes.y * num_probes.x
                    #     print("Baking", index, "out of", max_probes)
                    offs_x = start_point.x + x_pos / (num_probes.x + 0.5) * model_size.x
                    offs_y = start_point.y + y_pos / (num_probes.y + 0.5) * model_size.y
                    offs_z = start_point.z + z_pos / (num_probes.z + 0.5) * model_size.z

                    store_x = index % divisor
                    store_y = index // divisor

                    work_queue.append((Vec3(offs_x, offs_y, offs_z), LVecBase2i(store_x, store_y)))

        for i, (pos, store) in enumerate(work_queue):
            worker_handles[i % num_bakers][0].set_pos(pos)
            worker_handles[i % num_bakers][1][0] = store
            if i % num_bakers == num_bakers - 1:
                self.render_frame()
                progressbar.update(i)

        progressbar.finish()
        self.render_frame()

        print("Writing out data ..")
        self.graphicsEngine.extract_texture_data(final_data, self.win.gsg)
        final_data.write("raw-bake.png")

        print("Writing out configuration")
        with open("_bake_params.py", "w") as handle:
            handle.write("#Autogenerated\n")
            handle.write("from panda3d.core import LPoint3f, LVecBase3i, LVector3f\n")
            handle.write("BAKE_MESH_START = " + str(start_point) + "\n")
            handle.write("BAKE_MESH_END = " + str(end_point) + "\n")
            handle.write("BAKE_MESH_PROBECOUNT = " + str(num_probes) + "\n")
            handle.write("BAKE_DIVISOR = " + str(divisor) + "\n")
            handle.write("BAKE_SUN_VECTOR = " + str(sun_vector) + "\n")

        with open("_bake_params.glsl", "w") as handle:
            handle.write("// Autogenerated\n")
            handle.write("#define LPoint3f vec3\n")
            handle.write("#define LVector3f vec3\n")
            handle.write("#define LVecBase3i ivec3\n")
            handle.write("const vec3 bake_mesh_start = " + str(start_point) + ";\n")
            handle.write("const vec3 bake_mesh_end = " + str(end_point) + ";\n")
            handle.write("const ivec3 bake_mesh_probecount = " + str(num_probes) + ";\n")
            handle.write("const int bake_divisor = " + str(divisor) + ";\n")
            handle.write("const vec3 sun_vector = " + str(sun_vector) + ";\n")

    def render_frame(self):
        """ Convenience function to render a frame """
        self.graphicsEngine.render_frame()

    def get_mvp(self, cam_node):
        """ Computes the view-projection matrix of a camera """
        return render.get_transform(cam_node).get_mat() * cam_node.node().get_lens().get_projection_mat()


Application()
