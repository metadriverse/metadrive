"""

Render service to render previews of materials

"""

from __future__ import print_function

import sys
import socket
import time
import pickle

from threading import Thread

from panda3d.core import load_prc_file_data, Filename, Mat4
from panda3d.core import CS_zup_right, CS_yup_right, BamCache
from direct.showbase.ShowBase import ShowBase

sys.path.insert(0, "../../")
from rpcore import RenderPipeline, PointLight  # noqa


class Application(ShowBase):

    ICOMING_PORT = 62360

    def __init__(self):
        load_prc_file_data("", "win-size 512 512")
        load_prc_file_data("", "window-type offscreen")
        load_prc_file_data("", "model-cache-dir")
        load_prc_file_data("", "model-cache-textures #f")
        load_prc_file_data("", "textures-power-2 none")
        load_prc_file_data("", "alpha-bits 0")
        load_prc_file_data("", "print-pipe-types #f")

        # Construct render pipeline
        self.render_pipeline = RenderPipeline()
        self.render_pipeline.mount_mgr.config_dir = "config/"
        self.render_pipeline.create(self)

        self.setup_scene()

        # Disable model caching
        BamCache.get_global_ptr().cache_models = False

        self.update_queue = []
        self.start_listen()

        # Render initial frames
        for i in range(10):
            self.taskMgr.step()

        last_update = 0.0
        self.scene_node = None

        # Wait for updates
        while True:

            # Update once in a while
            curr_time = time.time()
            if curr_time > last_update + 1.0:
                last_update = curr_time
                self.taskMgr.step()

            if self.update_queue:
                if self.scene_node:
                    self.scene_node.remove_node()

                # Only take the latest packet
                payload = self.update_queue.pop(0)
                print("RENDERING:", payload)

                scene = self.loader.loadModel(Filename.from_os_specific(payload["scene"]))

                for light in scene.find_all_matches("**/+PointLight"):
                    light.remove_node()
                for light in scene.find_all_matches("**/+Spotlight"):
                    light.remove_node()

                # Find camera
                main_cam = scene.find("**/Camera")
                if main_cam:
                    transform_mat = main_cam.get_transform(self.render).get_mat()
                    transform_mat = Mat4.convert_mat(CS_zup_right, CS_yup_right) * transform_mat
                    self.camera.set_mat(transform_mat)
                else:
                    print("WARNING: No camera found")
                    self.camera.set_pos(0, -3.5, 0)
                    self.camera.look_at(0, -2.5, 0)

                self.camLens.set_fov(64.0)

                self.scene_node = scene
                scene.reparent_to(self.render)

                # Render scene
                for i in range(8):
                    self.taskMgr.step()

                dest_path = Filename.from_os_specific(payload["dest"])
                print("Saving screenshot to", dest_path)
                self.win.save_screenshot(dest_path)
                self.notify_about_finish(int(payload["pingback_port"]))

    def start_listen(self):
        """ Starts the listener thread """
        thread = Thread(target=self.listener_thread, args=(), name="ListenerThread")
        thread.setDaemon(True)
        thread.start()
        return thread

    def listener_thread(self):
        """ Thread which listens to incoming updates """
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        print("Listening on 127.0.0.1:" + str(self.ICOMING_PORT))
        try:
            sock.bind(("127.0.0.1", self.ICOMING_PORT))
            while True:
                data, addr = sock.recvfrom(8192)
                self.handle_data(data)
        except Exception as msg:
            print("Failed to bind to address! Reason:", msg)
        finally:
            sock.close()

    def handle_data(self, data):
        """ Handles a new update """
        # print("Got:", data)
        unpacked_data = pickle.loads(data)
        # print("Data = ", unpacked_data)
        self.update_queue.append(unpacked_data)

    def notify_about_finish(self, port):
        """ Notifies the caller that the result finished """
        print("Sending finish result to localhost:" + str(port))
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        try:
            sock.connect(("localhost", port))
        except Exception as msg:
            print("Could not send finish result: ", msg)
            return
        sock.sendall(b"done")
        print("Sent done flag.")
        sock.close()

    def setup_scene(self):
        """ Setups the basic scene geometry """
        self.disableMouse()
        self.render2d.hide()
        self.aspect2d.hide()

        light = PointLight()
        light.pos = 20.0, -0.85, -1.31
        light.radius = 100.0
        light.energy = 2500
        light.set_color_from_temperature(8000)
        # self.render_pipeline.add_light(light)

        light = PointLight()
        light.pos = -11.2, -13.84, -9.24
        light.radius = 1e20
        light.set_color_from_temperature(8000)
        light.energy = 2500
        # self.render_pipeline.add_light(light)

        # envprobe = self.render_pipeline.add_environment_probe()
        # envprobe.set_pos(0, -16.2, 4.4)
        # envprobe.set_scale(40, 40, 40)
        # envprobe.parallax_correction = False

Application()
