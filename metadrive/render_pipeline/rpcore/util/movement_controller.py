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


# Disable the "exactly one space required after comma" message, for the input
# bindings it looks nicer to insert some spaces (see the setup method)
# pylint: disable=bad-whitespace

from __future__ import print_function

import sys
import os
import subprocess

from panda3d.core import ModifierButtons, Vec3, PStatClient
from panda3d.core import Point3, CurveFitter


class MovementController(object):

    """ This is a helper class, used to controll the camera and enable various
    debugging features. It is not really part of the pipeline, but included to
    view the demo scenes. """

    def __init__(self, showbase):
        self.showbase = showbase
        self.movement = [0, 0, 0]
        self.velocity = Vec3(0.0)
        self.hpr_movement = [0, 0]
        self.speed = 0.4
        self.initial_position = Vec3(0)
        self.initial_destination = Vec3(0)
        self.initial_hpr = Vec3(0)
        self.mouse_enabled = False
        self.last_mouse_pos = [0, 0]
        self.mouse_sensivity = 0.7
        self.keyboard_hpr_speed = 0.4
        self.use_hpr = False
        self.smoothness = 6.0
        self.bobbing_amount = 1.5
        self.bobbing_speed = 0.5

    def set_initial_position(self, pos, target):
        """ Sets the initial camera position """
        self.initial_position = pos
        self.initial_destination = target
        self.use_hpr = False
        self.reset_to_initial()

    def set_initial_position_hpr(self, pos, hpr):
        """ Sets the initial camera position """
        self.initial_position = pos
        self.initial_hpr = hpr
        self.use_hpr = True
        self.reset_to_initial()

    def reset_to_initial(self):
        """ Resets the camera to the initial position """
        self.showbase.camera.set_pos(self.initial_position)

        if self.use_hpr:
            self.showbase.camera.set_hpr(self.initial_hpr)
        else:
            self.showbase.camera.look_at(
                self.initial_destination.x, self.initial_destination.y,
                self.initial_destination.z)

    def set_movement(self, direction, amount):
        self.movement[direction] = amount

    def set_hpr_movement(self, direction, amount):
        self.hpr_movement[direction] = amount

    def set_mouse_enabled(self, enabled):
        self.mouse_enabled = enabled

    def increase_speed(self):
        self.speed *= 1.4

    def decrease_speed(self):
        self.speed *= 0.6

    def unbind(self):
        """ Unbinds the movement controler and restores the previous state """
        raise NotImplementedError()

    @property
    def clock_obj(self):
        """ Returns the global clock object """
        return self.showbase.taskMgr.globalClock

    def setup(self):
        """ Attaches the movement controller and inits the keybindings """
        # x
        self.showbase.accept("raw-w", self.set_movement, [0, 1])
        self.showbase.accept("raw-w-up", self.set_movement, [0, 0])
        self.showbase.accept("raw-s", self.set_movement, [0, -1])
        self.showbase.accept("raw-s-up", self.set_movement, [0, 0])

        # y
        self.showbase.accept("raw-a", self.set_movement, [1, -1])
        self.showbase.accept("raw-a-up", self.set_movement, [1, 0])
        self.showbase.accept("raw-d", self.set_movement, [1, 1])
        self.showbase.accept("raw-d-up", self.set_movement, [1, 0])

        # z
        self.showbase.accept("space", self.set_movement, [2, 1])
        self.showbase.accept("space-up", self.set_movement, [2, 0])
        self.showbase.accept("shift", self.set_movement, [2, -1])
        self.showbase.accept("shift-up", self.set_movement, [2, 0])

        # wireframe + debug + buffer viewer
        self.showbase.accept("f3", self.showbase.toggle_wireframe)
        self.showbase.accept("f11", lambda: self.showbase.win.save_screenshot("screenshot.png"))
        self.showbase.accept("j", self.print_position)

        # mouse
        self.showbase.accept("mouse1", self.set_mouse_enabled, [True])
        self.showbase.accept("mouse1-up", self.set_mouse_enabled, [False])

        # arrow mouse navigation
        self.showbase.accept("arrow_up", self.set_hpr_movement, [1, 1])
        self.showbase.accept("arrow_up-up", self.set_hpr_movement, [1, 0])
        self.showbase.accept("arrow_down", self.set_hpr_movement, [1, -1])
        self.showbase.accept("arrow_down-up", self.set_hpr_movement, [1, 0])
        self.showbase.accept("arrow_left", self.set_hpr_movement, [0, 1])
        self.showbase.accept("arrow_left-up", self.set_hpr_movement, [0, 0])
        self.showbase.accept("arrow_right", self.set_hpr_movement, [0, -1])
        self.showbase.accept("arrow_right-up", self.set_hpr_movement, [0, 0])

        # increase / decrease speed
        self.showbase.accept("+", self.increase_speed)
        self.showbase.accept("-", self.decrease_speed)

        # disable modifier buttons to be able to move while pressing shift for
        # example
        self.showbase.mouseWatcherNode.set_modifier_buttons(ModifierButtons())
        self.showbase.buttonThrowers[0].node().set_modifier_buttons(ModifierButtons())

        # disable pandas builtin mouse control
        self.showbase.disableMouse()

        # add ourself as an update task which gets executed very early before
        # the rendering
        self.update_task = self.showbase.addTask(
            self.update, "RP_UpdateMovementController", sort=-40)

        # Hotkeys to connect to pstats and reset the initial position
        self.showbase.accept("1", PStatClient.connect)
        self.showbase.accept("3", self.reset_to_initial)

    def print_position(self):
        """ Prints the camera position and hpr """
        pos = self.showbase.cam.get_pos(self.showbase.render)
        hpr = self.showbase.cam.get_hpr(self.showbase.render)
        print("(Vec3({}, {}, {}), Vec3({}, {}, {})),".format(
            pos.x, pos.y, pos.z, hpr.x, hpr.y, hpr.z))

    def update(self, task):
        """ Internal update method """

        delta = self.clock_obj.get_dt()

        # Update mouse first
        if self.showbase.mouseWatcherNode.has_mouse():
            x = self.showbase.mouseWatcherNode.get_mouse_x()
            y = self.showbase.mouseWatcherNode.get_mouse_y()
            self.current_mouse_pos = (x * self.showbase.camLens.get_fov().x * self.mouse_sensivity,
                                      y * self.showbase.camLens.get_fov().y * self.mouse_sensivity)

            if self.mouse_enabled:
                diffx = self.last_mouse_pos[0] - self.current_mouse_pos[0]
                diffy = self.last_mouse_pos[1] - self.current_mouse_pos[1]

                # Don't move in the very beginning
                if self.last_mouse_pos[0] == 0 and self.last_mouse_pos[1] == 0:
                    diffx = 0
                    diffy = 0

                self.showbase.camera.set_h(self.showbase.camera.get_h() + diffx)
                self.showbase.camera.set_p(self.showbase.camera.get_p() - diffy)

            self.last_mouse_pos = self.current_mouse_pos[:]

        # Compute movement in render space
        movement_direction = (Vec3(self.movement[1], self.movement[0], 0) *
                              self.speed * delta * 100.0)

        # transform by the camera direction
        camera_quaternion = self.showbase.camera.get_quat(self.showbase.render)
        translated_direction = camera_quaternion.xform(movement_direction)

        # z-force is inddpendent of camera direction
        translated_direction.add_z(
            self.movement[2] * delta * 120.0 * self.speed)

        self.velocity += translated_direction * 0.15

        # apply the new position
        self.showbase.camera.set_pos(self.showbase.camera.get_pos() + self.velocity)

        # transform rotation (keyboard keys)
        rotation_speed = self.keyboard_hpr_speed * 100.0
        rotation_speed *= delta
        self.showbase.camera.set_hpr(
            self.showbase.camera.get_hpr() + Vec3(
                self.hpr_movement[0], self.hpr_movement[1], 0) * rotation_speed)

        # fade out velocity
        self.velocity = self.velocity * max(
            0.0, 1.0 - delta * 60.0 / max(0.01, self.smoothness))

        # bobbing
        ftime = self.clock_obj.get_frame_time()
        rotation = (ftime % self.bobbing_speed) / self.bobbing_speed
        rotation = (min(rotation, 1.0 - rotation) * 2.0 - 0.5) * 2.0
        if self.velocity.length_squared() > 1e-5 and self.speed > 1e-5:
            rotation *= self.bobbing_amount
            rotation *= min(1, self.velocity.length()) / self.speed * 0.5
        else:
            rotation = 0
        self.showbase.camera.set_r(rotation)
        return task.cont

    def play_motion_path(self, points, point_duration=1.2):
        """ Plays a motion path from the given set of points """
        fitter = CurveFitter()
        for i, (pos, hpr) in enumerate(points):
            fitter.add_xyz_hpr(i, pos, hpr)

        fitter.compute_tangents(1.0)
        curve = fitter.make_hermite()
        print("Starting motion path with", len(points), "CVs")

        self.showbase.render2d.hide()
        self.showbase.aspect2d.hide()

        self.curve = curve
        self.curve_time_start = self.clock_obj.get_frame_time()
        self.curve_time_end = self.clock_obj.get_frame_time() + len(points) * point_duration
        self.delta_time_sum = 0.0
        self.delta_time_count = 0
        self.showbase.addTask(self.camera_motion_update, "RP_CameraMotionPath", sort=-50)
        self.showbase.taskMgr.remove(self.update_task)

    def camera_motion_update(self, task):
        if self.clock_obj.get_frame_time() > self.curve_time_end:
            print("Camera motion path finished")

            # Print performance stats
            avg_ms = self.delta_time_sum / self.delta_time_count
            print("Average frame time (ms): {:4.1f}".format(avg_ms * 1000.0))
            print("Average frame rate: {:4.1f}".format(1.0 / avg_ms))

            self.update_task = self.showbase.addTask(
                self.update, "RP_UpdateMovementController", sort=-50)
            self.showbase.render2d.show()
            self.showbase.aspect2d.show()
            return task.done

        lerp = (self.clock_obj.get_frame_time() - self.curve_time_start) /\
            (self.curve_time_end - self.curve_time_start)
        lerp *= self.curve.get_max_t()

        pos, hpr = Point3(0), Vec3(0)
        self.curve.evaluate_xyz(lerp, pos)
        self.curve.evaluate_hpr(lerp, hpr)

        self.showbase.camera.set_pos(pos)
        self.showbase.camera.set_hpr(hpr)

        self.delta_time_sum += self.clock_obj.get_dt()
        self.delta_time_count += 1

        return task.cont
