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

# This tool offers an interface to modify materials
# Hacked together in a hour, I'm sorry.

from __future__ import print_function

import os
import sys
import time
import math
import tempfile
import colorsys
from threading import Thread
from functools import partial

os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0, os.getcwd())
sys.path.insert(0, "../../")

from rplibs.six import iteritems  # noqa
from rplibs.pyqt_imports import *  # noqa

from ui.main_window_generated import Ui_MainWindow  # noqa
from rpcore.util.network_communication import NetworkCommunication  # noqa

# Allow working with an older version of the material DB.
# ONLY for debugging the viewer.
ALLOW_OUTDATED_MATERIALS = False


class MaterialData:

    def __init__(self):
        self.name = ""
        self.shading_model = 0
        self.metallic = False
        self.roughness = 0.0
        self.specular = 0.0
        self.normal_strength = 0.0
        self.shading_model_param1 = 0.0
        self.shading_model_param2 = 0.0
        self.basecolor_r = 0.6
        self.basecolor_g = 0.6
        self.basecolor_b = 0.6


class MaterialEditor(QMainWindow, Ui_MainWindow):

    """ Interface to change the plugin settings """

    SHADING_MODELS = [
        ("Default", 0, None),
        ("Emissive", 1, None),
        ("Clearcoat", 2, None),
        ("Transparent", 3, "Transparency"),
        ("Skin", 4, None),
        ("Foliage", 5, None),
    ]

    def __init__(self):
        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)

        self.in_update = False
        self.materials = []
        self.material = MaterialData()

        self.setupUi(self)
        self.init_shading_models()
        self.init_bindings()
        self.update_material_list()
        self.on_material_selected()

    def init_bindings(self):
        qt_connect(self.cb_shading_model, "currentIndexChanged", self.read_from_ui)
        qt_connect(self.cb_material, "currentIndexChanged", self.on_material_selected)
        qt_connect(self.cb_metallic, "stateChanged", self.read_from_ui)

        self.sliders = [
            (self.slider_roughness, self.lbl_roughness, 0.0, 1.0, "roughness"),
            (self.slider_specular, self.lbl_specular, 1.0, 2.51, "specular"),
            (self.slider_normal, self.lbl_normal, 0.0, 1.0, "normal_strength"),
            (self.slider_param1, self.lbl_param1, 0.0, 1.0, "shading_model_param1"),
        ]

        for slider, lbl, start, end, prop in self.sliders:
            qt_connect(slider, "valueChanged", self.read_from_ui)

        qt_connect(self.basecolor_1, "valueChanged", self.read_from_ui)
        qt_connect(self.basecolor_2, "valueChanged", self.read_from_ui)
        qt_connect(self.basecolor_3, "valueChanged", self.read_from_ui)

        for cb in (self.cb_rgb, self.cb_srgb, self.cb_hsv):
            qt_connect(cb, "toggled", self.write_to_ui)

    def update_ui(self):
        # Basecolor
        labels = "R", "G", "B"
        if self.cb_hsv.isChecked():
            labels = "H", "S", "V"
        self.lbl_basecolor1.setText(labels[0])
        self.lbl_basecolor2.setText(labels[1])
        self.lbl_basecolor3.setText(labels[2])

        a, b, c = (self.basecolor_1.value() / 100.0,
                   self.basecolor_2.value() / 100.0,
                   self.basecolor_3.value() / 100.0)
        rgb = self.tuple_to_basecolor(a, b, c)
        self.lbl_basecolor_val1.setText("{:0.2f}".format(a))
        self.lbl_basecolor_val2.setText("{:0.2f}".format(b))
        self.lbl_basecolor_val3.setText("{:0.2f}".format(c))
        self.lbl_color_preview.setStyleSheet("background: rgb({}, {}, {});".format(
            int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)))

        # Shading model
        self._update_shading_model()

    def read_from_ui(self):
        if self.in_update:
            return

        # Rest of sliders
        for slider, lbl, start, end, prop in self.sliders:
            val = (slider.value() / 100.0) * (end - start) + start
            lbl.setText("{:0.2f}".format(val))
            setattr(self.material, prop, val)

        # Basecolor
        rgb = self._get_ui_basecolor_rgb()
        self.material.basecolor_r = rgb[0]
        self.material.basecolor_g = rgb[1]
        self.material.basecolor_b = rgb[2]

        # Metallic
        self.material.metallic = self.cb_metallic.isChecked()

        # Shading model
        self.material.shading_model = self.cb_shading_model.currentIndex()

        self.update_ui()
        self.send_update()

    def write_to_ui(self):
        self.in_update = True

        # Basecolor
        values = self.basecolor_to_tuple(self.material)
        self.basecolor_1.setValue(values[0] * 100.0)
        self.basecolor_2.setValue(values[1] * 100.0)
        self.basecolor_3.setValue(values[2] * 100.0)

        # Shading model
        self.cb_shading_model.setCurrentIndex(self.material.shading_model)

        # Metallics
        self.cb_metallic.setChecked(self.material.metallic)
        self.slider_specular.setEnabled(not self.material.metallic)

        # Rest of sliders
        for slider, lbl, start, end, prop in self.sliders:
            val = getattr(self.material, prop)
            slider.setValue((val - start) / (end - start) * 100.0)

        self.in_update = False
        self.update_ui()

    def _get_ui_basecolor_rgb(self):
        """ Extracts the RGB color which is currently edited in the UI """
        a, b, c = (self.basecolor_1.value() / 100.0,
                   self.basecolor_2.value() / 100.0,
                   self.basecolor_3.value() / 100.0)
        return self.tuple_to_basecolor(a, b, c)

    def update_material_list(self):
        temp_path = os.path.join(tempfile.gettempdir(), "rp_materials.data")
        print("Waiting for creation of", temp_path)
        if not ALLOW_OUTDATED_MATERIALS:
            try:
                os.remove(temp_path)
            except:
                pass
        NetworkCommunication.send_async(NetworkCommunication.MATERIAL_PORT, "dump_materials " + temp_path)
        start_time = time.time()
        while not os.path.isfile(temp_path) and time.time() - start_time < 5.0:
            time.sleep(0.5)
        if not os.path.isfile(temp_path):
            QMessageBox.critical(
                self, "Error", "Render Pipeline not responding! Make sure a render pipeline application is running, and try again later.")
            sys.exit(-1)
        if not ALLOW_OUTDATED_MATERIALS:
            time.sleep(0.5)
        self._load_material_list(temp_path)

    def _load_material_list(self, pth):
        self.materials = []
        self.cb_material.clear()
        with open(pth) as handle:
            for line in handle.readlines():
                parts = line.strip().split(" ")
                material = self._read_in_material(parts)
                self.materials.append(material)
                self.cb_material.addItem(material.name)

    def _read_in_material(self, parts):
        material = MaterialData()
        material.name = parts[0]
        material.basecolor_r = float(parts[1])
        material.basecolor_g = float(parts[2])
        material.basecolor_b = float(parts[3])
        material.roughness = float(parts[4])
        material.specular = float(parts[5])
        material.metallic = float(parts[6]) > 0.5
        material.shading_model = int(float(parts[7]))
        material.normal_strength = float(parts[8])
        material.shading_model_param1 = float(parts[9])
        material.shading_model_param2 = float(parts[10])
        return material

    def basecolor_to_tuple(self, mat):
        def to_srgb(v): return math.pow(v, 1.0 / 2.2)
        if self.cb_rgb.isChecked():
            return mat.basecolor_r, mat.basecolor_g, mat.basecolor_b
        elif self.cb_srgb.isChecked():
            return to_srgb(mat.basecolor_r), to_srgb(mat.basecolor_g), to_srgb(mat.basecolor_b)
        elif self.cb_hsv.isChecked():
            return colorsys.rgb_to_hsv(mat.basecolor_r, mat.basecolor_g, mat.basecolor_b)
        else:
            assert False

    def tuple_to_basecolor(self, a, b, c):
        def from_srgb(v): return math.pow(v, 2.2)
        if self.cb_rgb.isChecked():
            return a, b, c
        elif self.cb_srgb.isChecked():
            return from_srgb(a), from_srgb(b), from_srgb(c)
        elif self.cb_hsv.isChecked():
            return colorsys.hsv_to_rgb(a, b, c)
        else:
            assert False

    def on_material_selected(self):
        index = self.cb_material.currentIndex()
        if index < 0 or index >= len(self.materials):
            print("Invalid material with index", index, "only have", len(self.materials), "materials")
            return
        self.material = self.materials[index]
        print("Loaded material", self.material.name)
        self.write_to_ui()

    def send_update(self):
        serialized = ("{} " * 11).format(
            self.material.name,
            self.material.basecolor_r,
            self.material.basecolor_g,
            self.material.basecolor_b,
            self.material.roughness,
            self.material.specular,
            1.0 if self.material.metallic else 0.0,
            self.material.shading_model,
            self.material.normal_strength,
            self.material.shading_model_param1,
            self.material.shading_model_param2,
        )
        NetworkCommunication.send_async(NetworkCommunication.MATERIAL_PORT, "update_material " + serialized)

    def _update_shading_model(self):
        name, val, optional_param = self.SHADING_MODELS[self.cb_shading_model.currentIndex()]
        if optional_param is None:
            self.slider_param1.setEnabled(False)
            self.lbl_shading_model_param1.setText("Unused")
        else:
            self.slider_param1.setEnabled(True)
            self.lbl_shading_model_param1.setText(optional_param)

        self.cb_metallic.show()
        self.slider_roughness.setEnabled(True)
        self.slider_normal.setEnabled(True)

        if name == "Emissive":
            self.cb_metallic.hide()
            self.slider_roughness.setEnabled(False)
            self.slider_specular.setEnabled(False)
            self.slider_normal.setEnabled(False)
        elif name == "Clearcoat":
            self.cb_metallic.hide()
            self.slider_specular.setEnabled(False)
        elif name == "Transparent":
            pass
        elif name == "Skin":
            self.cb_metallic.hide()
        elif name == "Foliage":
            self.cb_metallic.hide()

    def init_shading_models(self):
        self.cb_shading_model.clear()
        for name, val, optional_param in self.SHADING_MODELS:
            self.cb_shading_model.addItem(name)

# Start application
app = QApplication(sys.argv)
qt_register_fonts()
editor = MaterialEditor()
editor.show()
app.exec_()
