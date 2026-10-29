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

# This tool offers an interface to configure the time of day settings

from __future__ import print_function, division

import os
import sys
import time
from threading import Thread

# Change to the current directory
os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__))))

# Add the render pipeline to the path
sys.path.insert(0, "../../")

from rplibs.six import iteritems  # noqa
from rplibs.pyqt_imports import *  # noqa

from curve_widget import CurveWidget  # noqa

from rpcore.pluginbase.manager import PluginManager  # noqa
from rpcore.mount_manager import MountManager  # noqa
from rpcore.util.network_communication import NetworkCommunication  # noqa

from ui.main_window_generated import Ui_MainWindow  # noqa
from ui.point_insert_dialog_generated import Ui_Dialog as Ui_PointDialog  # noqa


class PointDialog(QDialog, Ui_PointDialog):
    def __init__(self, parent):
        QDialog.__init__(self, parent)
        self.setupUi(self)
        qt_connect(self.btn_insert, "clicked()", lambda: self.done(1))

    def get_value(self):
        time = self.ipt_time.time()
        val = self.ipt_value.value()
        return time, val


class DayTimeEditor(QMainWindow, Ui_MainWindow):
    """ This is the main editor class which handles the user interface """
    def __init__(self):

        # Init mounts
        self._mount_mgr = MountManager(None)
        self._mount_mgr.mount()

        self._plugin_mgr = PluginManager(None)
        self._plugin_mgr.load()

        QMainWindow.__init__(self)
        self.setupUi()
        self._tree_widgets = []
        self._cmd_queue = set()

        self._selected_setting_handle = None
        self._selected_setting = None
        self._selected_plugin = None
        self._current_time = 0.5

        self._update_settings_list()
        self._on_time_changed(self.time_slider.value())

        self._bg_thread = Thread(target=self.updateThread)
        self._bg_thread.start()

    def set_settings_visible(self, visibility):
        if not visibility:
            self.frame_current_setting.hide()
        else:
            self.frame_current_setting.show()

    def closeEvent(self, event):  # noqa
        event.accept()
        import os
        os._exit(1)

    def updateThread(self):  # noqa
        """ Seperate update thread """

        while True:
            if self._cmd_queue:
                cmd = self._cmd_queue.pop()
                if cmd == "settime":
                    NetworkCommunication.send_async(
                        NetworkCommunication.DAYTIME_PORT, "settime " + str(self._current_time)
                    )
                    continue
                elif cmd == "write_settings":
                    self._plugin_mgr.save_daytime_overrides("/$$rpconfig/daytime.yaml")
                    NetworkCommunication.send_async(NetworkCommunication.DAYTIME_PORT, "loadconf")
                else:
                    print("Unkown cmd:", cmd)

            time.sleep(0.1)

    def setupUi(self):  # noqa
        """ Setups the UI Components """
        Ui_MainWindow.setupUi(self, self)
        self.settings_tree.setColumnWidth(0, 160)
        self.settings_tree.expandAll()

        self.edit_widget = CurveWidget(self)
        self.edit_widget.set_change_handler(self._on_curve_edited)
        self.prefab_edit_widget.addWidget(self.edit_widget)

        qt_connect(self.time_slider, "valueChanged(int)", self._on_time_changed)
        qt_connect(self.settings_tree, "itemSelectionChanged()", self._on_setting_selected)
        qt_connect(self.btn_insert_point, "clicked()", self._insert_point)
        qt_connect(self.btn_reset, "clicked()", self._reset_settings)

    def _reset_settings(self):
        """ Resets the current plugins settings """
        # QMessageBox.warning(self, "Houston, we have a problem!",
        #     "This functionality is not yet implemented! Blame tobspr if you need it.\n\n"
        #     "On a more serious note, you can still hand-edit config/daytime.yaml.",
        #     QMessageBox.Ok, QMessageBox.Ok)

        # Ask the user if he's really sure about it
        msg = "Are you sure you want to reset the control points of '" +\
              self._selected_setting_handle.label + "'?\n"
        msg += "!! This cannot be undone !! They will be lost forever (a long time!)."
        reply = QMessageBox.question(self, "Warning", msg, QMessageBox.Yes, QMessageBox.No)
        if reply == QMessageBox.Yes:

            QMessageBox.information(self, "Success", "Control points have been reset!")
            default = self._selected_setting_handle.default
            self._selected_setting_handle.curves[0].set_single_value(default)
            self._update_settings_list()
            self._cmd_queue.add("write_settings")

    def _insert_point(self):
        """ Asks the user to insert a new point """
        dialog = PointDialog(self)
        if dialog.exec_():
            time, val = dialog.get_value()
            minutes = (time.hour() * 60 + time.minute()) / (24 * 60)

            if (val < self._selected_setting_handle.minvalue or val > self._selected_setting_handle.maxvalue):
                QMessageBox.information(self, "Invalid Value", "Value is out of setting range!", QMessageBox.Ok)
                return

            val_linear = self._selected_setting_handle.get_linear_value(val)
            self._selected_setting_handle.curves[0].append_cv(minutes, val_linear)
            self._cmd_queue.add("write_settings")

    def _update_tree_widgets(self):
        """ Updates the tree widgets """
        for setting_handle, widget in self._tree_widgets:
            value = setting_handle.get_scaled_value_at(self._current_time)
            formatted = setting_handle.format(value)
            widget.setText(1, formatted)

            if setting_handle.type == "color":
                widget.setBackground(1, QBrush(QColor(*value)))

    def _on_curve_edited(self):
        """ Called when the curve got edited in the curve widget """
        self._cmd_queue.add("write_settings")
        self._update_tree_widgets()

    def _on_setting_selected(self):
        """ Called when a setting got selected in the settings tree """
        selected = self.settings_tree.selectedItems()
        if len(selected) != 1:
            self._selected_setting = None
            self._selected_plugin = None
            self._selected_setting_handle = None
            self.edit_widget.set_curves([])
            self.set_settings_visible(False)
        else:
            selected = selected[0]

            self._selected_plugin = selected._plugin_id
            self._selected_setting = selected._setting_id
            self._selected_setting_handle = selected._setting_handle

            self.lbl_current_setting.setText(self._selected_setting_handle.label)
            self.lbl_setting_desc.setText(self._selected_setting_handle.description)

            self.edit_widget.set_curves(self._selected_setting_handle.curves)

            if self._selected_setting_handle.type == "color":
                self.edit_widget.set_unit_processor(lambda x: str(int(x * 255)))
                self.btn_insert_point.hide()
            else:
                self.edit_widget.set_unit_processor(
                    lambda x: self._selected_setting_handle.format(self._selected_setting_handle.get_scaled_value(x))
                )
                self.btn_insert_point.show()

            self.set_settings_visible(True)
            self._update_tree_widgets()

    def _on_time_changed(self, val):
        """ Handler when the time slider got moved """
        hour = val // (60 * 60 * 60)
        minute = (val // (60 * 60)) % 60
        ftime = float(val) / (24 * 60 * 60 * 60)

        self.time_label.setText(str(hour).zfill(2) + ":" + str(minute).zfill(2))
        self.time_float_label.setText("{:1.4f}".format(ftime))
        self.edit_widget.set_current_time(ftime)
        self._current_time = ftime
        self._update_tree_widgets()
        self._cmd_queue.add("settime")

    def _update_settings_list(self):
        """ Updates the list of visible settings """

        self.settings_tree.clear()
        self._tree_widgets = []

        first_item = None

        for plugin_id, plugin in iteritems(self._plugin_mgr.instances):

            daytime_settings = self._plugin_mgr.day_settings[plugin_id]

            if not daytime_settings:
                # Skip plugins with empty settings
                continue

            plugin_head = QTreeWidgetItem(self.settings_tree)
            plugin_head.setText(0, plugin.name)
            plugin_head.setFlags(Qt.ItemIsEnabled)
            font = QFont()
            font.setBold(True)
            if not self._plugin_mgr.is_plugin_enabled(plugin_id):
                plugin_head.setText(0, plugin.name)
            plugin_head.setFont(0, font)

            # Display all settings
            for setting, setting_handle in iteritems(daytime_settings):
                setting_item = QTreeWidgetItem(plugin_head)
                setting_item.setText(0, setting_handle.label)
                if PYQT_VERSION == 4:
                    setting_item.setTextColor(0, QColor(150, 150, 150))
                else:
                    setting_item.setForeground(0, QColor(150, 150, 150))
                setting_item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable)
                setting_item._setting_id = setting
                setting_item._setting_handle = setting_handle
                setting_item._plugin_id = plugin_id
                setting_item.setToolTip(0, setting_handle.description)
                setting_item.setToolTip(1, setting_handle.description)
                self._tree_widgets.append((setting_handle, setting_item))
                if not first_item:
                    first_item = setting_item

        self.settings_tree.expandAll()
        if first_item:
            self.settings_tree.setCurrentItem(first_item)


# Start application
app = QApplication(sys.argv)
qt_register_fonts()
editor = DayTimeEditor()
editor.show()
app.exec_()
