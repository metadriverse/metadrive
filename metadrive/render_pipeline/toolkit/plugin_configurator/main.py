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

# This tool offers an interface to configure the pipeline

from __future__ import print_function

import os
import sys
import time
from threading import Thread
from functools import partial

# Change to the current directory
os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__))))

# Append the current directory to the path
sys.path.insert(0, os.getcwd())

# Add the render pipeline to the path
sys.path.insert(0, "../../")

from rplibs.six import iteritems  # noqa
from rplibs.pyqt_imports import *  #noqa

# Load the generated UI Layout
from ui.main_window_generated import Ui_MainWindow  # noqa

from rpcore.pluginbase.manager import PluginManager  # noqa
from rpcore.util.network_communication import NetworkCommunication  # noqa
from rpcore.mount_manager import MountManager  # noqa


class PluginConfigurator(QMainWindow, Ui_MainWindow):
    """ Interface to change the plugin settings """
    def __init__(self):

        # Init mounts
        self._mount_mgr = MountManager(None)
        self._mount_mgr.mount()

        self._plugin_mgr = PluginManager(None)
        self._plugin_mgr.requires_daytime_settings = False

        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        self._current_plugin = None
        self._current_plugin_instance = None
        self.lbl_restart_pipeline.hide()
        self._set_settings_visible(False)
        self._update_queue = list()

        qt_connect(self.lst_plugins, "itemSelectionChanged()", self.on_plugin_selected)
        qt_connect(self.lst_plugins, "itemChanged(QListWidgetItem*)", self.on_plugin_state_changed)
        qt_connect(self.btn_reset_plugin_settings, "clicked()", self.on_reset_plugin_settings)

        self._load_plugin_list()

        # Adjust column widths
        self.table_plugin_settings.setColumnWidth(0, 140)
        self.table_plugin_settings.setColumnWidth(1, 105)
        self.table_plugin_settings.setColumnWidth(2, 160)

        update_thread = Thread(target=self.update_thread, args=())
        update_thread.start()

    def closeEvent(self, event):  # noqa
        event.accept()
        import os
        os._exit(1)

    def on_reset_plugin_settings(self):
        """ Gets called when the user wants to reset settings of a plugin """

        # Ask the user if he's really sure about it
        msg = "Are you sure you want to reset the settings of '"
        msg += self._current_plugin_instance.name + "'?\n"
        msg += "This does not reset the Time of Day settings of this plugin.\n\n"
        msg += "!! This cannot be undone !! They will be lost forever (a long time!)."
        reply = QMessageBox.question(self, "Warning", msg, QMessageBox.Yes, QMessageBox.No)
        if reply == QMessageBox.Yes:

            QMessageBox.information(self, "Success", "Settings have been reset! You may have to restart the pipeline.")
            self._plugin_mgr.reset_plugin_settings(self._current_plugin)

            # Save config
            self._plugin_mgr.save_overrides("/$$rpconfig/plugins.yaml")

            # Always show the restart hint, even if its not always required
            self._show_restart_hint()

            # Re-render everything
            self._load_plugin_list()

    def on_plugin_state_changed(self, item):
        """ Handler when a plugin got activated/deactivated """
        plugin_id = item._plugin_id
        state = item.checkState() == Qt.Checked
        self._plugin_mgr.set_plugin_enabled(plugin_id, state)
        self._rewrite_plugin_config()
        self._show_restart_hint()

    def on_plugin_selected(self):
        """ Gets called when a plugin got selected in the plugin list """
        selected_item = self.lst_plugins.selectedItems()
        if not selected_item:
            self._current_plugin = None
            self._current_plugin_instance = None
            self._set_settings_visible(False)
            return
        assert len(selected_item) == 1
        selected_item = selected_item[0]
        self._current_plugin = selected_item._plugin_id
        self._current_plugin_instance = self._plugin_mgr.instances[self._current_plugin]
        assert (self._current_plugin_instance is not None)
        self._render_current_plugin()
        self._set_settings_visible(True)

    def update_thread(self):
        """ Internal update thread """
        while True:
            if len(self._update_queue) > 0:
                item = self._update_queue.pop(-1)
                NetworkCommunication.send_async(NetworkCommunication.CONFIG_PORT, item)

                if item.startswith("setval "):
                    setting_id = item.split()[1]
                    for entry in list(self._update_queue):
                        if entry.split()[1] == setting_id:
                            self._update_queue.remove(entry)
            time.sleep(0.2)

    def _rewrite_plugin_config(self):
        """ Rewrites the plugin configuration """
        self._plugin_mgr.save_overrides("/$$rpconfig/plugins.yaml")

    def _render_current_plugin(self):
        """ Displays the currently selected plugin """
        self.lbl_plugin_name.setText(
            self._current_plugin_instance.name.upper() + " <span style='color: #999; margin-left: 5px;'>[" +
            self._current_plugin_instance.plugin_id + "]</span>"
        )

        version_str = "Version " + self._current_plugin_instance.version
        version_str += " by " + self._current_plugin_instance.author

        self.lbl_plugin_version.setText(version_str)
        self.lbl_plugin_desc.setText(self._current_plugin_instance.description)

        self._render_current_settings()

    def _show_restart_hint(self):
        """ Shows a hint to restart the pipeline """
        self.lbl_restart_pipeline.show()

    def _render_current_settings(self):
        """ Renders the current plugin settings """
        settings = self._plugin_mgr.settings[self._current_plugin]

        # remove all rows
        while self.table_plugin_settings.rowCount() > 0:
            self.table_plugin_settings.removeRow(0)

        label_font = QFont()
        label_font.setPointSize(10)
        label_font.setFamily("Roboto")

        desc_font = QFont()
        desc_font.setPointSize(8)
        desc_font.setFamily("Roboto")

        for index, (name, handle) in enumerate(iteritems(settings)):
            if not handle.should_be_visible(settings):
                continue

            row_index = self.table_plugin_settings.rowCount()

            # Increase row count
            self.table_plugin_settings.insertRow(self.table_plugin_settings.rowCount())

            label = QLabel()
            label.setText(handle.label)
            label.setWordWrap(True)
            label.setFont(label_font)

            if not (handle.shader_runtime or handle.runtime):
                label.setStyleSheet("color: #999;")

            if handle.display_conditions:
                label.setStyleSheet(label.styleSheet() + "padding-left: 10px;")

            label.setMargin(10)

            self.table_plugin_settings.setCellWidget(row_index, 0, label)

            item_default = QTableWidgetItem()
            item_default.setText(str(handle.default))
            item_default.setTextAlignment(Qt.AlignCenter)
            self.table_plugin_settings.setItem(row_index, 1, item_default)

            setting_widget = self._get_widget_for_setting(name, handle)
            self.table_plugin_settings.setCellWidget(row_index, 2, setting_widget)

            label_desc = QLabel()
            label_desc.setText(handle.description)
            label_desc.setWordWrap(True)
            label_desc.setFont(desc_font)
            label_desc.setStyleSheet("color: #555;padding: 5px;")

            self.table_plugin_settings.setCellWidget(row_index, 3, label_desc)

    def _do_update_setting(self, setting_id, value):
        """ Updates a setting of the current plugin """

        # Check whether the setting is a runtime setting
        setting_handle = self._plugin_mgr.get_setting_handle(self._current_plugin, setting_id)

        # Skip the setting in case the value is equal
        if setting_handle.value == value:
            return

        # Otherwise set the new value
        setting_handle.set_value(value)
        self._rewrite_plugin_config()

        if not setting_handle.runtime and not setting_handle.shader_runtime:
            self._show_restart_hint()
        else:
            # In case the setting is dynamic, notice the pipeline about it:
            # print("Sending reload packet ...")
            self._update_queue.append("setval {}.{} {}".format(self._current_plugin, setting_id, value))

        # Update GUI, but only in case of enum and bool values, since they can trigger
        # display conditions:
        if setting_handle.type == "enum" or setting_handle.type == "bool":
            self._render_current_settings()

    def _on_setting_bool_changed(self, setting_id, value):
        self._do_update_setting(setting_id, value == Qt.Checked)

    def _on_setting_scalar_changed(self, setting_id, value):
        self._do_update_setting(setting_id, value)

    def _on_setting_enum_changed(self, setting_id, value):
        self._do_update_setting(setting_id, value)

    def _on_setting_power_of_two_changed(self, setting_id, value):
        self._do_update_setting(setting_id, value)

    def _on_setting_slider_changed(self, setting_id, setting_type, bound_objs, value):
        if setting_type == "float":
            value /= 100000.0
        self._do_update_setting(setting_id, value)

        for obj in bound_objs:
            obj.setValue(value)

    def _on_setting_spinbox_changed(self, setting_id, setting_type, bound_objs, value):
        self._do_update_setting(setting_id, value)
        # Assume objects are sliders, so we need to rescale the value
        for obj in bound_objs:
            obj.setValue(value * 100000.0 if setting_type == "float" else value)

    def _choose_path(self, setting_id, setting_handle, bound_objs):
        """ Shows a file chooser to show an path from """

        this_dir = os.path.dirname(os.path.realpath(__file__))
        plugin_dir = os.path.join(this_dir, "../../rpplugins/" + self._current_plugin, "resources")
        plugin_dir = os.path.abspath(plugin_dir)
        search_dir = os.path.join(plugin_dir, setting_handle.base_path)

        print("Plugin dir =", plugin_dir)
        print("Search dir =", search_dir)

        current_file = setting_handle.value.replace("\\", "/").split("/")[-1]
        print("Current file =", current_file)
        file_dlg = QFileDialog(self, "Choose File ..", search_dir, setting_handle.file_type)
        file_dlg.selectFile(current_file)
        # file_dlg.setViewMode(QFileDialog.Detail)

        if file_dlg.exec_():
            filename = file_dlg.selectedFiles()
            filename = filename[-1]
            print("QT selected files returned:", filename)

            filename = os.path.relpath(str(filename), plugin_dir)
            filename = filename.replace("\\", "/")
            print("Relative path is", filename)
            self._do_update_setting(setting_id, filename)

            display_file = filename.split("/")[-1]
            for obj in bound_objs:
                obj.setText(display_file)

    def _get_widget_for_setting(self, setting_id, setting):
        """ Returns an appropriate widget to control the given setting """

        widget = QWidget()
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        widget.setLayout(layout)

        if setting.type == "bool":
            box = QCheckBox()
            box.setChecked(Qt.Checked if setting.value else Qt.Unchecked)
            qt_connect(box, "stateChanged(int)", partial(self._on_setting_bool_changed, setting_id))
            layout.addWidget(box)

        elif setting.type == "float" or setting.type == "int":

            if setting.type == "float":
                box = QDoubleSpinBox()

                if setting.maxval - setting.minval <= 2.0:
                    box.setDecimals(4)
            else:
                box = QSpinBox()
            box.setMinimum(setting.minval)
            box.setMaximum(setting.maxval)
            box.setValue(setting.value)

            box.setAlignment(Qt.AlignCenter)

            slider = QSlider()
            slider.setOrientation(Qt.Horizontal)

            if setting.type == "float":
                box.setSingleStep(abs(setting.maxval - setting.minval) / 100.0)
                slider.setMinimum(setting.minval * 100000.0)
                slider.setMaximum(setting.maxval * 100000.0)
                slider.setValue(setting.value * 100000.0)
            elif setting.type == "int":
                box.setSingleStep(max(1, (setting.maxval - setting.minval) / 32))
                slider.setMinimum(setting.minval)
                slider.setMaximum(setting.maxval)
                slider.setValue(setting.value)

            layout.addWidget(box)
            layout.addWidget(slider)

            qt_connect(
                slider, "valueChanged(int)", partial(self._on_setting_slider_changed, setting_id, setting.type, [box])
            )

            value_type = "int" if setting.type == "int" else "double"

            qt_connect(
                box, "valueChanged(" + value_type + ")",
                partial(self._on_setting_spinbox_changed, setting_id, setting.type, [slider])
            )

        elif setting.type == "enum":
            box = QComboBox()
            for value in setting.values:
                box.addItem(value)
            qt_connect(box, "currentIndexChanged(QString)", partial(self._on_setting_enum_changed, setting_id))
            box.setCurrentIndex(setting.values.index(setting.value))
            box.setMinimumWidth(145)

            layout.addWidget(box)

        elif setting.type == "power_of_two":

            box = QComboBox()
            resolutions = [str(2**i) for i in range(1, 32) if 2**i >= setting.minval and 2**i <= setting.maxval]
            for value in resolutions:
                box.addItem(value)
            qt_connect(box, "currentIndexChanged(QString)", partial(self._on_setting_power_of_two_changed, setting_id))
            box.setCurrentIndex(resolutions.index(str(setting.value)))
            box.setMinimumWidth(145)
            layout.addWidget(box)

        elif setting.type == "sample_sequence":

            box = QComboBox()
            for value in setting.sequences:
                box.addItem(value)
            qt_connect(box, "currentIndexChanged(QString)", partial(self._on_setting_enum_changed, setting_id))
            box.setCurrentIndex(setting.sequences.index(str(setting.value)))
            box.setMinimumWidth(145)
            layout.addWidget(box)

        elif setting.type == "path":

            label = QLabel()
            display_file = setting.value.replace("\\", "/").split("/")[-1]

            desc_font = QFont()
            desc_font.setPointSize(7)
            desc_font.setFamily("Roboto")

            label.setText(display_file)
            label.setFont(desc_font)

            button = QPushButton()
            button.setText("Choose File ...")
            qt_connect(button, "clicked()", partial(self._choose_path, setting_id, setting, (label, )))

            layout.addWidget(label)
            layout.addWidget(button)

        else:
            print("ERROR: Unkown setting type:", setting.type)

        return widget

    def _set_settings_visible(self, flag):
        """ Sets whether the settings panel is visible or not """
        if flag:
            self.frame_details.show()
        else:
            self.frame_details.hide()

    def _load_plugin_list(self):
        """ Reloads the whole plugin list """
        print("Loading plugin list")

        # Reset selection
        self._current_plugin = None
        self._current_plugin_instance = None
        self._set_settings_visible(False)

        # Plugins are all plugins in the plugins directory
        self._plugin_mgr.unload()
        self._plugin_mgr.load()

        self.lst_plugins.clear()
        plugins = sorted(iteritems(self._plugin_mgr.instances), key=lambda plg: plg[1].name)

        item_font = QFont()
        item_font.setBold(False)
        item_font.setPointSize(10)

        for plugin_id, instance in plugins:

            item = QListWidgetItem()
            item.setText(" " + instance.name)
            item.setFont(item_font)

            if self._plugin_mgr.is_plugin_enabled(plugin_id):
                item.setCheckState(Qt.Checked)
            else:
                item.setCheckState(Qt.Unchecked)

            item._plugin_id = plugin_id
            self.lst_plugins.addItem(item)

        self.lst_plugins.setCurrentRow(0)


# Start application
app = QApplication(sys.argv)
qt_register_fonts()
configurator = PluginConfigurator()
configurator.show()
app.exec_()
