# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'main_window.ui'
#
# Created by: PyQt5 UI code generator 5.7
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(973, 632)
        MainWindow.setMinimumSize(QtCore.QSize(973, 632))
        MainWindow.setMaximumSize(QtCore.QSize(973, 632))
        palette = QtGui.QPalette()
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(63, 63, 63))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(52, 52, 52))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(21, 21, 21))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(28, 28, 28))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(21, 21, 21))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Active, QtGui.QPalette.ToolTipText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(63, 63, 63))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(52, 52, 52))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(21, 21, 21))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(28, 28, 28))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(21, 21, 21))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Inactive, QtGui.QPalette.ToolTipText, brush)
        brush = QtGui.QBrush(QtGui.QColor(21, 21, 21))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.WindowText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Button, brush)
        brush = QtGui.QBrush(QtGui.QColor(63, 63, 63))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Light, brush)
        brush = QtGui.QBrush(QtGui.QColor(52, 52, 52))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Midlight, brush)
        brush = QtGui.QBrush(QtGui.QColor(21, 21, 21))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Dark, brush)
        brush = QtGui.QBrush(QtGui.QColor(28, 28, 28))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Mid, brush)
        brush = QtGui.QBrush(QtGui.QColor(21, 21, 21))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Text, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.BrightText, brush)
        brush = QtGui.QBrush(QtGui.QColor(21, 21, 21))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ButtonText, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Base, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 255))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Window, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.Shadow, brush)
        brush = QtGui.QBrush(QtGui.QColor(42, 42, 42))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.AlternateBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(255, 255, 220))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipBase, brush)
        brush = QtGui.QBrush(QtGui.QColor(0, 0, 0))
        brush.setStyle(QtCore.Qt.SolidPattern)
        palette.setBrush(QtGui.QPalette.Disabled, QtGui.QPalette.ToolTipText, brush)
        MainWindow.setPalette(palette)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/res/res/icon.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setStyleSheet(
            "QMainWindow { background: #fff;}\n"
            "*, QLabel { font-family: Roboto; font-weight: 300; }\n"
            "\n"
            "QScrollBar {\n"
            "    background: #eee;\n"
            "}\n"
            "\n"
            "\n"
            "QScrollBar:vertical {\n"
            "                        width: 9px;\n"
            "    margin: 0;\n"
            "                      }\n"
            "\n"
            "                      QScrollBar::handle:vertical {\n"
            "                        min-height: 15px;\n"
            "                        background: #aaa;\n"
            "\n"
            "                      }\n"
            "\n"
            "\n"
            "                      QScrollBar::handle:vertical:hover {\n"
            "                        \n"
            "                        background: #999;\n"
            "\n"
            "                      }\n"
            "                      QScrollBar::add-line:vertical {\n"
            "                      }\n"
            "\n"
            "                      QScrollBar::sub-line:vertical {\n"
            "                      }\n"
            "\n"
            "\n"
            "                      QScrollBar::add-page:vertical {\n"
            "                        background: #ddd;\n"
            "\n"
            "                      }\n"
            "                      QScrollBar::sub-page:vertical {\n"
            "                        background: #ddd;\n"
            "                    }\n"
            ""
        )
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame_current_setting = QtWidgets.QFrame(self.centralwidget)
        self.frame_current_setting.setGeometry(QtCore.QRect(350, 80, 641, 551))
        self.frame_current_setting.setStyleSheet("QFrame { background: #e5e5e5;}")
        self.frame_current_setting.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_current_setting.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_current_setting.setObjectName("frame_current_setting")
        self.lbl_current_setting = QtWidgets.QLabel(self.frame_current_setting)
        self.lbl_current_setting.setGeometry(QtCore.QRect(20, 16, 221, 31))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(-1)
        font.setBold(False)
        font.setWeight(37)
        self.lbl_current_setting.setFont(font)
        self.lbl_current_setting.setStyleSheet("color: #555; border: 0;  font-size: 15px; text-transform: uppercase;")
        self.lbl_current_setting.setObjectName("lbl_current_setting")
        self.btn_reset = QtWidgets.QPushButton(self.frame_current_setting)
        self.btn_reset.setGeometry(QtCore.QRect(500, 14, 101, 31))
        self.btn_reset.setStyleSheet(
            "QPushButton {\n"
            "color: #eee;\n"
            "background: #666;\n"
            "border: 0; \n"
            "}\n"
            "\n"
            "QPushButton:hover {\n"
            "    background: #555;\n"
            "}\n"
            "\n"
            "QPushButton:pressed {\n"
            "    background: #111;\n"
            "}"
        )
        self.btn_reset.setObjectName("btn_reset")
        self.frame_3 = QtWidgets.QFrame(self.frame_current_setting)
        self.frame_3.setGeometry(QtCore.QRect(20, 94, 581, 441))
        self.frame_3.setStyleSheet("QFrame {\n" "}")
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.frame_3)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 571, 441))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.prefab_edit_widget = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.prefab_edit_widget.setContentsMargins(0, 0, 0, 0)
        self.prefab_edit_widget.setSpacing(0)
        self.prefab_edit_widget.setObjectName("prefab_edit_widget")
        self.lbl_setting_desc = QtWidgets.QLabel(self.frame_current_setting)
        self.lbl_setting_desc.setGeometry(QtCore.QRect(20, 50, 571, 41))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setBold(False)
        font.setWeight(37)
        self.lbl_setting_desc.setFont(font)
        self.lbl_setting_desc.setStyleSheet("color: #888;")
        self.lbl_setting_desc.setWordWrap(True)
        self.lbl_setting_desc.setObjectName("lbl_setting_desc")
        self.btn_insert_point = QtWidgets.QPushButton(self.frame_current_setting)
        self.btn_insert_point.setGeometry(QtCore.QRect(368, 14, 131, 31))
        self.btn_insert_point.setStyleSheet(
            "QPushButton {\n"
            "color: #eee;\n"
            "background: #666;\n"
            "border: 0; \n"
            "}\n"
            "\n"
            "QPushButton:hover {\n"
            "    background: #555;\n"
            "}\n"
            "\n"
            "QPushButton:pressed {\n"
            "    background: #111;\n"
            "}"
        )
        self.btn_insert_point.setObjectName("btn_insert_point")
        self.frame_2 = QtWidgets.QFrame(self.centralwidget)
        self.frame_2.setGeometry(QtCore.QRect(-1, 80, 351, 551))
        self.frame_2.setStyleSheet("background: #eee;")
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.settings_tree = QtWidgets.QTreeWidget(self.frame_2)
        self.settings_tree.setGeometry(QtCore.QRect(14, 20, 311, 511))
        self.settings_tree.setStyleSheet(
            "QTreeWidget::item {\n"
            "padding: 7px 7px 6px;\n"
            "background: #ddd;\n"
            "outline: 0 !important;\n"
            "margin-bottom: 1px;\n"
            "margin-right: 0px;\n"
            "color: #777;\n"
            "border: 0;\n"
            "border-radius: 0;\n"
            "}\n"
            "\n"
            "QTreeWidget::item:hover {\n"
            "background: #ccc;\n"
            "}\n"
            "\n"
            "QTreeWidget::item:selected {\n"
            "background: #555;\n"
            "color: #eee;\n"
            "}\n"
            "\n"
            "QTreeWidget {\n"
            "padding: 0;\n"
            "color: #eee;\n"
            "background: transparent;\n"
            "\n"
            "}\n"
            "\n"
            "* {\n"
            "outline: 0;\n"
            "}\n"
            "\n"
            ""
        )
        self.settings_tree.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.settings_tree.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.settings_tree.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self.settings_tree.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.settings_tree.setProperty("showDropIndicator", False)
        self.settings_tree.setAlternatingRowColors(False)
        self.settings_tree.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.settings_tree.setVerticalScrollMode(QtWidgets.QAbstractItemView.ScrollPerPixel)
        self.settings_tree.setIndentation(18)
        self.settings_tree.setRootIsDecorated(True)
        self.settings_tree.setUniformRowHeights(False)
        self.settings_tree.setItemsExpandable(True)
        self.settings_tree.setAnimated(True)
        self.settings_tree.setAllColumnsShowFocus(False)
        self.settings_tree.setWordWrap(True)
        self.settings_tree.setHeaderHidden(True)
        self.settings_tree.setObjectName("settings_tree")
        item_0 = QtWidgets.QTreeWidgetItem(self.settings_tree)
        item_1 = QtWidgets.QTreeWidgetItem(item_0)
        item_1 = QtWidgets.QTreeWidgetItem(item_0)
        item_1 = QtWidgets.QTreeWidgetItem(item_0)
        item_0 = QtWidgets.QTreeWidgetItem(self.settings_tree)
        item_0.setFlags(
            QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsDragEnabled | QtCore.Qt.ItemIsUserCheckable
            | QtCore.Qt.ItemIsEnabled | QtCore.Qt.ItemIsTristate
        )
        item_1 = QtWidgets.QTreeWidgetItem(item_0)
        self.settings_tree.header().setVisible(False)
        self.settings_tree.header().setCascadingSectionResizes(False)
        self.settings_tree.header().setDefaultSectionSize(150)
        self.settings_tree.header().setStretchLastSection(False)
        self.frame_4 = QtWidgets.QFrame(self.centralwidget)
        self.frame_4.setGeometry(QtCore.QRect(420, 10, 551, 61))
        self.frame_4.setStyleSheet("QFrame {\n" "    background: #fff;\n" "}\n" "")
        self.frame_4.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame_4.setObjectName("frame_4")
        self.time_slider = QtWidgets.QSlider(self.frame_4)
        self.time_slider.setGeometry(QtCore.QRect(10, 11, 521, 20))
        self.time_slider.setMaximum(5184000)
        self.time_slider.setSingleStep(3600)
        self.time_slider.setPageStep(3600)
        self.time_slider.setProperty("value", 2592000)
        self.time_slider.setTracking(True)
        self.time_slider.setOrientation(QtCore.Qt.Horizontal)
        self.time_slider.setInvertedAppearance(False)
        self.time_slider.setInvertedControls(False)
        self.time_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.time_slider.setTickInterval(432000)
        self.time_slider.setObjectName("time_slider")
        self.label_2 = QtWidgets.QLabel(self.frame_4)
        self.label_2.setGeometry(QtCore.QRect(254, 30, 32, 32))
        self.label_2.setText("")
        self.label_2.setPixmap(QtGui.QPixmap(":/res/res/12_00.png"))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.frame_4)
        self.label_3.setGeometry(QtCore.QRect(341, 30, 32, 32))
        self.label_3.setText("")
        self.label_3.setPixmap(QtGui.QPixmap(":/res/res/16_00.png"))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(self.frame_4)
        self.label_4.setGeometry(QtCore.QRect(0, 30, 32, 32))
        self.label_4.setText("")
        self.label_4.setPixmap(QtGui.QPixmap(":/res/res/0_00.png"))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.frame_4)
        self.label_5.setGeometry(QtCore.QRect(84, 30, 32, 32))
        self.label_5.setText("")
        self.label_5.setPixmap(QtGui.QPixmap(":/res/res/4_00.png"))
        self.label_5.setObjectName("label_5")
        self.label_6 = QtWidgets.QLabel(self.frame_4)
        self.label_6.setGeometry(QtCore.QRect(170, 30, 32, 32))
        self.label_6.setText("")
        self.label_6.setPixmap(QtGui.QPixmap(":/res/res/8_00.png"))
        self.label_6.setObjectName("label_6")
        self.label_7 = QtWidgets.QLabel(self.frame_4)
        self.label_7.setGeometry(QtCore.QRect(422, 30, 32, 32))
        self.label_7.setText("")
        self.label_7.setPixmap(QtGui.QPixmap(":/res/res/20_00.png"))
        self.label_7.setObjectName("label_7")
        self.label_8 = QtWidgets.QLabel(self.frame_4)
        self.label_8.setGeometry(QtCore.QRect(510, 30, 32, 32))
        self.label_8.setText("")
        self.label_8.setPixmap(QtGui.QPixmap(":/res/res/0_00.png"))
        self.label_8.setObjectName("label_8")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 35, 261, 31))
        self.label.setStyleSheet("font-size: 20px; color: #2aa877;\n" "font-weight: 100;")
        self.label.setObjectName("label")
        self.time_float_label = QtWidgets.QLabel(self.centralwidget)
        self.time_float_label.setGeometry(QtCore.QRect(350, 47, 41, 18))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(-1)
        font.setBold(False)
        font.setWeight(37)
        self.time_float_label.setFont(font)
        self.time_float_label.setStyleSheet("color: #777; font-size: 12px;")
        self.time_float_label.setObjectName("time_float_label")
        self.time_label = QtWidgets.QLabel(self.centralwidget)
        self.time_label.setGeometry(QtCore.QRect(350, 20, 81, 31))
        font = QtGui.QFont()
        font.setFamily("Roboto")
        font.setPointSize(-1)
        font.setBold(False)
        font.setWeight(37)
        self.time_label.setFont(font)
        self.time_label.setStyleSheet("color: #444; font-size: 22px;")
        self.time_label.setObjectName("time_label")
        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(20, 16, 261, 31))
        self.label_9.setStyleSheet("font-size: 13px;\n" "color: #aaa;\n" "font-weight: 600;")
        self.label_9.setObjectName("label_9")
        self.frame_2.raise_()
        self.frame_current_setting.raise_()
        self.frame_4.raise_()
        self.label.raise_()
        self.time_float_label.raise_()
        self.time_label.raise_()
        self.label_9.raise_()
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Day Time Editor"))
        self.lbl_current_setting.setText(_translate("MainWindow", "Occlusion Strength"))
        self.btn_reset.setText(_translate("MainWindow", "Reset curve"))
        self.lbl_setting_desc.setText(
            _translate(
                "MainWindow",
                "Description: Some Description about the setting, to give the user a rough idea what this does. This description may be so long, it can even go into the next line."
            )
        )
        self.btn_insert_point.setText(_translate("MainWindow", "Insert point from data"))
        self.settings_tree.setSortingEnabled(False)
        self.settings_tree.headerItem().setText(0, _translate("MainWindow", "Setting"))
        self.settings_tree.headerItem().setText(1, _translate("MainWindow", "Value"))
        __sortingEnabled = self.settings_tree.isSortingEnabled()
        self.settings_tree.setSortingEnabled(False)
        self.settings_tree.topLevelItem(0).setText(0, _translate("MainWindow", "Scattering"))
        self.settings_tree.topLevelItem(0).child(0).setText(0, _translate("MainWindow", "Sun Height"))
        self.settings_tree.topLevelItem(0).child(0).setText(1, _translate("MainWindow", "0,4"))
        self.settings_tree.topLevelItem(0).child(1).setText(0, _translate("MainWindow", "Sun Color"))
        self.settings_tree.topLevelItem(0).child(1).setText(1, _translate("MainWindow", "[128, 255, 100]"))
        self.settings_tree.topLevelItem(0).child(2).setText(0, _translate("MainWindow", "Sun Angle"))
        self.settings_tree.topLevelItem(0).child(2).setText(1, _translate("MainWindow", "180 °"))
        self.settings_tree.topLevelItem(1).setText(0, _translate("MainWindow", "Ambient Occlusion"))
        self.settings_tree.topLevelItem(1).child(0).setText(0, _translate("MainWindow", "Occlusion Strength"))
        self.settings_tree.topLevelItem(1).child(0).setText(1, _translate("MainWindow", "0.5"))
        self.settings_tree.setSortingEnabled(__sortingEnabled)
        self.label.setText(_translate("MainWindow", "TIME OF DAY EDITOR"))
        self.time_float_label.setText(_translate("MainWindow", "0.486"))
        self.time_label.setText(_translate("MainWindow", "11:15"))
        self.label_9.setText(_translate("MainWindow", "RENDER PIPELINE"))


from . import resources_rc
