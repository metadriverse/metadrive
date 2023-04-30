# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'point_insert.ui'
#
# Created by: PyQt5 UI code generator 5.7
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(342, 197)
        Dialog.setMinimumSize(QtCore.QSize(342, 197))
        Dialog.setMaximumSize(QtCore.QSize(342, 197))
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(10, 10, 301, 71))
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(Dialog)
        self.label_2.setGeometry(QtCore.QRect(10, 100, 46, 21))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(Dialog)
        self.label_3.setGeometry(QtCore.QRect(10, 130, 46, 21))
        self.label_3.setObjectName("label_3")
        self.ipt_value = QtWidgets.QDoubleSpinBox(Dialog)
        self.ipt_value.setGeometry(QtCore.QRect(70, 130, 121, 22))
        self.ipt_value.setMaximum(99999.99)
        self.ipt_value.setObjectName("ipt_value")
        self.ipt_time = QtWidgets.QTimeEdit(Dialog)
        self.ipt_time.setGeometry(QtCore.QRect(70, 100, 121, 22))
        self.ipt_time.setObjectName("ipt_time")
        self.btn_insert = QtWidgets.QPushButton(Dialog)
        self.btn_insert.setGeometry(QtCore.QRect(240, 160, 75, 23))
        self.btn_insert.setFocusPolicy(QtCore.Qt.TabFocus)
        self.btn_insert.setObjectName("btn_insert")

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        Dialog.setTabOrder(self.ipt_time, self.ipt_value)
        Dialog.setTabOrder(self.ipt_value, self.btn_insert)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Insert Point"))
        self.label.setText(
            _translate(
                "Dialog", "Enter the values of the point you want to insert. \n"
                "\n"
                "Did you know: If you dont want to enter concrete values, just click and drag anywhere on the curve to insert a new point."
            )
        )
        self.label_2.setText(_translate("Dialog", "Time:"))
        self.label_3.setText(_translate("Dialog", "Value:"))
        self.ipt_time.setDisplayFormat(_translate("Dialog", "HH:mm"))
        self.btn_insert.setText(_translate("Dialog", "Insert"))


from . import resources_rc
