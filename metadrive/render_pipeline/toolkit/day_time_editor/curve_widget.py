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

from __future__ import print_function
from rplibs.six.moves import range  # pylint: disable=import-error

import math

from rplibs.pyqt_imports import *  # noqa


class CurveWidget(QWidget):
    """ This is a resizeable Widget which shows an editable curve which can
    be modified. """
    def __init__(self, parent):
        """ Constructs the CurveWidget, we start with an initial curve """
        QWidget.__init__(self, parent)
        self.setFocusPolicy(Qt.ClickFocus)
        self._curves = []

        # Store current display time
        self._current_time = 0.5

        # Widget render constants
        self._cv_point_size = 3
        self._legend_border = 52
        self._bar_h = 30

        # Currently dragged control point, format is:
        # (CurveIndex, PointIndex, Drag-Offset (x,y))
        self._drag_point = None
        self._drag_time = 0.0

        # Currently selected control point, format is:
        # (CurveIndex, PointIndex)
        self._selected_point = None

        self._unit_processor = lambda v: str(round(v, 2))
        self._change_handler = lambda: None

    def set_unit_processor(self, proc):
        """ Sets the function which gets called to map values from 0 .. 1 to
        values like 10% to 30% """
        self._unit_processor = proc

    def set_change_handler(self, handler):
        """ Sets a function which gets called when the data in this widget got
        edited """
        self._change_handler = handler

    def paintEvent(self, e):  # noqa
        """ Internal QT paint event, draws the entire widget """
        qp = QPainter()
        qp.begin(self)
        self._draw(qp)
        qp.end()

    def mousePressEvent(self, QMouseEvent):  # noqa
        """ Internal mouse-press handler """
        self._drag_point = None
        self._selected_point = None
        mouse_pos = QMouseEvent.pos()
        mouse_x = mouse_pos.x() - self._legend_border
        mouse_y = mouse_pos.y() - self._bar_h

        for index, curve in enumerate(self._curves):

            # Check for clicks on control points
            for cv_index, (x, y) in enumerate(curve.control_points):
                point_x = self._get_x_value_for(x)
                point_y = self._get_y_value_for(y) - self._bar_h
                if abs(point_x - mouse_x) < self._cv_point_size + 6:
                    if (abs(point_y - mouse_y)) < self._cv_point_size + 6:
                        drag_x_offset = mouse_x - point_x
                        drag_y_offset = mouse_y - point_y
                        mpos_relx = float(mouse_x) / (self.width() - self._legend_border)
                        self._drag_point = (index, cv_index, (drag_x_offset, drag_y_offset))
                        self._drag_time = mpos_relx
                        self._selected_point = (index, cv_index)

            # If still no intersection, check if we clicked a curve
            if mouse_x > 0 and mouse_x < self.width() - self._legend_border:
                if mouse_y > 0 and mouse_y < self.height() - self._legend_border:
                    mpos_relx = float(mouse_x) / (self.width() - self._legend_border)
                    curve_py = curve.get_value(mpos_relx)
                    curve_offy = self._get_y_value_for(curve_py) - self._bar_h

                    if abs(curve_offy - mouse_y) < 8 and self._selected_point is None:
                        # Clicked on curve, spawn new point
                        cv_index = curve.append_cv(mpos_relx, curve_py)

                        self._selected_point = (index, cv_index)
                        self._drag_point = (index, cv_index, (0, 0))
                        self._drag_time = mpos_relx
                        self._change_handler()
        self.update()

    def mouseReleaseEvent(self, event):  # noqa
        """ Internal mouse-release handler """
        self._drag_point = None
        self._drag_time = -1
        self.update()

    def mouseMoveEvent(self, event):  # noqa
        """ Internal mouse-move handler """
        if len(self._curves) < 1:
            return

        if self._drag_point is not None:
            mouse_x = event.pos().x() - self._drag_point[2][0] - self._legend_border
            mouse_y = event.pos().y() - self._drag_point[2][1] - self._bar_h

            # Convert to local coordinate
            local_x = max(0, min(1, mouse_x / float(self.width() - self._legend_border)))
            local_y = mouse_y / float(self.height() - self._legend_border - self._bar_h)
            local_y = 1 - max(0, min(1, local_y))

            self._drag_time = local_x

            # Set new point data
            self._curves[self._drag_point[0]].set_cv_value(self._drag_point[1], local_x, local_y)

            # Redraw curve
            self._curves[self._drag_point[0]].build_curve()
            self.update()
            self._change_handler()

    def keyPressEvent(self, event):  # noqa
        """ Internal keypress handler """
        # Delete anchor point
        if event.key() == Qt.Key_Delete:
            self.delete_current_point()

    def delete_current_point(self):
        """ Deletes the currently selected point """
        if self._selected_point is not None:
            self._curves[self._selected_point[0]].remove_cv(self._selected_point[1])
            self._selected_point = None
            self._drag_point = None
            self.update()
            self._change_handler()

    def set_curves(self, curves):
        """ Sets the list of displayed curves """
        self._selected_point = None
        self._drag_point = None
        self._curves = curves
        self.update()

    def _get_y_value_for(self, local_value):
        """ Converts a value from 0 to 1 to a value from 0 .. canvas height """
        local_value = max(0, min(1.0, 1.0 - local_value))
        local_value *= self.height() - self._legend_border - self._bar_h
        local_value += self._bar_h
        return local_value

    def _get_x_value_for(self, local_value):
        """ Converts a value from 0 to 1 to a value from 0 .. canvas width """
        local_value = max(0, min(1.0, local_value))
        local_value *= self.width() - self._legend_border
        return local_value

    def set_current_time(self, local_time):
        """ Sets the current displayed time, should range from 0 to 1 """
        self._current_time = max(0.0, min(1.0, local_time))
        self.update()

    def _draw(self, painter):
        """ Internal method to draw the widget """

        canvas_width = self.width() - self._legend_border
        canvas_height = self.height() - self._legend_border - self._bar_h

        # Draw field background
        # painter.setPen(QColor(200, 200, 200))
        # painter.setBrush(QColor(230, 230, 230))
        # painter.drawRect(0, 0, self.width() - 1, self.height() - 1)

        # Draw legend

        # Compute amount of horizontal / vertical lines
        num_vert_lines = 12  # 24 / 12 = 2, one entry per 2 hours
        line_spacing_x = (self.width() - self._legend_border) / num_vert_lines
        line_spacing_y = (self.height() - self._legend_border - self._bar_h) / 20.0
        num_horiz_lines = int(math.ceil(canvas_height / float(line_spacing_y)) + 1)

        # Draw vertical lines
        painter.setPen(QColor(200, 200, 200))
        for i in range(num_vert_lines + 1):
            line_pos = i * line_spacing_x + self._legend_border - 1
            painter.drawLine(line_pos, self._bar_h, line_pos, canvas_height + self._bar_h)

        # Draw horizontal lines
        painter.setPen(QColor(200, 200, 200))
        for i in range(num_horiz_lines):
            line_pos = canvas_height - i * line_spacing_y + self._bar_h
            painter.drawLine(self._legend_border, line_pos, self.width(), line_pos)

        # Draw vetical legend labels
        painter.setPen(QColor(120, 120, 120))
        for i in range(num_horiz_lines):
            line_pos = canvas_height - i * line_spacing_y + self._bar_h
            # painter.drawText(6, line_pos + 3, str(round(float(i) / (num_horiz_lines-1), 2)))
            painter.drawText(6, line_pos + 3, self._unit_processor(float(i) / (num_horiz_lines - 1)))

        # Draw horizontal legend labels
        for i in range(num_vert_lines + 1):
            line_pos = i * line_spacing_x + self._legend_border
            offpos_x = -14
            if i == 0:
                offpos_x = -2
            elif i == num_vert_lines:
                offpos_x = -27
            time_string = str(int(float(i) / num_vert_lines * 24)).zfill(2) + ":00"
            painter.drawText(line_pos + offpos_x, canvas_height + self._bar_h + 18, time_string)

        # Draw curve
        for index, curve in enumerate(self._curves):
            painter.setPen(QColor(*curve.color))
            last_value = 0
            for i in range(canvas_width):
                rel_offset = i / (canvas_width - 1.0)
                curve_height = self._get_y_value_for(curve.get_value(rel_offset))

                if i == 0:
                    last_value = curve_height

                painter.drawLine(self._legend_border + i - 1, last_value, self._legend_border + i, curve_height)
                last_value = curve_height

            # Draw the CV points of the curve
            painter.setBrush(QColor(240, 240, 240))

            for cv_index, (x, y) in enumerate(curve.control_points):
                offs_x = x * canvas_width + self._legend_border
                offs_y = (1 - y) * canvas_height + self._bar_h

                if (self._selected_point and self._selected_point[0] == index and self._selected_point[1] == cv_index):
                    painter.setPen(QColor(255, 0, 0))
                else:
                    painter.setPen(QColor(100, 100, 100))
                painter.drawRect(
                    offs_x - self._cv_point_size, offs_y - self._cv_point_size, 2 * self._cv_point_size,
                    2 * self._cv_point_size
                )

        # Draw bar background
        bar_half_height = 4
        bar_top_pos = 10

        painter.setBrush(QColor(255, 0, 0))
        painter.setPen(QColor(110, 110, 110))

        painter.drawRect(
            self._legend_border - 1, bar_top_pos - 1,
            self.width() - self._legend_border, 2 * bar_half_height + 2
        )

        # Draw bar
        if len(self._curves) == 0:
            return

        if len(self._curves) == 1:
            bar_curve = self._curves[0]
        else:
            bar_curve = self._curves[0:3]

        for i in range(canvas_width - 1):
            xpos = self._legend_border + i
            relv = float(i) / float(canvas_width)

            if len(self._curves) == 1:
                val = max(0, min(255, int(bar_curve.get_value(relv) * 255.0)))
                painter.setPen(QColor(val, val, val))
            else:
                r = max(0, min(255, int(bar_curve[0].get_value(relv) * 255.0)))
                g = max(0, min(255, int(bar_curve[1].get_value(relv) * 255.0)))
                b = max(0, min(255, int(bar_curve[2].get_value(relv) * 255.0)))
                painter.setPen(QColor(r, g, b))
            painter.drawLine(xpos, bar_top_pos, xpos, bar_top_pos + 2 * bar_half_height)

        # Draw selected time
        if self._drag_point:
            painter.setBrush(QColor(200, 200, 200))
            painter.setPen(QColor(90, 90, 90))
            offs_x = max(0, min(canvas_width + 10, self._drag_time * canvas_width + self._legend_border - 19))
            offs_y = self.height() - self._legend_border
            minutes = int(self._drag_time * 24 * 60)
            hours = minutes / 60
            minutes = minutes % 60
            painter.drawRect(offs_x, self.height() - self._legend_border + 5, 40, 20)
            painter.drawText(offs_x + 7, offs_y + 20, "{:02}:{:02}".format(hours, minutes))

            painter.setPen(QColor(150, 150, 150))
            painter.drawLine(offs_x + 19, bar_top_pos + 15, offs_x + 19, self.height() - self._legend_border + 5)

        # Display current time
        pen = QPen()
        pen.setColor(QColor(255, 100, 100))
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)

        xoffs = self._legend_border + self._current_time * (canvas_width - 1)
        painter.drawLine(xoffs, self._bar_h, xoffs, self._bar_h + canvas_height)

        # Draw usage hints
        painter.setPen(QColor(100, 100, 100))
        painter.drawText(
            5,
            self.height() - 2, "Click on the curve to add new control points, click and drag "
            "existing points to move them."
        )
