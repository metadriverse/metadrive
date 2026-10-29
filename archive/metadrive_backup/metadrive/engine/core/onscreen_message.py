from typing import Optional, Union

from direct.showbase import OnScreenDebug
from panda3d.core import Vec4

from metadrive.constants import HELP_MESSAGE, DEBUG_MESSAGE


class ScreenMessage(OnScreenDebug.OnScreenDebug):
    """
    Simply inherit from the original debug class of panda3d to show debug message on screen
    """
    POS = (0.1, -0.2)
    SCALE = None

    def __init__(self, refresh_plain_text=False, debug=False):
        super(ScreenMessage, self).__init__()
        self.debug = debug
        self.enabled = True
        self.load()
        self.plain_text = set()
        self._refresh_plain_text = refresh_plain_text
        self._show_help_message = False

    def _update_data(self, data: Optional[Union[dict, str]]):
        # Several Node will be destructing or constructing again and again when debug pgraph
        self.onScreenText.cleanup()
        if isinstance(data, str):
            self.clear_all_plain_text()
            self.plain_text.add(data)
        elif isinstance(data, dict):
            for k, v in data.items():
                self.add(k, v)

    def load(self):
        super(ScreenMessage, self).load()
        self.onScreenText.setBg(Vec4(0, 0, 0, 0.5))
        self.onScreenText.setPos(*self.POS)
        self.onScreenText.textNode.setCardAsMargin(0.6, 0.6, 0.5, 0.1)
        if self.SCALE is not None:
            self.onScreenText.setScale(self.SCALE)

    def set_scale(self, scale=None):
        self.SCALE = scale

    def render(self, data: Optional[Union[dict, str]] = None):
        self._update_data(data)
        if not self.enabled:
            return
        if not self.onScreenText:
            self.load()
        self.onScreenText.clearText()
        if self._show_help_message:
            hlp_msg = HELP_MESSAGE + DEBUG_MESSAGE if self.debug else HELP_MESSAGE
            self.onScreenText.appendText(hlp_msg)
            return

        # Render plain text first
        for v in self.plain_text:
            self.onScreenText.appendText(v)

        # Render numerical values
        entries = list(self.data.items())
        entries.sort()
        for k, v in entries:
            if v[0] == self.frame:
                # It was updated this frame (key equals value):
                # isNew = " is"
                isNew = ":"
            else:
                # This data is not for the current
                # frame (key roughly equals value):
                # isNew = "was"
                isNew = ":"
            value = v[1]
            if type(value) == float:
                value = "% 10.4f" % (value, )
            # else: other types will be converted to str by the "%s"
            if type(value) == str:
                value = value.strip()
            if k:
                v_text = "%-100s\n" % (k.strip() + isNew + str(value))
            else:
                v_text = "{}\n".format(str(value))
            self.onScreenText.appendText(v_text)

        self.frame += 1

    def clear_plain_text(self, string):
        if string in self.plain_text:
            self.plain_text.remove(string)

    def clear_all_plain_text(self):
        self.plain_text.clear()

    def toggle_help_message(self):
        self.clear_all_plain_text()
        if self._show_help_message:
            self._show_help_message = False
        else:
            self._show_help_message = True
