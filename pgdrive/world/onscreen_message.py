from typing import Optional, Union

from direct.showbase import OnScreenDebug
from panda3d.core import Vec4


class PgOnScreenMessage(OnScreenDebug.OnScreenDebug):
    """
    Simply inherit from the original debug class of panda3d
    """
    POS = (0.1, -0.2)

    def __init__(self):
        super(PgOnScreenMessage, self).__init__()
        self.enabled = True
        self.load()
        self.plain_text = set()

    def update_data(self, data: Optional[Union[dict, str]]):
        self.onScreenText.cleanup()
        if isinstance(data, str):
            self.plain_text.add(data)
        elif isinstance(data, dict):
            for k, v in data.items():
                self.add(k, v)

    def load(self):
        super(PgOnScreenMessage, self).load()
        self.onScreenText.setBg(Vec4(0, 0, 0, 0.5))
        self.onScreenText.setPos(*self.POS)
        self.onScreenText.textNode.setCardAsMargin(0.6, 0.6, 0.5, 0.1)

    def render(self):
        if not self.enabled:
            return
        if not self.onScreenText:
            self.load()
        self.onScreenText.clearText()

        print('Fucking you: plan test:', self.plain_text)

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
                isNew = "="
            else:
                # This data is not for the current
                # frame (key roughly equals value):
                # isNew = "was"
                isNew = "~"
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
        self.plain_text.remove(string)
