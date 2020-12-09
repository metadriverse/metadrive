from direct.showbase import OnScreenDebug
from panda3d.core import Vec4, TextNode


class PgOnScreenMessage(OnScreenDebug.OnScreenDebug):
    """
    Simply inherit from the original debug class of panda3d
    """
    POS = (0.1, -0.2)

    def __init__(self):
        super(PgOnScreenMessage, self).__init__()
        self.enabled = True
        self.load()

    def update_data(self, data: dict):
        self.onScreenText.cleanup()
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
            v_text = "%-100s\n" % (k.strip() + isNew + str(value))
            self.onScreenText.appendText(v_text)
        self.frame += 1
