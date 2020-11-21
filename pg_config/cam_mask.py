from panda3d.core import BitMask32


class CamMask:
    MainCam = BitMask32.bit(9)
    Shadow = BitMask32.bit(10)
    FrontCam = BitMask32.bit(11)
    MiniMap = BitMask32.bit(12)
    PARA_VIS = BitMask32.bit(13)
