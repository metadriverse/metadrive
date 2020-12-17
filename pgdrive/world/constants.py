from panda3d.core import Vec4

from pgdrive.pg_config.body_name import BodyName

pg_edition = "PGDrive v0.1.0"

help_message = "Keyboard Shortcuts:\n" \
               "  W: Acceleration\n" \
               "  S: Braking\n" \
               "  A: Moving Left\n" \
               "  D: Moving Right\n" \
               "  R: Reset the Environment\n" \
               "  H: Help Message\n" \
               "  1: Box Debug Mode\n" \
               "  2: WireFrame Debug Mode\n" \
               "  3: Texture Debug Mode\n" \
               "  4: Print Debug Message\n" \
               "  F: Switch FPS between unlimited, 60 Hz and \n" \
               "     real simulation frequency\n" \
               "  Esc: Quit\n"

# priority and color
COLLISION_INFO_COLOR = dict(
    red=(0, Vec4(195 / 255, 0, 0, 1)),
    orange=(1, Vec4(218 / 255, 80 / 255, 0, 1)),
    yellow=(2, Vec4(218 / 255, 163 / 255, 0, 1)),
    green=(3, Vec4(65 / 255, 163 / 255, 0, 1))
)
COLOR = {
    BodyName.Side_walk: "red",
    BodyName.Continuous_line: "orange",
    BodyName.Stripped_line: "yellow",
    BodyName.Traffic_vehicle: "red"
}
