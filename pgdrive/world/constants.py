from panda3d.core import Vec4

from pgdrive.pg_config.body_name import BodyName

PG_EDITION = "PGDrive v0.1.0"

HELP_MESSAGE = "Keyboard Shortcuts:\n" \
               "  W: Acceleration\n" \
               "  S: Braking\n" \
               "  A: Moving Left\n" \
               "  D: Moving Right\n" \
               "  R: Reset the Environment\n" \
               "  H: Help Message\n" \
               "  F: Switch FPS between unlimited and realtime\n" \
               "  Esc: Quit\n"

DEBUG_MESSAGE = "  1: Box Debug Mode\n" \
                "  2: WireFrame Debug Mode\n" \
                "  3: Texture Debug Mode\n" \
                "  4: Print Node Message\n"

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

RENDER_MODE_NONE = "none"  # Do not render
RENDER_MODE_ONSCREEN = "onscreen"  # Pop up a window and draw image in it
RENDER_MODE_OFFSCREEN = "offscreen"  # Draw image in buffer and collect image from memory

# Deprecated!
# RENDER_MODE_TOPDOWN = "topdown"  # Use pygame to draw the top down view of
