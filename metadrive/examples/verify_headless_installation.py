from panda3d.core import loadPrcFileData

from metadrive.tests.vis_functionality.vis_installation import capture_headless_image

if __name__ == "__main__":
    loadPrcFileData("", "notify-level-task fatal")
    capture_headless_image(headless=True)
