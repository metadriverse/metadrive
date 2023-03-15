from panda3d.core import loadPrcFileData

from metadrive.tests.vis_functionality.vis_installation import verify_installation

if __name__ == "__main__":
    loadPrcFileData("", "notify-level-task fatal")
    verify_installation(headless=True)
