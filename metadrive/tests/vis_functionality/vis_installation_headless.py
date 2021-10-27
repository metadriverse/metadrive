from metadrive.tests.vis_functionality.vis_installation import vis_installation
from panda3d.core import loadPrcFileData

if __name__ == "__main__":
    loadPrcFileData("", "notify-level-task fatal")
    vis_installation(headless=True)
