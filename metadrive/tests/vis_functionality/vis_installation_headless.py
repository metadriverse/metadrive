from panda3d.core import loadPrcFileData

from metadrive.tests.test_installation import verify_installation

if __name__ == "__main__":
    loadPrcFileData("", "notify-level-task fatal")
    verify_installation()
