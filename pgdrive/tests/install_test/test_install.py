from pgdrive.tests.install_test.get_image import test_install
from panda3d.core import loadPrcFileData

loadPrcFileData("", "notify-level-task fatal")

if __name__ == "__main__":
    test_install(False)
