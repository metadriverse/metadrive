from panda3d.core import loadPrcFileData

from pgdrive.tests.install_test.test_install import test_install

if __name__ == "__main__":
    loadPrcFileData("", "notify-level-task fatal")
    test_install(True)
