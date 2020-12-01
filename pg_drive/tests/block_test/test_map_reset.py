from pg_drive.scene_creator.algorithm.BIG import BigGenerateMethod
from pg_drive.tests.block_test.test_block_base import TestBlock
from pg_drive.utils.visualization_loader import VisLoader

if __name__ == "__main__":
    """
    Press "c" and "a" to test
    """
    test = TestBlock(True)
    VisLoader.init_loader(test.loader, test.asset_path)
    test.test_reset(BigGenerateMethod.BLOCK_SEQUENCE, "CrTRXOS")
    # test.run()
