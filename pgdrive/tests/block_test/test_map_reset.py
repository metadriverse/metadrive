from pgdrive.scene_creator.algorithm.BIG import BigGenerateMethod
from pgdrive.tests.block_test.test_block_base import TestBlock
from pgdrive.utils.asset_loader import AssetLoader

if __name__ == "__main__":
    """
    Press "c" and "a" to test
    """
    test = TestBlock(True)
    AssetLoader.init_loader(test, test.asset_path)
    test.test_reset(BigGenerateMethod.BLOCK_SEQUENCE, "CrTRXOS")
    # test.run()
