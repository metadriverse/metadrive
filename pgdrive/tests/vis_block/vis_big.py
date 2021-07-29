from pgdrive.engine.asset_loader import initialize_asset_loader
from pgdrive.component.algorithm.BIG import BIG
from pgdrive.component.road.road_network import RoadNetwork
from pgdrive.tests.vis_block.vis_block_base import TestBlock
from pgdrive.utils.engine_utils import set_global_random_seed


def vis_big(debug: bool = False, block_type_version="v1", random_seed=None):
    test = TestBlock(debug=debug)

    if block_type_version == "v1":
        random_seed = random_seed or 888
        test.cam.setPos(-200, -350, 2000)
    elif block_type_version == "v2":
        random_seed = random_seed or 333
        test.cam.setPos(300, 400, 2000)

    initialize_asset_loader(test)
    set_global_random_seed(4)
    global_network = RoadNetwork()

    big = BIG(2, 3.5, global_network, test.render, test.world, block_type_version=block_type_version)
    test.vis_big(big)
    test.big.block_num = 40
    # big.generate(BigGenerateMethod.BLOCK_NUM, 10)
    test.run()


if __name__ == "__main__":
    vis_big()
