from metadrive.engine.asset_loader import initialize_asset_loader, AssetLoader
from metadrive.tests.vis_block.vis_block_base import TestBlock
from metadrive.utils.vertex import add_class_label


def _test_add_class():
    test = TestBlock(window_type="none")
    initialize_asset_loader(test)
    model = test.loader.loadModel(AssetLoader.file_path("models", "lada", "vehicle.gltf"))
    add_class_label(model, 1)

    # processGeomNode(geomNode)

    model.reparentTo(test.render)
    test.taskMgr.step()
    test.taskMgr.step()
    test.taskMgr.step()
