"""
This script aims to convert nuplan scenarios to ScenarioDescription, so that we can load any nuplan scenarios into
MetaDrive.
"""
import tqdm

from metadrive.utils.nuscenes_utils.utils import convert_one_scene

try:
    from nuscenes import NuScenes
except ImportError:
    print("Can not find nuscenes-devkit")

if __name__ == "__main__":

    version = 'v1.0-mini'
    verbose = True
    dataroot = '/home/shady/data/nuscenes'
    # Init.
    nusc = NuScenes(version=version, verbose=verbose, dataroot=dataroot)
    scenes = nusc.scene
    ret = {}
    for scene in tqdm.tqdm(scenes):
        ret[scene["token"]] = convert_one_scene(scene["token"], nusc)
        # TODO sanity_check()
