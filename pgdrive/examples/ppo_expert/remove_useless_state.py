"""
This script is used to remove the optimizer state in the checkpoint. So that we can compress 2/3 of the checkpoint size.
This script is put here for reference only. In formal release, the original checkpoint file will be removed so
this script will become not runnable.
"""
import os.path as osp
import pickle

import numpy as np

ckpt_path = osp.join(osp.dirname(__file__), "checkpoint-compressed")
if __name__ == '__main__':
    with open(ckpt_path, "rb") as f:
        data = f.read()
    unpickled = pickle.loads(data)
    worker = pickle.loads(unpickled.pop("worker"))
    if "_optimizer_variables" in worker["state"]["default_policy"]:
        worker["state"]["default_policy"].pop("_optimizer_variables")
    pickled_worker = pickle.dumps(worker)
    weights = worker["state"]["default_policy"]
    weights = {k: v for k, v in weights.items() if "value" not in k}
    np.savez_compressed("expert_weights.npz", **weights)
