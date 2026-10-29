#!/usr/bin/env python
from panda3d.core import loadPrcFileData
from metadrive.tests.test_installation import verify_installation
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--camera", type=str, default="main", choices=["main", "rgb", "depth"])
    args = parser.parse_args()
    loadPrcFileData("", "notify-level-task fatal")
    verify_installation(args.cuda, args.camera)
