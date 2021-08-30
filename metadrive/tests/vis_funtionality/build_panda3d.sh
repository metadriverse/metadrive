#!/bin/bash
# This script is deprecated.
# Kept it here for reference only!
xcode-select --install
git clone https://github.com/panda3d/panda3d.git
cd panda3d

# Download this thing
wget https://www.panda3d.org/download/panda3d-1.10.8/panda3d-1.10.8-tools-mac.tar.gz
# And put the "third-party" folder into "panda3d"

python_libdir="/Users/pengzhenghao/opt/anaconda3/envs/metadrive/lib"
python_incdir="/Users/pengzhenghao/opt/anaconda3/envs/metadrive/include"
ln -s ${python_libdir}/libpython3.7m.dylib ${python_libdir}/libpython3.7.dylib
python ./makepanda/makepanda.py --everything --no-opencv --no-fmod --no-eigen --wheel --thread 8 \
  --python-incdir ${python_incdir} \
  --python-libdir ${python_libdir}
