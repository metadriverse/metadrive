# This script is deprecated.
# Kept it here for reference only!

git clone https://github.com/panda3d/panda3d.git
cd panda3d
python_libdir="/Users/pengzhenghao/opt/anaconda3/envs/pgdrive/lib"
ln -s ${python_libdir}/libpython3.7m.dylib ${python_libdir}/libpython3.7.dylib
python ./makepanda/makepanda.py --everything --no-opencv --no-fmod --no-eigen --wheel --thread 8 \
  --python-incdir /Users/pengzhenghao/opt/anaconda3/envs/pgdrive/include \
  --python-libdir /Users/pengzhenghao/opt/anaconda3/envs/pgdrive/lib
