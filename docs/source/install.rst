.. _install:

######################
Installing PGDrive
######################

By leveraging the power of panda3d, PGDrive can be run on personal laptop, cluster, headless server with different OS.

.. note:: There are still some problem with the visualization models, causing slight difference on different platforms.
    We still work in progress to solve this.

Install PGDrive on MacOs, Windows and Linux
###############################################

The installation procedure on these different platforms is same and easy, we recommend to use the command following to install::

    pip install git+https://github.com/decisionforce/pgdrive.git

or you can install via::

    git clone https://github.com/decisionforce/pgdrive.git

    cd pgdrive

    pip install -e .

Install PGDrive on headless machine or cluster
#################################################
If lidar information is enough for your agent, you can also install PGDrive on your headless machine as same as we mentioned above.
However, if you want to use image to train your agent on headless machine, you have to compile from the source code of panda3d.
Follow the instructions on the main page of `panda3d <https://github.com/panda3d/panda3d>`_, and then use the command following to compile panda3d::

    python ./makepanda/makepanda.py --everything --no-x11 --no-opencv --no-fmodex --python-incdir /path/to/your/conda_env/include/ --python-libdir /path/to/your/conda_env/lib/ --thread 8 --wheel


It will give you a panda3d which can run in EGL environment without the X11 support.
Install the wheel file by pip install panda3d-1.10.xxx.whl, and utilize the power of cluster to train your agent!

.. note:: The boolean in "pg_world_config" named "headless_rgb" must be set to True, when training the agent of image input.

Verify Installation
#########################
Run commands below to verify the installation::

    python -m pgdrive.tests.install_test.test_no_image

or::

    python -m pgdrive.tests.install_test.test_get_image

Successfully running the first script means the PGDrive physics world works well.
And the second script will generate *three* images under offscreen mode, by which you can check the if the scene is drawn correctly.

To verify the installation on cluster, run following command instead::

    python -m pgdrive.tests.install_test.test_get_image_headless

Please, fetch the images on cluster and check the images generated on the headless machine.