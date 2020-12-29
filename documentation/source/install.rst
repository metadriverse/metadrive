.. _install:

######################
Installing PGDrive
######################

By leveraging the power of panda3d, PGDrive can be run on personal laptop, cluster, headless server with different OS.

Install PGDrive on macOS, Windows and Linux in the easiest way
###############################################################

The installation procedure on these different platforms is same and easy, we recommend to use the command following to install::

    pip install git+https://github.com/decisionforce/pgdrive.git

or you can install via::

    git clone https://github.com/decisionforce/pgdrive.git
    cd pgdrive
    pip install -e .

The basic functionality, namely the render-less simulation can be conducted extremely easily. However, if you wish to
use rendering features such as the RGB, the installation need more efforts, especially in headless machine or cluster.

Verify the installation of PGDrive
#############################################

To check whether PGDrive v0.1.1 is successfully installed, please run::

    python -m pgdrive.examples.profile_pgdrive



You can also verify the efficiency of PGDrive through the printed messages.
Note that the above script is supposed to be runnable in all places.
Please do not run the above command in the folder that has a sub-folder called :code:`./pgdrive`.

Install the PGDrive with offscreen rendering functionality
##############################################################

This section introduce the procedure to enable PGDrive with RGB rendering in headless machine.
If the lidar information is enough for your task, you can simply install PGDrive on your headless machine using the way we mentioned above.

.. note:: You have to set the :code:`config["pg_world_config"]["headless_image"] = True` when training the agent using image as input.

However, if you want to use image to train your agent on headless machine, you have to compile a customized Panda3D.
The customized Panda3D is built from the source code of panda3d, following the instructions in `Panda3D: Building Panda3D <https://github.com/panda3d/panda3d#building-panda3d>`_.
After setting up the Panda3D dependencies, we will build our own wheel through the following command::

    python ./makepanda/makepanda.py --everything --no-x11 --no-opencv --no-fmodex \
      --python-incdir /path/to/your/conda_env/include/ \
      --python-libdir /path/to/your/conda_env/lib/ \
      --thread 8 --wheel

It will give you a Panda3D wheel which can run in EGL environment without the X11 support. Now please install the wheel file by::

    pip install panda3d-1.10.xxx.whl

Now, PGDrive will utilize the power of cluster to train your agent!

.. warning:: Compiling Panda3D from source might require the **administrator permission** to install some libraries.
    We are working to provide a pre-built Panda3D for cluster users of PGDrive to make it easy to use on headless machines.

Verify the offscreen rendering functionality of PGDrive
############################################################

.. note:: An easy installation of PGDrive in macOS will fail the following verification.

Please run commands below to verify the installation::

    python -m pgdrive.tests.install_test.test_install

Successfully running this script means the PGDrive works well, and an image will be shown to help you check if PGDrive
can launch and capture image in offscreen mode

To verify the installation on cluster, run following command instead::

    python -m pgdrive.tests.install_test.test_headless_install

The script will generate images to local directory. Please fetch anc check those images from cluster to ensure PGDrive can draw scene
and capture images without X11.
