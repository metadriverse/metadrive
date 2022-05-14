.. _install:

######################
Installing MetaDrive
######################


Install MetaDrive by few lines!
############################################

The installation of MetaDrive on different platforms is straightforward and easy!

We recommend to use the command following to install::

    git clone https://github.com/metadriverse/metadrive.git
    cd metadrive
    pip install -e .


To check whether MetaDrive is successfully installed, please run::

    python -m metadrive.examples.profile_metadrive


You can also verify the efficiency of MetaDrive through the printed messages. This script is supposed to be runnable in all places.

.. note:: Please do not run the above command in the folder that has a sub-folder called :code:`./metadrive`.


.. _install_headless:

Install MetaDrive with offscreen rendering
############################################



The default observation contains information on ego vehicle's states, Lidar-like cloud points showing neighboring vehicles, and information about navigation and tasks. Besides, you can also try the Pygame-based Top-down rendering (See :ref:`use_pygame_rendering`), which is also runnable in most headless machine without any special treatment.


If the above observation is not enough for your RL algorithms and you wish to use the Panda3D camera to provide realistic RGB images as the observation, please continue reading this section.


If your machine already has a screen, please try the following script to verify whether the Panda3D window can successfully pop up.

    python -m metadrive.examples.drive_in_single_agent_env

.. note:: Please do not run the above command in the folder that has a sub-folder called :code:`./metadrive`.

If the screen successfully shows up, then you can move on to :ref:`use_native_rendering` and skip this section.


However, if you want to use image to train your agent on headless machine, you have to compile a customized Panda3D.
The customized Panda3D is built from the source code of panda3d following the instructions in `Panda3D: Building Panda3D <https://github.com/panda3d/panda3d#building-panda3d>`_. Please refer to the link to setup Panda3D dependencies. After setting up dependencies, we build our own wheel through the following command::

    python ./makepanda/makepanda.py --everything --no-x11 --no-opencv --no-fmodex \
      --python-incdir /path/to/your/conda_env/include/ \
      --python-libdir /path/to/your/conda_env/lib/ \
      --thread 8 --wheel


It will give you a Panda3D wheel which can run in EGL environment without the X11 support. Now please install the wheel file by::

    pip install panda3d-1.10.xxx.whl


In principle, the installation of MetaDrive in headless machine is finished.
To verify the installation on cluster, run following command instead::

    python -m metadrive.tests.test_headless


The script will generate images to current directory. Please fetch anc check those images from cluster to ensure MetaDrive can draw scene and capture images.

If the captured images is complete as following, then the installation in headless machine is successful and please move on to :ref:`use_native_rendering`.

.. note:: You have to set the :code:`config["headless_machine_render"] = True` when training the agent using images as observation.

.. warning:: Compiling Panda3D from source might require the **administrator permission** to install some libraries.
    We are working to provide a pre-built Panda3D for cluster users of MetaDrive to make it easy to use on headless machines.


