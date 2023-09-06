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

Install MetaDrive with headless rendering
############################################

The latest MetaDrive is already built to support headless-rendering. But for a double check, run following command::

    python -m metadrive.examples.verify_headless_installation

The script will generate two **same** images to current directory, one from agent observation, the other from panda3d internal rendering buffer.
Please fetch anc check those images from cluster to ensure MetaDrive can draw scene and capture images correctly.
By default, it only generates images from the main camera. Set ```--camera [rgb/depth]``` to check other cameras.
Also, ```--cuda``` flag can be added to test image_on_cuda pipeline for your headless machine.

If the captured images is complete as following, then the installation in headless machine is successful and please move on to :ref:`use_native_rendering`.




.. _install_render_cuda:

Install MetaDrive with advanced offscreen rendering
#####################################################
The default observation contains information on ego vehicle's states, Lidar-like cloud points showing neighboring vehicles, and information about navigation and tasks. Besides, you can also try the Pygame-based Top-down rendering (See :ref:`use_pygame_rendering`), which is also runnable in most headless machine without any special treatment.


If the above observation is not enough for your RL algorithms and you wish to use the Panda3D camera to provide realistic RGB images as the observation, please continue reading this section.

Generally, the default installation method supports getting rendered image. In this case, images will be returned as numpy array, which is retrieved from GPU and costs latency. We provide an advanced function to allow accessing images on GPU directly,
so that you can read them by **Torch** or **Tensorflow**. With such a treatment, the image-based data sampling can be **10x** faster! See: https://github.com/metadriverse/metadrive/issues/306

Requirements:

* CUDA Runtime >= 12.0
* Windows or Linux

Installation:

#. After cloning the repo, use ``pip install -e .[cuda]`` to install, or ``pip install -e metadrive-simulator[cuda]`` if you are using pip.
#. Install Torch: ``conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge``
#. Install CuPy: ``pip install cupy-cuda11x``
#. Install Cuda-Python: ``conda install -c nvidia cuda-python``
#. For verifying your installation, cd ``metadrive/examples`` and run ``python verify_image_observation.py --cuda``


After running the script, if no error messages, then congratulations! It works. you can also use ``python verify_image_observation.py --render`` to visualize the image observations.
Besides, removing ``--cuda`` flag enables benchmarking the original image collection pipeline as a comparison.
And ``--camera`` argument is for choosing sensors from ["rgb", "depth, "semantic", "main" (default)].

