######################
MetaDrive Documentation
######################

.. image:: ../../metadrive/assets/MetaDrive.png
   :width: 300
   :align: center 

Welcome to the MetaDrive documentation. MetaDrive is an open-ended driving simulator with infinite scenes.
The key features of MetaDrive includes:

- **Lightweight**: Extremely easy to download, install and run in almost all platform.
- **Realistic**: Accurate physics simulation and multiple sensory input including RGB camera, Lidar and sensory data.
- **Efficient**: Up to 500 simulation step per second.
- **Open-ended**: Support generating infinite scenes and configuring various traffic, vehicle, and environmental settings.

This documentation let you get familiar with the installation and basic utilization of MetaDrive.
Please go through :doc:`install` to install MetaDrive and try the examples in :doc:`get_start` to enjoy MetaDrive!

Interesting experiment results can be found in `our paper <https://arxiv.org/pdf/2012.13681>`_.
You can also visit `our webpage <https://decisionforce.github.io/metadrive/>`_ and `GitHub repo <https://github.com/decisionforce/metadrive>`_! Please feel free to contact us if you have any suggestions or ideas!


Table of Content
################
.. toctree::
    :caption: Home

    self

.. toctree::
   :maxdepth: 2
   :caption: Quick Start

   install.rst
   get_start.rst

.. toctree::
   :maxdepth: 2
   :caption: Configuration

   vehicle_config.rst
   env_config.rst

Citation
########

If you find this work useful in your project, please consider to cite it through:

.. code-block:: latex

    @article{li2020improving,
      title={Improving the Generalization of End-to-End Driving through Procedural Generation},
      author={Li, Quanyi and Peng, Zhenghao and Zhang, Qihang and Qiu, Cong and Liu, Chunxiao and Zhou, Bolei},
      journal={arXiv preprint arXiv:2012.13681},
      year={2020}
    }
