

.. image:: ../../metadrive/assets/logo-horizon.png
   :width: 1800
   :align: center


|
|

########################
MetaDrive Documentation
########################

Welcome to the MetaDrive documentation!
MetaDrive is an efficient and compositional driving simulator for reinforcement learning community!
The key features of MetaDrive includes:

- **Lightweight**: Extremely easy to download, install and run in almost all platforms.
- **Realistic**: Accurate physics simulation and multiple sensory input including RGB camera, Lidar and sensory data.
- **Efficient**: Up to 300 simulation step per second.
- **Open-ended**: Support generating infinite scenes and configuring various traffics, vehicles, and environmental settings.

This documentation brings you the information on installation, philosophy, usages and development tips of MetaDrive!


You can also visit `our webpage <https://decisionforce.github.io/metadrive/>`_ and `GitHub repo <https://github.com/decisionforce/metadrive>`_.
Please feel free to contact us if you have any suggestions or ideas!


Citation
########

We wrote a white paper on this project, but the citation information is not yet well prepared!
Please contact us if you find this work useful in your project.




.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Quick Start

   install.rst
   get_start.rst

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: RL Training with MetaDrive

   rl_environments.rst
   observation.rst
   reward_and_cost.rst
   action_and_dynamics.rst
   config_system.rst
   read_data_from_dataset.rst


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Concept and Customization

   concept.rst
   development.rst
