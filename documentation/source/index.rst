

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

- **Lightweight**: Extremely easy to download, install and run in almost all platforms. Up to 300 simulation step per second and easy to parallel.
- **Realistic**: Accurate physics simulation and multiple sensory input including Lidar, sensory data, top-down semantic map and first-person view images.
- **Compositional**: Support generating infinite scenes and configuring various traffics, vehicles, and environmental settings.

This documentation brings you the information on installation, usages and more of MetaDrive!

You can also visit the `GitHub repo <https://github.com/decisionforce/metadrive>`_ of MetaDrive.
Please feel free to contact us if you have any suggestions or ideas!


.. Citation
.. ########

.. We wrote a white paper on this project, but the citation information is not yet well prepared!
.. Please contact us if you find this work useful in your project.


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
   reward_cost_and_termination_function.rst
   action_and_dynamics.rst
   config_system.rst
   read_data_from_dataset.rst

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Concept and Customization

   concept.rst
   development.rst


.. raw:: html

    <br>
    <table width="100%" style="margin: 0 0; text-align: center;">
        <tr>
            <td>
            <iframe width="560" height="315" src="https://www.youtube.com/embed/3ziJPqC_-T4" title="MetaDrive Introduction" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
            </td>
        </tr>
    </table>


