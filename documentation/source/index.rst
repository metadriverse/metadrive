.. image:: figs/logo-horizon.png
   :width: 1800
   :align: center


|
|

########################
MetaDrive Documentation
########################


Welcome to the MetaDrive documentation!
MetaDrive is an efficient and compositional driving simulator with the following key features:

* Compositional: It supports generating infinite scenes with various road maps and traffic settings for the research of generalizable RL.
* Lightweight: It is easy to install and run. It can run up to 1,500 FPS on a standard PC.
* Realistic: Accurate physics simulation and multiple sensory input including Lidar, RGB images, top-down semantic map and first-person view images.


This documentation brings you the information on installation, usages and more of MetaDrive!

You can also visit the `GitHub repo <https://github.com/metadriverse/metadrive>`_ of MetaDrive.
Please feel free to contact us if you have any suggestions or ideas!




.. toctree::
   :maxdepth: 2
   :hidden:
   :caption: Quick Start

   install.rst
   get_start.rst
   before_reading.ipynb

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: RL Training with MetaDrive

   rl_environments.ipynb
   config_system.ipynb
   obs.ipynb
   action.ipynb
   reward_cost_done.ipynb
   training.ipynb
   multigoal_intersection.ipynb

.. toctree::
   :includehidden:
   :maxdepth: 1
   :caption: Concepts and Customization

   new_env.ipynb
   system_design.ipynb
   sensors.ipynb
   top_down_render.ipynb
   panda_render.ipynb
   map.ipynb
   vehicle.ipynb
   navigation.ipynb
   scenario_description.ipynb
   record_replay.ipynb
   simgen_render.ipynb

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Debugging

   debug_mode.ipynb
   points_and_lines.ipynb
   log_msg.ipynb

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Ecosystem

   scenarionet.rst
   ros.rst
   di_drive.rst
   sumo.rst
   opendrive.rst


.. raw:: html

    <table width="100%" style="margin: 0pt 0pt; text-align: center;">
        <tr>
            <td>
                <video style="display:block; width:100%; height:auto; "
                       autoplay="autoplay" muted loop="loop" controls playsinline>
                    <source src="https://raw.githubusercontent.com/decisionforce/archive/master/MetaDrive/metadrive_teaser.mp4"
                            type="video/mp4"/>
                </video>
            </td>
        </tr>
    </table>
    <br><br>


We also make a video to benchmark the FPS of MetaDrive in different platforms. It can run at +2000FPS at MacBook Pro. Please checkout the
`YouTube video <https://youtu.be/8P4jKAvq_54>`_.


Relevant Projects
#################

.. raw:: html

    <b>
        Learning to Simulate Self-driven Particles System with Coordinated Policy Optimization
    </b> <br>
    Zhenghao Peng, Quanyi Li, Chunxiao Liu, Bolei Zhou
    <br>
    <i>NeurIPS 2021</i><br>
    [<a href="https://arxiv.org/pdf/2110.13827.pdf" target="_blank">Paper</a>]
    [<a href="https://github.com/decisionforce/CoPO" target="_blank">Code</a>]
    [<a href="https://decisionforce.github.io/CoPO" target="_blank">Webpage</a>]
    [<a href="https://decisionforce.github.io/CoPO/copo_poster.pdf" target="_blank">Poster</a>]
    [<a href="https://youtu.be/sOw43l8lwxE" target="_blank">Talk</a>]

    <br><br>
    <b>Safe Driving via Expert Guided Policy Optimization</b> <br>
    Zhenghao Peng*, Quanyi Li*, Chunxiao Liu, Bolei Zhou
    <br>
    <i>Conference on Robot Learning (CoRL) 2021</i><br>
    [<a href="https://arxiv.org/pdf/2110.06831.pdf" target="_blank">Paper</a>]
    [<a href="https://github.com/decisionforce/EGPO" target="_blank">Code</a>]
    [<a href="https://decisionforce.github.io/EGPO/" target="_blank">Webpage</a>]
    [<a href="https://decisionforce.github.io/EGPO/images/egpo_poster.png" target="_blank">Poster</a>]

    <br><br>
    <b>
        Efficient Learning of Safe Driving Policy via Human-AI Copilot Optimization
    </b> <br>
    Quanyi Li*, Zhenghao Peng*, Bolei Zhou
    <br>
    <i>ICLR 2022</i><br>
    [<a href="https://arxiv.org/pdf/2202.10341.pdf" target="_blank">Paper</a>]
    [<a href="https://github.com/decisionforce/HACO" target="_blank">Code</a>]
    [<a href="https://decisionforce.github.io/HACO" target="_blank">Webpage</a>]

    <br><br>


And more:

* Quanyi Li, Zhenghao Peng, Haibin Wu, Lan Feng, Bolei Zhou. "Human-AI Shared Control via Policy Dissection." (**NeurIPS 2022**)

* Yang, Yujie, Yuxuan Jiang, Yichen Liu, Jianyu Chen, and Shengbo Eben Li. "Model-Free Safe Reinforcement Learning through Neural Barrier Certificate." IEEE Robotics and Automation Letters (2023).

* Feng, Lan, Quanyi Li, Zhenghao Peng, Shuhan Tan, and Bolei Zhou. "TrafficGen: Learning to Generate Diverse and Realistic Traffic Scenarios." (**ICRA 2023**)

* Zhenghai Xue, Zhenghao Peng, Quanyi Li, Zhihan Liu, Bolei Zhou. "Guarded Policy Optimization with Imperfect Online Demonstrations." (**ICLR 2023**)




Citation
########

You can read `our white paper <https://arxiv.org/pdf/2109.12674.pdf>`_ describing the details of MetaDrive! If you use MetaDrive in your own work, please cite:

.. code-block:: latex

    @article{li2021metadrive,
      title={MetaDrive: Composing Diverse Driving Scenarios for Generalizable Reinforcement Learning},
      author={Li, Quanyi and Peng, Zhenghao and Xue, Zhenghai and Zhang, Qihang and Zhou, Bolei},
      journal={arXiv preprint arXiv:2109.12674},
      year={2021}
    }

