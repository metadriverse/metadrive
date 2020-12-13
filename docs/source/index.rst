.. PGDrive documentation master file, created by
   sphinx-quickstart on Tue Dec  8 13:36:14 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

#####################
PGDrive Home
#####################

Welcome to the PGDrive home.
PGDrive is a lightweight autonomous driving environment.
Referring to this documentation, you can utilize PGDrive to research diverse driving **topics**, including:

- Reinforcement learning
- Imitation leaning
- Modular autonomous driving

We now provide several **sensors** to collect environment information:

- Rgb Camera
- Depth Camera
- Bird-view Camera
- Lidar

Various cameras can be equipped on your autonomous driving car to sense the environment.
Lidar in PGDrive is a pseudu-lidar.
Besides the could points, the information of surrounding vehicles like speed, position, heading can also be obtained
from lidar.

Based on procedural generation technology, our map generator can generate numerous maps and driving scenes, in which your
AI driver can interact with other traffic vehicles driven by IDM model.


Content
#############
.. toctree::
   :maxdepth: 2

   self
   install
   getting_start
   generalization_env_config



