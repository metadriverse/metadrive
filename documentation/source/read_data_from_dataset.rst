.. _read_data_from_dataset:


########################
Read Data from Dataset
########################


We provide a script to enable users to drive in a real scenario. Please follow the instructions 
in :code:`metadrive/component/map/README.md` and run::
Setting up Argoverse dataset
#############################

To run the `drive_in_argoverse.py` example, you need to install the argoverse-api and download the map files.

Install argoverse-api repo
*********************************************

Clone the official argoverse API repo to some place then install it:


.. code-block:: bash

    git clone https://github.com/argoai/argoverse-api.git
    cd argoverse-api
    pip install -e .


Download and extract argoverse map files
*********************************************

.. code-block:: bash

    cd argoverse-api
    wget https://s3.amazonaws.com/argoai-argoverse/hd_maps.tar.gz
    tar -xzvf hd_maps.tar.gz

After unzip the map files, your directory structure should look like this:

.. code-block::

    argoverse-api
    └── map_files
    └── license
    └── argoverse
        └── data_loading
        └── evaluation
        └── map_representation
        └── utils
        └── visualization
    ...



Enjoy Driving in Real Town!
############################################

You can launch a script to drive manually in a town in Pittsburgh with replayed traffic flow recorded in Argoverse.
Note: Press T to launch auto-driving! Enjoy!

.. code-block:: bash

    # Make sure current folder does not have a sub-folder named metadrive
    python -m metadrive.examples.drive_in_argoverse



Specify the Data to Replay
###############################

MetaDrive currently supports replay the map and traffic flow in Argoverse dataset.
As shown in `ArgoverseEnv <https://github.com/decisionforce/metadrive/blob/main/metadrive/envs/argoverse_env.py>`_,
we specify few parameters to limit the map in a region of the city. Here is the detailed explaination of those parameters:


- :code:`argoverse_city` (str = "PIT"): Optional in ["PIT", XXX]. The shortcut of the specified city.
- :code:`argoverse_map_xcenter` (float): The XXXX

	python metadrive/examples/drive_in_argoverse.py

We will add more scenarios for customized selection in the near future. For users' reference, we explain the role
of several values in :code:`metadrive/envs/argoverse_env.py`:

	- :code:`xcenter`, :code:`ycenter`, :code:`radius`: Only roads within the circle with corresponding center and radius will be loaded.
	- 
