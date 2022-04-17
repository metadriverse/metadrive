.. _read_data_from_dataset:


########################
Read Data from Dataset
########################

.. raw:: html

    <br>
    <table width="100%" style="margin: 0 0; text-align: center;">
        <tr>
            <td>
            <iframe width="560" height="315" src="https://www.youtube.com/embed/EbC6-grcmRY" title="MetaDrive Multi-agent Environments" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
            </td>
        </tr>
    </table>
    <br>


Setting up Argoverse dataset
#############################

To run the :code:`drive_in_argoverse.py` example, you need to install the argoverse-api and download the map files.

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

MetaDrive currently supports replaying the map and traffic flow in the sample dataset of argoverse-tracking.
We will add more scenarios for customized selection in the near future.
As shown in `ArgoverseEnv Class <https://github.com/metadriverse/metadrive/blob/main/metadrive/envs/argoverse_env.py>`_,
we use a few parameters to specify the data to be loaded. Here is the detailed explanation of those parameters:


- :code:`argoverse_city`: The shortcut of the specified city.
- :code:`argoverse_map_center/radius`, :code:`radius`: Only the roads and traffic within the circle centering in :code:`argoverse_map_center` with radius :code:`argoverse_map_radius` will be loaded.
- :code:`argoverse_spawn_lane_index`: Node index indicating where the ego agent is initialized.
- :code:`argoverse_destination`: Node index indicating the destination of the ego agent.
- :code:`argoverse_log_id`: We select one sample of argoverse-tracking data with this ID as the folder name.
