.. _read_data_from_dataset:


########################
Read Data from Dataset
########################

.. warning:: This page is under construction!

We provide a script to enable users to drive in a real scenario. Please follow the instructions 
in :code:`metadrive/component/map/README.md` and run::

	python metadrive/examples/drive_in_argoverse.py

We will add more scenarios for customized selection in the near future. For users' reference, we explain the role
of several values in :code:`metadrive/envs/argoverse_env.py`:

	- :code:`xcenter`, :code:`ycenter`, :code:`radius`: Only roads within the circle with corresponding center and radius will be loaded.
	- 
