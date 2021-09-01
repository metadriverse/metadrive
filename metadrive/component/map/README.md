# Tutorial on setting up Argoverse dataset

To run the `drive_in_argoverse.py` example, you need to install the argoverse-api and download the map files.

## Install argoverse-api repo

Clone the official argoverse API repo to some place then install it:

```bash
git clone https://github.com/argoai/argoverse-api.git
cd argoverse-api
pip install -e .
```

## Download and extract argoverse map files

```bash
cd argoverse-api
wget https://s3.amazonaws.com/argoai-argoverse/hd_maps.tar.gz
tar -xzvf hd_maps.tar.gz
```

After unzip the map files, your directory structure should look like this:

```
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
```

## Verify the installation

Please run the following script to verify the installation.
Note: Press T to launch auto-driving! Enjoy!

```bash
# Make sure current folder does not have a sub-folder named metadrive
python -m metadrive.examples.drive_in_argoverse
```

