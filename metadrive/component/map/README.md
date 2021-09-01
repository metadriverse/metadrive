To run the `drive_in_argoverse.py` example, you need to install the argoverse-api and download the map files.

# Download argoverse-api
```bash
DOWNLOAD_PATH=$HOME"/tmp"
mkdir -p $DOWNLOAD_PATH
cd $DOWNLOAD_PATH
git clone https://github.com/argoai/argoverse-api.git
```

# Install argoverse-api
```bash
cd argoverse-api
pip install -e .
```

# Download and extract argoverse map files
```bash
wget https://s3.amazonaws.com/argoai-argoverse/hd_maps.tar.gz
tar -xzvf hd_maps.tar.gz
rm hd_maps.tar.gz
```

Your directory structure should look something like this:
```
argoverse-api
└── argoverse
    └── data_loading
    └── evaluation
    └── map_representation
    └── utils
    └── visualization
└── map_files
└── license
...
```