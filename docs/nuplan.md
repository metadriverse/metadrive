> **Warning**
> This document should be merged into other official documentation!


# Integrate Nuplan dataset with MetaDrive


## Set up the environment

Note:

* Nuplan require `python>=3.9`

```bash

# Enter the folder of MetaDrive and install nuplan dependencies
cd ~/metadrive
pip install -e .[nuplan]

# Install pytorch by yourself. Read: https://pytorch.org/get-started/locally/
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# Exit MetaDrive folder, install nuplan toolkit
cd ~
git clone https://github.com/motional/nuplan-devkit.git
cd nuplan-devkit

# First, remove "opencv-python<=4.5.1.48" in requirements.txt
pip install -r requirements.txt

# Then, install opencv-python
pip install opencv-python

# Install nuplan
pip install -e .

# Further more dependencies:
pip install "pytorch-lightning==2.0.0"
```

## Download dataset

Officla dataset setup tutorial is at:  https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html

Here is our workflow:

* Step 1: Go to https://www.nuscenes.org/nuplan#download to download (1) maps, (2) mini split.
* Step 2: Unzip the map to `~/nuplan/dataset` (it will create a `maps` folder) and unzip the mini split to
`~/nuplan/dataset` and rename folders to make sure they comply [file hierarchy](https://nuplan-devkit.readthedocs.io/en/latest/dataset_setup.html#filesystem-hierarchy).
* Step 3: Add the following paths to `~/.bashrc`:

```bash
export NUPLAN_DATA_ROOT="$HOME/nuplan/dataset"
export NUPLAN_MAPS_ROOT="$HOME/nuplan/dataset/maps"
export NUPLAN_EXP_ROOT="$HOME/nuplan/exp"
```


We also provide a script to accelerate environment setup. First, `cd` to the folder where you already downloaded the zip files.

```bash
# Create folders
mkdir -p ~/nuplan/dataset/nuplan-v1.1

unzip nuplan-maps-v1.0.zip -d ~/nuplan/dataset/  # Will create maps automatically
unzip nuplan-v1.1_mini.zip -d ~/nuplan/dataset/

mv ~/nuplan/dataset/data/cache/mini ~/nuplan/dataset/nuplan-v1.1/  # Will move "mini" folder to nuplan-v1.1
rm ~/nuplan/dataset/data/ -r  # Remove empty folder

# Add the paths to .bashrc:
vim ~/.bashrc

export NUPLAN_DATA_ROOT="$HOME/nuplan/dataset"
export NUPLAN_MAPS_ROOT="$HOME/nuplan/dataset/maps"
export NUPLAN_EXP_ROOT="$HOME/nuplan/exp"
```

Now you can run the following scripts to verify the installation:

```bash
python metadrive/envs/real_data_envs/nuplan_env.py
```
