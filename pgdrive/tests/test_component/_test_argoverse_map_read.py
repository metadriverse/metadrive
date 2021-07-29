"""
    (1) git clone git@github.com:argoai/argoverse-api.git
    (2) download argoverse maps at https://www.argoverse.org/data.html#download-link
    (3) uncompress the .tar file and rename the downloaded director to map_files
    (4) move the /map_files to argoverse-api repo
    (5) pip install -e /path_to_argoverse-api_directory/
"""
from argoverse.map_representation.map_api import ArgoverseMap

am = ArgoverseMap()
