import pickle
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
from metadrive.engine.asset_loader import AssetLoader


def read_waymo_data(file_path):
    with open(file_path, "rb+") as waymo_file:
        data = pickle.load(waymo_file)
    return data

def draw_waymo_map(data):
    figure(figsize=(8, 6), dpi=500)
    for key, value in data["map"].items():
        if value.get("type", None) =="center_lane":
            plt.scatter([x[0] for x in value["polyline"]], [y[1] for y in value["polyline"]],s=0.5)
    plt.show()

if __name__=="__main__":
    file_path = AssetLoader.file_path("waymo", "48.pkl", linux_style=False)
    data = read_waymo_data(file_path)
    draw_waymo_map(data)
