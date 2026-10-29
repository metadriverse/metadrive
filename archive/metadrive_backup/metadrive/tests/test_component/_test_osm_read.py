import osmnx as ox
from pyrosm import OSM

# get real map in PBF fromat from https://extract.bbbike.org/
# install pyrosm and all dependencies via pip install pyrosm osmnx networkx

# Initialize the reader
osm = OSM("cpii.osm.pbf")

# Get all walkable roads and the nodes
nodes, edges = osm.get_network(nodes=True, network_type="driving")
# Create NetworkX graph
G = osm.to_graph(nodes, edges, graph_type="networkx")

ccc = edges.head(40)
# print(ccc.geometry)
ox.plot_graph(G)
