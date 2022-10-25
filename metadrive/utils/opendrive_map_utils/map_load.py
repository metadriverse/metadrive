from metadrive.utils.opendrive_map_utils.parser import parse_opendrive
from metadrive.utils.opendrive_map_utils.link_index import LinkIndex
from lxml import etree


def load_opendrive_map(path):
    # Load road network and print some statistics
    try:
        fh = open(path, "r")
        openDriveXml = parse_opendrive(etree.parse(fh).getroot())
        fh.close()
        return openDriveXml
    except (etree.XMLSyntaxError) as e:
        print("XML Syntax Error: {}".format(e))
        return None
    except (TypeError, AttributeError, ValueError) as e:
        print("Value Error: {}".format(e))
        return None


if __name__ == "__main__":
    map = load_opendrive_map("/home/shady/data/carla_911/CarlaUE4/Content/Carla/Maps/OpenDrive/Town01.xodr")
    link_index = LinkIndex()
    link_index.create_from_opendrive(map)
    print(link_index)
