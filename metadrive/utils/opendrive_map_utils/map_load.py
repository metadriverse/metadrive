from metadrive.utils.opendrive_map_utils.parser import parse_opendrive
from metadrive.utils.opendrive_map_utils.link_index import LinkIndex
from lxml import etree


def get_lane_width(lane):
    if not len(lane.widths) <= 1:
        print("too many lane width value! move to logging warning")
    if len(lane.widths) == 1:
        width = lane.widths[0].polynomial_coefficients[0]
        # assert sum(lane.widths[0].polynomial_coefficients) == self.width, "Only support fixed lane width"
    else:
        # TODO LQY: remove 4.0
        width = lane.roadMark.get("width", 4.0)
    return float(width) if isinstance(width, str) else width


def get_lane_id(lane):
    section_id = lane.lane_section.idx
    road_id = lane.lane_section.parentRoad.id
    lane_id = lane.id
    return "{}-{}-{}".format(road_id, section_id, lane_id)


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
    map = load_opendrive_map("C:\\Users\\x1\\Desktop\\Town01.xodr")
    link_index = LinkIndex()
    link_index.create_from_opendrive(map)
    print(link_index)
