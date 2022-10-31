from metadrive.component.pgblock.bottleneck import Merge, Split
from metadrive.component.pgblock.curve import Curve
from metadrive.component.pgblock.fork import InFork, OutFork
from metadrive.component.pgblock.parking_lot import ParkingLot
from metadrive.component.pgblock.ramp import InRampOnStraight, OutRampOnStraight
from metadrive.component.pgblock.roundabout import Roundabout
from metadrive.component.pgblock.std_intersection import StdInterSection
from metadrive.component.pgblock.std_t_intersection import StdTInterSection
from metadrive.component.pgblock.straight import Straight
from metadrive.component.pgblock.tollgate import TollGate
from metadrive.component.pgblock.bidirection import Bidirection


class PGBlockDistConfig:
    MAX_LANE_NUM = 5
    MIN_LANE_NUM = 1

    # Register the block types here! Set their probability to 0.0 if you don't wish it appears in standard MetaDrive.
    BLOCK_TYPE_DISTRIBUTION_V1 = {
        Curve: 0.5,
        Straight: 0.1,
        StdInterSection: 0.075,
        Roundabout: 0.05,
        StdTInterSection: 0.075,
        InRampOnStraight: 0.1,
        OutRampOnStraight: 0.1,
        InFork: 0.00,
        OutFork: 0.00,
        Merge: 0.00,
        Split: 0.00,
        ParkingLot: 0.00,
        TollGate: 0.00
    }

    BLOCK_TYPE_DISTRIBUTION_V2 = {
        # 0.3 for curves
        Curve: 0.3,
        # 0.3 for straight
        Straight: 0.1,
        InRampOnStraight: 0.1,
        OutRampOnStraight: 0.1,
        # 0.3 for intersection
        StdInterSection: 0.15,
        StdTInterSection: 0.15,
        # 0.1 for roundabout
        Roundabout: 0.1,
        InFork: 0.00,
        OutFork: 0.00,
        Merge: 0.00,
        Split: 0.00,
        ParkingLot: 0.00,
        TollGate: 0.00,
        Bidirection: 0.00
    }

    @classmethod
    def all_blocks(cls, version: str = "v2"):
        return list(cls._get_dist(version).keys())

    @classmethod
    def get_block(cls, block_id: str, version: str = "v2"):
        for block in cls.all_blocks(version):
            if block.ID == block_id:
                return block
        raise ValueError("No {} block type".format(block_id))

    @classmethod
    def block_probability(cls, version: str = "v2"):
        return list(cls._get_dist(version).values())

    @classmethod
    def _get_dist(cls, version: str):
        # if version == "v1":
        #     return cls.BLOCK_TYPE_DISTRIBUTION_V1
        if version == "v2":
            return cls.BLOCK_TYPE_DISTRIBUTION_V2
        else:
            raise ValueError("Unknown version: {}".format(version))
