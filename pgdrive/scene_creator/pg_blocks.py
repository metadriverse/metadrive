from pgdrive.scene_creator.blocks.bottleneck import Merge, Split
from pgdrive.scene_creator.blocks.curve import Curve
from pgdrive.scene_creator.blocks.fork import InFork, OutFork
from pgdrive.scene_creator.blocks.ramp import InRampOnStraight, OutRampOnStraight
from pgdrive.scene_creator.blocks.roundabout import Roundabout
from pgdrive.scene_creator.blocks.std_intersection import StdInterSection
from pgdrive.scene_creator.blocks.std_t_intersection import StdTInterSection
from pgdrive.scene_creator.blocks.straight import Straight
from pgdrive.scene_creator.blocks.parking_lot import ParkingLot
from pgdrive.scene_creator.blocks.tollgate import TollGate


class PGBlock:
    # Register the block types here! Set their probability to 0.0 if you don't wish it appears in standard PGDrive.
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
        TollGate: 0.00
    }

    @classmethod
    def all_blocks(cls, version: str):
        return list(cls._get_dist(version).keys())

    @classmethod
    def get_block(cls, block_id: str, version: str):
        for block in cls.all_blocks(version):
            if block.ID == block_id:
                return block
        raise ValueError("No {} block type".format(block_id))

    @classmethod
    def block_probability(cls, version: str):
        return list(cls._get_dist(version).values())

    @classmethod
    def _get_dist(cls, version: str):
        if version == "v1":
            return cls.BLOCK_TYPE_DISTRIBUTION_V1
        elif version == "v2":
            return cls.BLOCK_TYPE_DISTRIBUTION_V2
        else:
            raise ValueError("Unknown version: {}".format(version))
