from metadrive import MetaDriveEnv
from metadrive.component.pgblock.first_block import FirstPGBlock
from metadrive.component.vehicle.vehicle_type import VaryingDynamicsVehicle, DefaultVehicle
from metadrive.engine import initialize_engine, close_engine


def test_varying_dynamics_vehicle():
    try:

        env = MetaDriveEnv()
        env.reset()
        default_config = env.config

        # close_engine()
        # initialize_engine(default_config)

        v_config = default_config.copy()["vehicle_config"]
        v_config["spawn_lane_index"] = (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0)
        v_config["navigation_module"] = None

        v = VaryingDynamicsVehicle(v_config, random_seed=0)
        ref_v = DefaultVehicle(v_config, random_seed=0)
        assert v.WIDTH == ref_v.WIDTH
        assert v.LENGTH == ref_v.LENGTH
        assert v.HEIGHT == ref_v.HEIGHT

        for width in [ref_v.WIDTH, 2, 3, 4]:
            for height in [ref_v.HEIGHT, 1, 2]:
                for length in [ref_v.LENGTH, 4, 6, 9, 13, 15]:
                    for friction in [0.2, 0.8, 1.0, 1.2, 1.5, 2.0]:
                        v.reset(
                            vehicle_config={
                                "spawn_lane_index": (FirstPGBlock.NODE_1, FirstPGBlock.NODE_2, 0),
                                "width": width,
                                "height": height,
                                "length": length,
                                "wheel_friction": friction
                            }
                        )
                        assert v.WIDTH == width
                        assert v.LENGTH == length
                        assert v.HEIGHT == height
                        for wheel in v.wheels:
                            assert abs(wheel.getFrictionSlip() - friction) < 1e-5, (wheel.getFrictionSlip(), friction)

    finally:
        close_engine()


if __name__ == "__main__":
    test_varying_dynamics_vehicle()
