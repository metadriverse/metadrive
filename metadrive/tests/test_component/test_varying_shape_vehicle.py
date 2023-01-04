from metadrive import MetaDriveEnv
from metadrive.component.vehicle.vehicle_type import VaryingShapeVehicle, DefaultVehicle
from metadrive.engine import initialize_engine, close_engine


def test_varying_shape_vehicle():
    try:

        env = MetaDriveEnv()
        env.reset()
        default_config = env.config

        # close_engine()
        # initialize_engine(default_config)

        v_config = default_config.copy()["vehicle_config"]

        v_config["need_navigation"] = False

        v = VaryingShapeVehicle(v_config, random_seed=0)
        ref_v = DefaultVehicle(v_config, random_seed=0)
        assert v.WIDTH == ref_v.WIDTH
        assert v.LENGTH == ref_v.LENGTH
        assert v.HEIGHT == ref_v.HEIGHT

        for width in [ref_v.WIDTH, 2, 3, 4]:
            for height in [ref_v.HEIGHT, 1, 2]:
                for length in [ref_v.LENGTH, 4, 6, 9, 13, 15]:
                    v.reset(vehicle_config={"width": width, "height": height, "length": length})
                    assert v.WIDTH == width
                    assert v.LENGTH == length
                    assert v.HEIGHT == height

    finally:
        close_engine()


if __name__ == "__main__":
    test_varying_shape_vehicle()
