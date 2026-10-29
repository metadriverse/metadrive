from metadrive.component.sensors.depth_camera import DepthCamera
from metadrive.envs.safe_metadrive_env import SafeMetaDriveEnv
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    env = SafeMetaDriveEnv(
        {
            "num_scenarios": 1,
            "traffic_density": 0.,
            "accident_prob": 1.,
            "start_seed": 4,
            "map": "SSSSS",
            "manual_control": True,
            # "use_render": True,
            "image_observation": True,
            "norm_pixel": False,
            "use_render": True,
            "debug": True,
            "interface_panel": ["depth_camera"],
            "sensors": dict(depth_camera=(DepthCamera, 800, 600)),
            "vehicle_config": dict(image_source="depth_camera"),
            # "map_config": {
            #     BaseMap.GENERATE_TYPE: MapGenerateMethod.BIG_BLOCK_NUM,
            #     BaseMap.GENERATE_CONFIG: 12,
            #     BaseMap.LANE_WIDTH: 3.5,
            #     BaseMap.LANE_NUM: 3,
            # }
        }
    )
    env.reset()

    import cv2

    for i in range(1, 100000):
        o, r, tm, tc, info = env.step([0, 1])
        assert env.observation_space.contains(o)

        depth_image_display = o["image"][..., 0, -1]
        if env.config["norm_pixel"]:
            depth_image_display = (o["image"][..., 0, -1] * 255).astype(np.uint8)

        # Apply a colormap to the depth image for better visualization
        depth_image_colormap = cv2.applyColorMap(depth_image_display, cv2.COLORMAP_VIRIDIS)

        # Display the depth image with a colormap
        cv2.imshow("Depth Image", depth_image_colormap)
        cv2.waitKey(1)

        if env.config["use_render"]:
            # for i in range(ImageObservation.STACK_SIZE):
            #     ObservationType.show_gray_scale_array(o["image"][:, :, i])
            env.render()
        # if tm or tc:
        #     # print("Reset")
        #     env.reset()
    env.close()
