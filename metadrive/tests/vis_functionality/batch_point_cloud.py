import os

import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1600, height=900, visible=True)

    # 8bit 1channel or 8bit 3channel, and depth image is: 16bit 1channel image
    pcds = []
    # for h in range(-180, 180, 20):
    h = 0
    dir = "C:\\Users\\x1\\Desktop\\neurips_2023\\sensor_video\\depth"
    files = os.listdir(dir)
    if "output.mp4" in files:
        files.remove("output.mp4")
    s = sorted(files, key=lambda x: int(x[6:-4]))
    for k, file in enumerate(s):
        raw_depth = o3d.io.read_image(os.path.join(dir, file))
        raw_depth = np.array(raw_depth)[..., 0]
        raw_depth *= 255
        raw_depth = raw_depth.astype(np.uint16)
        raw_depth = o3d.geometry.Image(raw_depth)
        # extrinsic = np.array([[np.cos(h), -np.sin(h), 0, 0],
        #                       [np.sin(h), np.cos(h), 0, 0],
        #                       [0, 0, 1, 0],
        #                       [0, 0, 0, 1]])
        extrinsic = np.array([[1, 0, 0, 0], [0, np.cos(h), -np.sin(h), 0], [0, np.sin(h), np.cos(h), 0], [0, 0, 0, 1]])
        intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        intrinsic.set_intrinsics(1600, 900, 1.866, 2.066, 800, 450)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            depth=raw_depth,
            intrinsic=intrinsic,
            extrinsic=extrinsic,
            project_valid_depth_only=True,
            stride=5,
            # depth_scale=10000.0,
            # depth_trunc=10000
        )
        # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        r = pcd.get_rotation_matrix_from_xyz((0, 0, np.pi))
        pcd = pcd.rotate(r, center=(0, 0, 0))
        # pcd = pcd.scale(10, center=(0, 0, 0))
        # pcd = pcd.scale(10, center=(0, 0, 0))
        # pcd.translate(np.array([0,0,-100]))
        vis.add_geometry(pcd)
        ctr = vis.get_view_control()
        assert id(ctr) == id(vis.get_view_control())  # assertion error.
        # ctr.set_up([0, -1, 0])
        # ctr.set_front([0, 0, -1])
        # ctr.set_lookat([1, 0, 0])
        ctr.set_zoom(0.25)
        #
        # camera_parameters = ctr.convert_to_pinhole_camera_parameters()
        # ex = np.array(camera_parameters.extrinsic)
        # ex[2][3]+=10
        # camera_parameters.extrinsic = ex
        # ctr.convert_from_pinhole_camera_parameters(camera_parameters)

        # vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()

        vis.capture_screen_image("img_{}.png".format(k))
        vis.remove_geometry(pcd)

    vis.destroy_window()

    # CX_DEPTH=0
    # CY_DEPTH=0
    # FX_DEPTH=300
    # FY_DEPTH=300
    # raw_depth = o3d.io.read_image('debug_{}.jpg'.format(0))
    # raw_depth = np.array(raw_depth)[..., 0]
    # raw_depth = raw_depth.astype(np.uint8)
    # width = 800
    # height = 600
    # # o3d.visualization.draw_geometries(pcds)
    #
    # jj = np.tile(range(width), height)
    # ii = np.repeat(range(height), width)
    # # Compute constants:
    # xx = (jj - CX_DEPTH) / FX_DEPTH
    # yy = (ii - CY_DEPTH) / FY_DEPTH
    # # transform depth image to vector of z:
    # length = height * width
    # z = raw_depth.reshape(height * width)
    # # compute point cloud
    # pcd = np.dstack((xx * z, yy * z, z)).reshape((length, 3))
    # pcd_o3d = o3d.geometry.PointCloud()  # create point cloud object
    # pcd_o3d.points = o3d.utility.Vector3dVector(pcd)  # set pcd_np as the point cloud points
    # # Visualize:
    # o3d.visualization.draw_geometries([pcd_o3d])
