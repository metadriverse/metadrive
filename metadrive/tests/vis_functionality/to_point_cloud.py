import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    # 8bit 1channel or 8bit 3channel, and depth image is: 16bit 1channel image
    pcds = []
    # for h in range(-180, 180, 20):
    h = 0
    raw_depth = o3d.io.read_image("C:\\Users\\x1\\Desktop\\neurips_2023\\sensor_video\\depth\\depth_36.jpg")
    raw_depth = np.array(raw_depth)[..., 0]
    raw_depth *= 255
    raw_depth = raw_depth.astype(np.uint16)
    raw_depth = o3d.geometry.Image(raw_depth)
    extrinsic = np.array([[np.cos(h), -np.sin(h), 0, 0], [np.sin(h), np.cos(h), 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    # extrinsic = np.array([[1, 0, 0, 0],
    #                       [0, np.cos(h), -np.sin(h), 0],
    #                       [0, np.sin(h), np.cos(h), 1.5],
    #                       [0, 0, 0, 1]])
    intrinsic = o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    intrinsic.set_intrinsics(1600, 900, 0.866, 0.866, 800, 450)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        depth=raw_depth,
        intrinsic=intrinsic,
        extrinsic=extrinsic,
        project_valid_depth_only=False,
        stride=4
        # depth_scale=1000.0,
        # depth_trunc=1000.0
    )
    # pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    r = pcd.get_rotation_matrix_from_xyz((0, 0, np.pi))
    pcd = pcd.rotate(r, center=(0, 0, 0))
    pcds.append(pcd)

    o3d.visualization.draw_geometries(pcds)

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
