import open3d as o3d

print(o3d.__version__)

# depth_raw = read_image("../../TestData/RGBD/depth/00000.png")
# pcd = create_point_cloud_from_rgbd_image(depth_raw, PinholeCameraIntrinsic(
#     PinholeCameraIntrinsicParameters.PrimeSenseDefault))
# draw_geometries([pcd])