from metadrive.component.sensors.depth_camera import DepthCamera
from metadrive.engine.engine_utils import get_engine
from metadrive.engine.asset_loader import AssetLoader
from metadrive.constants import CamMask
import numpy as np
import random

class DepthLidar(DepthCamera):
    def __init__(self, width, height, engine, *, cuda=False) :
        super().__init__(width, height, engine, cuda=cuda)
        num_points = height*width
        visualize = False
        self.visualize = visualize
        self.num_points = num_points
        self.offset = 3
        self.cloud_points = []
        _,vfov = self.lens.getFov()
        self.lens.setFov(40,vfov)

        self.points = None
        self.mask = None
    
    def setup_cloud_points(self, num_points, cloud_points):
        engine = get_engine()
        AssetLoader.init_loader(engine)
        loader = AssetLoader.get_loader()
        for _ in range(num_points):
            node_point = loader.load_model(AssetLoader.file_path("models", "sphere.egg"))
            node_point.setColor(0,0,1)
            node_point.setScale(0.05, 0.05,0.05) 
            node_point.hide(CamMask.AllOn)
            
            node_point.reparentTo(engine.render)
            cloud_points.append(node_point)

    def update_cloud_points(self, cloud_points, positions):
        off_position = positions.shape[0]
        for idx in range(off_position):
            node = self.cloud_points[idx]
            node.show(CamMask.MainCam)
            node.setPos(positions[idx][0],positions[idx][1],positions[idx][2])
        for idx, node in enumerate(cloud_points[off_position:]):
            node.hide(CamMask.AllOn)


    def perceive(self, base_object, clip=True) -> np.ndarray:
        if len(self.cloud_points)==0 and self.visualize:
            self.setup_cloud_points(self.num_points, self.cloud_points)

        hfov,vfov = self.lens.fov 
        depth = super(DepthLidar,self).get_image(base_object)
        result = super(DepthLidar,self).perceive(base_object,clip)
        points_camframe, mask = self.generate_cloud_points(depth[...,-1], self.BUFFER_H,self.BUFFER_W, hfov,vfov)
        transformation = self.get_matrix()
        def to_homogeneous(coords):
            new_column = np.ones((coords.shape[0], 1))
            return np.hstack((coords, new_column))
        homogeneous_points = to_homogeneous(points_camframe[mask])
        world_homo_points = homogeneous_points @ transformation
        if self.visualize:
            self.update_cloud_points(self.cloud_points, world_homo_points[:,:-1])
        self.points = points_camframe
        self.mask = mask
        return result
    
    
    def track(self,base_object):
        base_object.origin.hide(self.CAM_MASK)
        self.origin.setPos(base_object.origin, 0, -self.offset, 0)
        return super(DepthLidar,self).track(base_object)


    def get_matrix(self) -> np.array:
        mat = self.cam.getTransform(get_engine().render).getMat()
        mat_array = np.array(
            [
                [mat.getCell(i,j) for j in range(4)] for i in range(4)
            ] 
        )
        return mat_array
    
    
 
    def generate_cloud_points(self, img_array, H,W, hfov, vfov):
        """
        W: num pixel in width; H: num pixel in Height.
        hfov: horizontal fov in deg; vfov: vertical fov in deg
        """
        def find_f(H,W,hfov,vfov):
            fx = (W/2)/(np.tan(np.deg2rad(hfov)/2))
            fy = (H/2)/(np.tan(np.deg2rad(vfov)/2))
            return fx,fy
        fx,fy = find_f(H,W,hfov,vfov)
        def pix2vox(img,fx,fy,H,W):
            def val2dist(img):
                return np.exp(np.log(16) * img/255)*5
            distance_buffer = val2dist(img)
            H,W = img.shape
            def convert_from_uvd( u, v, d, cx,cy,focalx,focaly):
                x_over_z = (u-cx) / focalx
                y_over_z = (v-cy) / focaly
                z = d / np.sqrt( 1+x_over_z**2 + y_over_z**2)
                x = x_over_z * z
                y = y_over_z * z
                return x, y, z
            points = []
            for v in range(H):
                for u in range(W):
                    point = convert_from_uvd(u,v,distance_buffer[v,u],W/2,H/2,fx,fy)
                    points.append(point)
            return np.asarray(points)
        vox = pix2vox(img_array,fx,fy,H,W)
        mask1 = vox[:,2] <= (50 + self.offset)
        mask2 = vox[:,2] >= self.offset
        mask = mask1 == mask2 
        #vox = vox[vox[:,2]<=50 + self.offset]
        #vox = vox[vox[:,2]> self.offset]
        vox = vox[:,[0,2,1]]
        vox[:,2] = -vox[:,2]
        #indices = np.random.choice(vox.shape[0], size=self.num_points, replace=False)
        return vox, mask#vox[indices]
