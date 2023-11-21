from metadrive.component.sensors.depth_camera import DepthCamera
from metadrive.engine.engine_utils import get_engine
import numpy as np
import random

class depth_lidar(DepthCamera):
    def __init__(self, num_points, width, height, engine, *, cuda=False) :
        super().__init__(width, height, engine, cuda=cuda)
        self.num_points = num_points
        
    def track(self, base_object):
        return super().track(base_object)
    
    def perceive(self, base_object, clip=True) -> np.ndarray:
        hfov,vfov = self.cam.lens.fov 
        depth = super().perceive(base_object, clip)
        points_camframe = self.cloud_points(depth, self.BUFFER_H,self.BUFFER_W, hfov,vfov)
        transformation = self.get_matrix()
        return points_camframe
    
    
    
    def get_matrix(self) -> np.array:
        mat = self.cam.getTransform(get_engine().render).getMat()
        mat_array = np.array(
            [
                [mat.getCell(i,j) for j in range(4)] for i in range(4)
            ] 
        )
        return mat_array
    
 
    def cloud_points(self, img_array, H,W, hfov, vfov):
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
        vox = vox[vox[:,2]<=50]
        vox = 
        vox = vox.tolist()
        random.shuffle(vox)
        return vox[:512]
