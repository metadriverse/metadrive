import numpy as np
from numpy.linalg import inv
import cv2
import matplotlib.pyplot as plt

def testing(img_array,demo = False):
    def pix2vox(img, camera_parameters):
        def val2dist(img):
            return -np.exp(np.log(16) * img/255)*5
        
        distance_buffer = val2dist(img)
        H,W = img.shape
        X,Y=np.meshgrid(np.arange(-W/2,W/2,1), np.arange(-H/2,H/2,1)) #so that the center of image correspond to (0,0)
        XYZ = np.array([-X.flatten(),Y.flatten(), np.ones(H*W)])      #append 1 for Z.
        K = camera_parameters["intrinsics"]
        eye_X_Y_Z = (inv(K)@XYZ)*distance_buffer.flatten()            #shape is ( 3,num_points)
        #print(inv(K)@XYZ)
        return eye_X_Y_Z.T
    params = dict(
        intrinsics = np.asarray(
            ((0.9330126047134399*512, 0, 0),           #            ((lens.focal* #horizontal pixel, 0, 0),           #
            (0, 0.9330126047134399/0.5*256, 0),        #            (0, lens.focal * aspect ratio*#vertical pixel, 0),#
            (0, 0, 1))                                 #            (0, 0, 1                           ))#
        )
    )
    vox = pix2vox(img_array, params)
    vox = vox[vox[:,2]>=-50] #correspond to all points with depth less than 50 meters
    vox = vox[(vox[:,1]-vox[:,1].min())>0.6] #correspond to all points with height at least 0.6 (unit? There remains some scaling issue)
                                             #above ground
    np.random.shuffle(vox)  #shuffle the points since we need relatively uniform samples later.
    vox = vox[:1000]        #take only 1000 points for visualization. 
    vox = vox.T
    xdata,ydata,zdata = vox[0,:],vox[1,:],vox[2,:]
    ret_x = -zdata
    ret_y = -xdata
    ret_z = ydata
    if demo:
        fig = plt.figure(figsize=(10,7))
        ax = plt.axes(projection='3d')
        ax.scatter3D(xdata,ydata,zdata,c = zdata, cmap = "viridis",)
        ax.view_init(elev=90, azim=-90)
        ax.set_xlim(-20,20)
        ax.set_ylim(-1.5,5)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        plt.show()
    return np.asarray([ret_x, ret_y, ret_z])


if __name__ == '__main__':
    img = cv2.imread("./depth_pic.png") #modify this to a picture captured by the depth camera with current camera setting
    img_array = np.asarray(img)
    cv2.imshow("some",img_array)
    print(img_array.shape)
    depth_img_array = img_array[...,-1]
    print(depth_img_array.shape)
    testing(depth_img_array,True)




     