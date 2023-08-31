import cv2
import numpy as np
buffer = np.zeros((256,512,3))
buffer[:,:,2] = 255
cv2.imwrite("red.png", buffer)