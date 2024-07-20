import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import Wrap
import math
import cv2
import sys
# array=np.array([[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]])
# print(np.roll(array,np.floor(np.array(array.shape) / 2).astype(int)*(-1),axis=(1,0)))
# letter=cv2.imread("C:\\xampp\\htdocs\\Visual-inspection\\PSF\\letter.png")
# print(np.sum(letter[1][1][:])== 255*3)

# for i in range(letter.shape[0]):
#     for j in range(letter.shape[1]):
#         if np.sum(letter[i][j][:]) > 200 and np.sum(letter[i][j][:]) != 255*3:
#             letter[i][j]=(255,255,255)
        
# plt.imshow(letter, cmap='gray')
# plt.show()
r=np.array([1.43917424, 1.4168656,  1.39492525, 0])
zernike_value=np.array([1.43917424, 1.4168656,  1.39492525, 1.37337083])
zernike_value = np.where(r>1 , 0, zernike_value)
print(zernike_value)