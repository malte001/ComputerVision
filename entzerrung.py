

import dataclasses
import math
import matplotlib.pyplot as plt
import skimage
import numpy as np

## defines for functions

def rectification(im, a1, a2, a3, b1, b2, b3, c1, c2):
       
    h=im.shape[0]
    w=im.shape[1]
    # define new empty image
    im_new=np.zeros((h,w,im.shape[2]))

    for y_old in range(im_new.shape[1]): # y dir 
        for x_old in range(im_new.shape[0]): # x dir 
            denuminator=(b1*c2-b2*c1)*x_old+(a2*c1-a1*c2)*y_old+a1*b2-a2*b1
            x_new=int( np.ndarray.round( ( (b2-c2*b3)*x_old+(a3*c2-a2)*y_old+a2*b3-a3*b2 ) / denuminator ) )
            y_new=int( np.ndarray.round( ( (b3*c1-b1)*x_old+(a1-a3*c1)*y_old+a3*b1-a1*b3 ) / denuminator ) )
            
            if 0<=x_new<im.shape[0]-1 and 0<=y_new<im.shape[1]-1:
                im_new[x_old,y_old,:]=im[x_new,y_new,:]

    return im_new




## Init, picture and object points

# Matrix zur Bestimmung der Parameter a bis c
# multiplikation mit inverser für lsg-vektor
# M*a_vec=x_vec
M=np.zeros((8,8))
 
picture_points = [[338, 345],[432, 313],[335, 545],[423, 681]]
object_points = [[100, 250], [657, 250], [100, 610], [657, 610]]
x1_ob = object_points[0][0]
y1_ob = object_points[0][1]
x2_ob = object_points[1][0]
y2_ob = object_points[1][1]
x3_ob = object_points[2][0]
y3_ob = object_points[2][1]
x4_ob = object_points[3][0]
y4_ob = object_points[3][1]

x1_im = picture_points[0][0]
y1_im = picture_points[0][1]
x2_im = picture_points[1][0]
y2_im = picture_points[1][1]
x3_im = picture_points[2][0]
y3_im = picture_points[2][1]
x4_im = picture_points[3][0]
y4_im = picture_points[3][1]

M = np.array([
    [x1_im, y1_im, 1, 0, 0, 0, -x1_ob*x1_im, -x1_ob*y1_im],
    [0, 0, 0, x1_im, y1_im, 1, -y1_ob*x1_im, -y1_ob*y1_im],
    [x2_im, y2_im, 1, 0, 0, 0, -x2_ob*x2_im, -x2_ob*y2_im],
    [0, 0, 0, x2_im, y2_im, 1, -y2_ob*x2_im, -y2_ob*y2_im],
    [x3_im, y3_im, 1, 0, 0, 0, -x3_ob*x3_im, -x3_ob*y3_im],
    [0, 0, 0, x3_im, y3_im, 1, -y3_ob*x3_im, -y3_ob*y3_im],
    [x4_im, y4_im, 1, 0, 0, 0, -x4_ob*x4_im, -x4_ob*y4_im],
    [0, 0, 0, x4_im, y4_im, 1, -y4_ob*x4_im, -y4_ob*y4_im]
])

x_vec=np.array([[x1_ob],[y1_ob],[x2_ob],[y2_ob],[x3_ob],[y3_ob],[x4_ob],[y4_ob]])
invM=np.linalg.inv(M)
a = np.matmul(invM,x_vec)
print(x_vec)
print(a)
a1=a[0]
a2=a[1]
a3=a[2]
b1=a[3]
b2=a[4]
b3=a[5]
c1=a[6]
c2=a[7]

## End Init


## 
## read image
# im = skimage.io.imread('gletscher.jpg')
im = skimage.io.imread('schraegbild_tempelhof.jpg')

plt.figure()
plt.imshow(im)
im = im.astype('float')

im_new=rectification(im, a1, a2, a3, b1, b2, b3, c1, c2)
plt.figure()
im_new = im_new.astype('uint8')
plt.imshow(im_new)

# Bilder anzeigen
plt.show()
Test1=1



