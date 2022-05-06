from cmath import pi
import dataclasses
import math
from turtle import onclick, window_height
import matplotlib.pyplot as plt
import skimage
import numpy as np
from PIL import Image, ImageFilter

## defines for functions

def bi_interpolate(im, x, y):
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1
    P1 = im[ y0, x0 ]
    P2 = im[ y1, x0 ]
    P3 = im[ y0, x1 ]
    P4 = im[ y1, x1 ]
    A1 = (x1-x) * (y1-y)
    A2 = (x1-x) * (y-y0)
    A3 = (x-x0) * (y1-y)
    A4 = (x-x0) * (y-y0)
    return A1*P1 + A2*P2 + A3*P3 + A4*P4

def rectification(im, a):
    # Matrix zur Bestimmung der Parameter a bis c
    # multiplikation mit inverser für lsg-vektor
    # M*a_vec=x_vec   
    a1=a[0]
    a2=a[1]
    a3=a[2]
    b1=a[3]
    b2=a[4]
    b3=a[5]
    c1=a[6]
    c2=a[7]

    h=im.shape[0]+700
    w=im.shape[1]+700
    # define new empty image
    im_new=np.zeros((h,w,im.shape[2]))

    for y_old in range(im_new.shape[1]): # y dir 
        for x_old in range(im_new.shape[0]): # x dir 
            denuminator=(b1*c2-b2*c1)*x_old+(a2*c1-a1*c2)*y_old+a1*b2-a2*b1
            ## gleich wie 1, 1 integrieren, modular aufbauen
            x_new=int( np.ndarray.round( ( (b2-c2*b3)*x_old+(a3*c2-a2)*y_old+a2*b3-a3*b2 ) / denuminator ) )
            y_new=int( np.ndarray.round( ( (b3*c1-b1)*x_old+(a1-a3*c1)*y_old+a3*b1-a1*b3 ) / denuminator ) )
            
            if 0<=x_new<im.shape[0]-1 and 0<=y_new<im.shape[1]-1:
                im_new[x_old,y_old,:]=im[x_new,y_new,:]

    return im_new

def calc_parameter(object_points, picture_points, pseudoInv):
    n_ob = len(object_points)
    n_im = len(picture_points)
    if n_ob!=n_im:
        print('False input parameter: lenght does not match')
        return False
    if pseudoInv=='No' or pseudoInv=='no':
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
        ## Automatisiert matrix M und vektor x_vec befüllen
        x_vec=np.array([[x1_ob],[y1_ob],[x2_ob],[y2_ob],[x3_ob],[y3_ob],[x4_ob],[y4_ob]])
        invM=np.linalg.inv(M)
        a = np.matmul(invM,x_vec)
        return a
    
    else: ## Use Pseudoinverse for n dimensional matrices (more object and image points!)
        # init with zeros, depended on number of picture and object points
        M = np.zeros((2*len(object_points),8))
        x = np.zeros((2*len(object_points),1))
        # fill matrix and vec in loop
        for point in range(len(object_points)):
            M[2*point]  =   [picture_points[point][0], picture_points[point][1], 1, 0, 0, 0, -object_points[point][0]*picture_points[point][0], -object_points[point][0]*picture_points[point][1]]
            M[2*point+1]=   [0, 0, 0, picture_points[point][0], picture_points[point][1], 1, -object_points[point][1]*picture_points[point][0], -object_points[point][1]*picture_points[point][1]]
            x[[2*point,2*point+1]]=[ [object_points[point][0]],[object_points[point][1]] ]
        pinv_M=np.linalg.pinv(M)
        a=np.matmul(pinv_M,x)
        return a

def weight(im):
    M=im.shape[0]
    N=im.shape[1]
    w = np.zeros([M,N])
    for i in range(M):
        for j in range(N):            
            if np.array_equal(im[i,j,:],[0.,0.,0.]):
                w[i,j]=0                
            else:
                w[i,j]=(1-2/M*abs(i-M/2))*(1-2/N*abs(j-N/2))
    return w

def fuse(ims, weights ,opt):
    width=ims[1].shape[0]
    height=ims[1].shape[1]
    im_fused=np.zeros((width,height,ims[1].shape[2]))
    if opt=='weight':
        for x in range(width):
            for y in range(height):
                if weights[0][x,y]>weights[1][x,y]:
                    im_fused[x,y,:]=ims[0][x,y,:]
                else:
                    im_fused[x,y,:]=ims[1][x,y,:]
    else:
        for x in range(width):
            for y in range(height):
                norm = weights[0][x,y]+weights[1][x,y]
                if norm !=0:
                    weight_0=weights[0][x,y]/norm
                    weight_1=weights[1][x,y]/norm
                    im_fused[x,y,:]=weight_0*ims[0][x,y,:]+weight_1*ims[1][x,y,:]
                else:
                    im_fused[x,y,:]=[0,0,0]
    return im_fused

def multi_band_blending(ims):
    im_tp=[]
    im_hp=[]
    weights=[]
    weights_tp=[]
    weights_hp=[]
    width=ims[1].shape[0]
    height=ims[1].shape[1]
    im_mbb=np.zeros((width,height,ims[1].shape[2]))
    im_tp[0]= ims[0].filter(ImageFilter.GaussianBlur)
    im_tp[1]= ims[1].filter(ImageFilter.GaussianBlur)
    im_hp[0]= ims[0]-im_tp[0]
    im_hp[1]= ims[1]-im_tp[1]
    weights_tp[0]=weight(im_tp[0])
    weights_tp[1]=weight(im_tp[1])

    im_mbb_tp=fuse(im_tp,weights_tp,'mix')
    im_mbb_hp=fuse(im_hp,weights_hp,'weight')
    weights[0]=weight(im_mbb_tp)
    weights[1]=weight(im_mbb_hp)
    im_mbb=fuse([im_mbb_tp,im_mbb_hp], weights,'mix')

    return im_mbb
## Init, picture and object points
# maps
# picture_points = [[338, 345],[432, 313],[335, 545],[423, 681]]
# object_points = [[100, 250], [657, 250], [100, 610], [657, 610]]
# test bild org Koord
# picture_points = [[70, 105],[219, 108],[4, 314],[272, 318]]
# object_points = [[0, 0], [210, 0], [0, 295], [210, 295]]
# test bild try Koord
# object_points = [[0, 0], [0, 210], [295, 0], [295, 210]]
# picture_points = [[105, 70],[108, 219],[314, 4],[318, 272]]

## Flagge x links -> rechts, y oben -> unten
# P8 0, 855
# P7 0,0
# P6 775,612
# P5 775,430
# P4 775,230
# P3 790,0
# P2 1490,855
# P1 1490,0
## Flagge_rechts
# P1 626,253
# P2 643,834
# P3 153,245
# P4 143,400
# P5 144,534
# P6 146,661
## Flagge_links
# P3 606 208
# P4 595 367
# P5 598 503
# P6 600 631
# P7 87 234
# P8 99 798 

## ## Task 3
## pic and obj points
## im 2
picture_points_im2 = [[253, 626],[834, 643],[245, 153],[400, 143],[534, 144],[661, 146]]
object_points_im2 = [[0, 1490], [855, 1490], [0, 790], [230, 775], [430, 775], [612, 775]]

## im 3
picture_points_im3 = [[208, 606],[367, 595],[503, 598],[631, 600],[234, 87],[798, 99]]
object_points_im3 = [[0, 790], [230, 775], [430, 775], [612, 775], [0, 0], [855, 0]]

## Calc vector a
a_vec_im2 = calc_parameter(object_points_im2, picture_points_im2, 'yes')
a_vec_im3 = calc_parameter(object_points_im3, picture_points_im3, 'yes')

## read image
# im = skimage.io.imread('gletscher.jpg')
# im = skimage.io.imread('schraegbild_tempelhof.jpg')
# im = skimage.io.imread('Test_Bild.png')
# plt.figure()
# plt.imshow(im)
# a_vec= calc_parameter(object_points, picture_points, 'no')
# im_new=rectification(im, a_vec)
# im_new = im_new.astype('uint8')
# plt.figure()
# plt.imshow(im_new)

## read images
im_2 = skimage.io.imread('Flagge_rechts.jpeg')
im_3 = skimage.io.imread('Flagge_links.jpeg')

plt.figure()
plt.imshow(im_2)
plt.figure()
plt.imshow(im_3)

## convert images
im_2 = im_2.astype('float')
im_3 = im_3.astype('float')

## Create top view
im_new_2=rectification(im_2, a_vec_im2)
im_new_2_plot = im_new_2.astype('uint8')
plt.figure()
plt.imshow(im_new_2_plot)
im_new_3=rectification(im_3, a_vec_im3)
plt.figure()
im_new_3_plot = im_new_3.astype('uint8')
plt.imshow(im_new_3_plot)

## Fuse image
# calc weights
w_im_2=weight(im_new_2)
w_im_3=weight(im_new_3)
# with higher weight
im_fused_w=fuse([im_new_2,im_new_3],[w_im_2,w_im_3],'weight')
im_fused_w=im_fused_w.astype('uint8')
plt.figure('larger weight')
plt.imshow(im_fused_w)
# with mix weight
im_fused_m=fuse([im_new_2,im_new_3],[w_im_2,w_im_3],'m')
im_fused_m=im_fused_m.astype('uint8')
plt.figure('weighted sum')
plt.imshow(im_fused_m)
# multi
im_fused_multi=fuse([im_new_2,im_new_3],[w_im_2,w_im_3],'m')
im_fused_multi=im_fused_multi.astype('uint8')
plt.figure('multi band')
plt.imshow(im_fused_multi)
## Bilder anzeigen
plt.show()
Test1=1



