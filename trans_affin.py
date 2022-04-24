import dataclasses
import math
import matplotlib.pyplot as plt
import skimage
import numpy as np

def affin_trans(A,b,interp,im):
    
    d = np.sqrt([im.shape[0]*im.shape[0] + im.shape[1]*im.shape[1]]) 
    d = int(np.ceil(d) )    #round up and cast to int
    
    w = d #+b(1)                    # width of new image
    h = d #+b(0)                    # height of new image
    # w = im.shape[1]                 # width of new image
    # h = im.shape[0]                 # height of new image
    
    ## Calc edges of new image P00-P11
    ##      P00--------P01
    ##      |           |
    ##      |           |
    ##      P10--------P11
    
    # P00 =np.matmul(A,np.array([[0],[0]]))+b     
    # P01 =np.matmul(A, np.array( [[0],[im.shape[1]]]) )+b
    # P10 =np.matmul(A,np.array( [[im.shape[0]],[0]] ) )+b
    # P11 =np.matmul(A,np.array([[im.shape[0]], [im.shape[1]]]))+b
    # print('Eckpostionen waeren:')
    # print(P00)
    # print(P10)
    # print(P01)
    # print(P11)
    
    # transX = np.min( [P10[0],P01[0],P01[0],P10[0]] )    
    # tranyY = np.min( [P10[1],P01[1],P01[1],P10[1]] )
    # trans = np.array([[-transX],[tranyY]])    
    
    # print(np.abs(P10[0])+np.abs(P01[0]))

    im_new = np.zeros((h,w,im.shape[2])) # create new image
    im_new_buf = np.zeros((h,w,2)) # create 3d array with depth of two for coordinates 
    inv_A = np.linalg.inv(A)
    
    ## calc belonging pixel of original image -> write in im_new_buf to buffer belonging coordinate
    for i in range(w):
        for j in range(h):            
            x, y = np.matmul( inv_A, np.array([[j],[i]]) )  # x-height y-width | pixel of old image, float value, never hits perfect
            x=x-b[0]
            y=y-b[1]
            # j-height, i-width | pixel of new image and write in 3 dim matrix
            im_new_buf[j,i,:] = [x , y]
            ##print(im_new_buf[j,i,:])
    
    
    ## nearest neighbour:
    if interp==1:
        im_new_buf = np.ndarray.round(im_new_buf, decimals=0)
        for i in range(w):
            for j in range(h):
                [x, y] =  im_new_buf[j,i,:] 
                x =int(x)
                y=int(y)
                if 0<=x<im.shape[0] and 0<=y<im.shape[1]:
                    ##print(im[x,y,:]  )
                    im_new[j,i,:] = im[x,y,:]                    
                else:
                    im_new[j,i,:] = [0, 0, 0]

    ## bilinear interpolation
    elif interp==2:
        for i in range(w):
            for j in range(h):
                [x, y] =  im_new_buf[j,i,:] 
                test=1

                if 0<=x<im.shape[0] and 0<=y<im.shape[1]:

                    if 0<x<(im.shape[0]-1) and 0<y<(im.shape[1]-1): # Zwischenberich, nicht an einem Bildrand
                        decimal_x=math.modf(x)
                        decimal_y=math.modf(y)
                        upper_x =int(math.floor(x))
                        lower_x =int(math.ceil(x))
                        left_y =int(math.floor(y))
                        right_y =int(math.ceil(y))
                        ## Flaechen wie in Aufgabenblatt
                        A1=decimal_x[0]*decimal_y[0]
                        A2=decimal_x[0]*(1-decimal_y[0])
                        A3=(1-decimal_x[0])*decimal_y[0]
                        A4=(1-decimal_x[0])*(1-decimal_y[0])
                        ##print(sum([A1,A2,A3,A4]))
                        im_new[j,i,:] = A1*im[upper_x,left_y,:]+A2*im[upper_x,right_y,:]+A3*im[lower_x,left_y,:]+A3*im[lower_x,right_y,:]
                        if j>=389 and i>=298 and j<=391 and i<=300:
                            print(im_new[j,i,:])
                        
                    
                    ## Da keine Flaeche am Bildrand moeglich, wird nur die Laenge verwendet
                    # Randbereich y==0, linker Bildrand # Randbereich y==max, rechter Bildrand  
                    elif (0<x<im.shape[0] and y==0) or (y==(im.shape[1]-1) and 0<x<im.shape[0]): 
                        y=int(y)
                        decimal=math.modf(x)
                        upper_x =int(math.floor(x))
                        lower_x =int(math.ceil(x))
                        im_new[j,i,:] = decimal[0]*im[upper_x,y,:] + (1-decimal[0])*im[lower_x,y,:]
                    # Randbereich x==0, oberer Bildrand # Randbereich x==max, unterer Bildrand
                    elif  (0<y<im.shape[1]-1 and x==0) or (0<y<(im.shape[1]-1) and x==(im.shape[0]-1)): 
                        x=int(x)
                        decimal=math.modf(y)
                        left_y =int(math.floor(y))
                        right_y =int(math.ceil(y))
                        im_new[j,i,:] = decimal[0]*im[x,left_y,:] + (1-decimal[0])*im[x,right_y,:]
                    # Kante oben links # Kante oben rechts # Kante unten links # Kante unten recht
                    elif  (y==0 and x==0) or (y==(im.shape[1]-1) and x==0) or (y==0 and x==(im.shape[0]-1)) or (y==(im.shape[1]-1) and x==(im.shape[0]-1)): 
                        x=int(x)
                        y=int(y)
                        im_new[j,i,:] = im[x,y,:]      
                    else:
                        im_new[j,i,:] = [255, 0, 0]    
                else:
                    im_new[j,i,:] = [0, 0, 0]    
    return im_new

## read image
# im = skimage.io.imread('gletscher.jpg')
im = skimage.io.imread('ambassadors.jpg')

plt.figure()
plt.imshow(im)
im = im.astype('float')

## Aufgabe Gletscher
Interp=2
# testing 
# Verschieben, Translation
b = np.array([[0],[0]])

# Drehung, Rotation um 30°
alpha = -np.deg2rad(30)
A = np.array([[np.cos(alpha),-np.sin(alpha)],[np.sin(alpha),np.cos(alpha)]])

##Pixel 390, 299 ändert sich bei bildtypumwandlung

im_new = affin_trans(A,b,Interp,im)
im_new1 = im_new.astype('uint8')
plt.figure()
plt.imshow(im_new1, cmap='gray')

# # Stauchen/Strecken
# a=0.7
# A = np.array([[a,0],[0,a]])
# im_new = affin_trans(A,b,Interp,im)
# im_new2 = im_new.astype('uint8')
# plt.figure()
# plt.imshow(im_new2)

# # Stauchen/Strecken in x Richtung um 0.8
# A = np.array([[0.8,0],[0,1.2]])
# im_new = affin_trans(A,b,Interp,im)
# im_new3 = im_new.astype('uint8')
# plt.figure()
# plt.imshow(im_new3)

# Bilder anzeigen
plt.show()





# ## Schädel
# #
# Interp=1
# #testing 
# #Verschieben, Translation
# b = np.array([[200],[0]])

# # versch
# alpha = -np.deg2rad(30)
# a=1
# A = np.array([[np.cos(alpha),-np.sin(alpha)],[np.sin(alpha),np.cos(alpha)]])
# im_new = affin_trans(A,b,Interp,im)
# im_new1 = im_new.astype('uint8')
# plt.figure()
# plt.imshow(im_new1)

# # Stauchen/Strecken
# a=0.7
# A = np.array([[a,0],[0,a]])
# im_new = affin_trans(A,b,Interp,im)
# im_new2 = im_new.astype('uint8')
# plt.figure()
# plt.imshow(im_new2)

# # Stauchen/Strecken in x Richtung um 0.8
# A = np.array([[0.8,0],[0,1.2]])
# im_new = affin_trans(A,b,Interp,im)
# im_new3 = im_new.astype('uint8')
# plt.figure()
# plt.imshow(im_new3)

# # Bilder anzeigen
# plt.show()




