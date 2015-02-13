#!/usr/bin/python2.7
import cv2
from PIL import Image
from matplotlib import pyplot as plt
from collections import namedtuple
from PIL import ImageEnhance
import scipy.misc
import numpy as np

def cont(img):

    im = Image.open(img)  
    wo,ho=im.size
    area=wo*ho*(1.0)   
    enh = ImageEnhance.Contrast(im)
    yes=enh.enhance(1.1)
    Im = np.asarray(yes)

    img_gray = cv2.cvtColor(Im,cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(img_gray,100,255,cv2.THRESH_BINARY)

    kernel = np.ones((3,3),np.uint8)         
    thresh = cv2.bitwise_not(thresh1)                           
    dilation = cv2.dilate(thresh,kernel,iterations = 1)
    di = cv2.bitwise_not(dilation)

    scipy.misc.imsave("thresh1.png",thresh1)
    scipy.misc.imsave("threshdilate.png",di)

    image=Image.fromarray(di).convert('RGB')

    plt.subplot(211),plt.imshow(image)
    plt.show()

    pixa=image.load()

    plt.subplot(211),plt.imshow(image)
    plt.show()

    ######convert ('RGB') ????!!!!?!!!??!

    contours, hierarchy = cv2.findContours(dilation,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    img_rgb = cv2.cvtColor(di,cv2.COLOR_GRAY2BGR)  
    cv2.drawContours(img_rgb,contours,-1,(0,250,0),2)


    #print len(contours)
    para=((0.00002)*area)
    for cnt in contours:
	ar=cv2.contourArea(cnt)
        print ar
        if ar<para:
            x,y,w,h = cv2.boundingRect(cnt)
            for g in range(w):
                for t in range(h):
                    if cv2.pointPolygonTest(cnt,(x+g,y+t), False)>=0:
                        pixa[x+g,y+t]=(255,255,255) 

    #plt.subplot(211),plt.imshow(dilation)
    #plt.show()   
    scipy.misc.imsave("dilinvert.png",dilation)

    #plt.subplot(211),plt.imshow(image)
    #plt.show()   
    scipy.misc.imsave("threshdilatepil.png",image)

