import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import scipy.misc
import pytesseract
import expeval
from expeval import dothings

WHITE = [255,255,255]

def getimg(im):
    imag=cv2.imread(im)
    img_gray = cv2.cvtColor(imag,cv2.COLOR_BGR2GRAY)
    kernel = np.ones((3,3),np.uint8)         
    thresh = cv2.bitwise_not(img_gray) 
                              
    dilation = cv2.dilate(thresh,kernel,iterations = 1)
    di = cv2.bitwise_not(dilation)

    
    gim=Image.open(im).convert('RGB')
    wo,ho=gim.size
    area=wo*ho*(1.0)
 
    img_rgb = cv2.cvtColor(di,cv2.COLOR_GRAY2BGR)  
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    j=0
    para=((0.00002)*area)
    bloo=[]
    for cnt in contours:
	x,y,w,h = cv2.boundingRect(cnt)
        ar=cv2.contourArea(cnt)
        if ar>para:
	    bloo.append((x,y,w,h))
            bbox=(x,y-2,x+w,y+h+2)
            cropped = gim.crop(bbox)
	    img1 = np.asarray(cropped)
            constant= cv2.copyMakeBorder(img1,5,5,5,5,cv2.BORDER_CONSTANT,value=WHITE)
	    scipy.misc.imsave('cropped'+str(j)+'.png',constant)
  	    j=j+1

    scipy.misc.imsave('dotted.png',gim)
    simpleway(bloo,3)    ##special case of 3 by 3 taken
    #calc(bloo)

    	
def simpleway(clist,numsquare):

    leng=len(clist)
    diffe=[]
    l=[[] for i in range(numsquare)]
    h=0
    clist.sort(key=lambda tup:tup[1])

    for i in range(leng-1):
        diff=clist[i+1][1]-clist[i][1]
	diffe.append(diff)
    twodiff=sorted(diffe)

    for i in range(leng-1):
	if diffe[i] in twodiff[-(numsquare-1):]:
	    l[h].append(clist[i])
	    h=h+1
	else:
	    l[h].append(clist[i])
    l[h].append(clist[leng-1])

    #print l	
    frag=[[[] for j in range(numsquare)] for i in range(numsquare)]


    for i in range(numsquare):
        diffx=[]
        c=0
        l[i].sort(key=lambda tup:tup[0])
	ko=len(l[i])
	for k in range(ko-1):
            diff=l[i][k+1][0]-l[i][k][0]
	    diffx.append(diff)
        twodiffx=sorted(diffx)

        for f in range(ko-1):
	    if diffx[f] in twodiffx[-(numsquare-1):]:
	        #print diffx[f]
	        #print ('yes greater2')
	        frag[i][c].append(l[i][f])
	        c=c+1
	    else:
	        frag[i][c].append(l[i][f])
        frag[i][c].append(l[i][ko-1])
    
    #for d in range(numsquare):
	#print frag[d]
    return frag

			
def calc(clist,model='det',prov='manual',num=3):     ##special case of 3 by 3 taken,can be changed later
    if model=='det' and prov=='manual':
	l=[[] for i in range(num)]
	j=0
        firstbox=clist[0]
	for c in clist:
	    print c[1]
	    if (firstbox[1]-40)<=c[1]<=(firstbox[1]+40):
		l[j].append(c)
                continue
	    else:
		firstbox=c
		j=j+1
		l[j].append(c)
        print l
    else:
	print('boo')
    gro=len(l)
    for r in range(gro):
        l[r].sort(key=lambda tup:tup[0])
    frag=[[[] for i in range(3)] for i in range(4)]
    mo=len(l)
    for m in range(mo):
        lo=len(l[m])
	fust=l[m][0]
	g=0
	for n in range(lo):
	    if l[m][n][0]-fust[0]-fust[2]<=100:
		frag[m][g].append(l[m][n])
		continue
	    else:
		fust=l[m][n]
		g=g+1
		frag[m][g].append(l[m][n])
    print frag


def numconvert(frag,num,imglist):
    intlist=[[] for i in range(num)]
    for y in range(num):
	for nu in range(num):
	    ro=len(frag[y][nu])
	    text=''
	    for t in range(ro):
                text = text + pytesseract.image_to_string(imglist[y][nu][t],'English',10,'True')   ##will not work,method to be decided for efficient running
	    actual=int(text)
	    intlist[y].append(actual)  
    return intlist 
    #return expeval.det(intlist)

