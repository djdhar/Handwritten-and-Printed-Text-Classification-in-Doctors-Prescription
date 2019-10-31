import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir('/home/dj/Downloads/Prescription_Samples/C') if isfile(join('/home/dj/Downloads/Prescription_Samples/C', f))]
des = '/home/dj/Downloads/Prescription_Samples/D/'
des2 = '/home/dj/Downloads/Prescription_Samples/E/'
tracker=0
for s in onlyfiles:
	try:
		#s = 'sample6.jpg'
		img = cv2.imread('/home/dj/Downloads/Prescription_Samples/C/'+s)
		fram = img.copy()
		#if not img.shape():
		#	continue
		gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
		cv2.imwrite("gray.jpg",gray)
		linek = np.zeros((11,11),dtype=np.uint8)
		linek[5,...]=1
		cv2.imwrite("link.jpg",linek)
		x=cv2.morphologyEx(gray, cv2.MORPH_OPEN, linek ,iterations=1)
		cv2.imwrite("x.jpg",x)
		gray-=x
		cv2.imwrite("final.jpg",gray)
		#ret,gray = cv2.threshold(gray,10,255,cv2.THRESH_BINARY_INV)
			#gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
			#            cv2.THRESH_BINARY,11,2)
		kernel = np.ones((5,5), np.uint8)
		ret2,gray = cv2.threshold(gray,10,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		gray = cv2.dilate(gray, kernel, iterations=1) 
		contours2, hierarchy = cv2.findContours(gray,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
		x=0
		while x<len(contours2):
    			(start_x,start_y,width,height)= cv2.boundingRect(contours2[x])
    			cv2.rectangle(img, (start_x,start_y),(width+start_x,height+start_y),(255,0 , 0), 2)
    			maskROI = fram[start_y:start_y+height, start_x:start_x+width]
    			cv2.imwrite(des2+str(tracker)+".png",maskROI)
    			tracker=tracker+1
    			x=x+1
		ret,gray = cv2.threshold(gray,10,255,cv2.THRESH_BINARY_INV)
		#cv2.imshow('gray',gray)
		cv2.imwrite(des+'res'+s,img)
		#cv2.waitKey(0)
	except:
		print(s)
