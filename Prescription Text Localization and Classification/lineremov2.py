import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from joblib import dump, load
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from joblib import dump, load
import pickle
import pdb

def tester(clf,mpred):
	clf = load("data.joblib")
	#mpred = [[40,113,0.353982300884956,	0.453097345132743,	0.008849557522124]]
	dj = clf.predict(mpred)
	print(dj[0])
	return dj[0]

def checker(img):	
	#img = cv2.imread("/home/dj/Downloads/Prescription_Samples/B_/"+folder_name+"/"+s)
	arr=[]
	dj=[]
	rows = img.shape[0]
	cols = img.shape[1]
	img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
	#print(rows)
	#print(cols)
	#print(rows/cols)
	arr.append(rows)
	arr.append(cols)
	arr.append(rows/cols)
	retval,bwMask =cv2.threshold(img, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
	mycnt=0
	myavg=0
	for xx in range (0,cols):
		mycnt=0
		for yy in range (0,rows):
			if bwMask[yy,xx]==0:
				mycnt=mycnt+1
				
		myavg=myavg+(mycnt*1.0)/rows
	myavg=myavg/cols
	#print(myavg)
	arr.append(myavg)
	change=0
	for xx in range (0,rows):
		mycnt=0
		for yy in range (0,cols-1):
			if bwMask[xx:yy].all()!=bwMask[xx:yy+1].all():
				mycnt=mycnt+1
		change=change+(mycnt*1.0)/cols
	change=change/(rows)
	#print(change)
	arr.append(change)
	#file = open('res2.csv','a')
	#file.write(str(rows)+","+str(cols)+","+str(rows/cols)+","+str(myavg)+","+str(change)+","+folder_name+"\n") 
	dj.append(arr)
	return dj



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
		cv2.imwrite("final.jpg",gray)
		ret2,gray = cv2.threshold(gray,10,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		cv2.imwrite("otsu.jpg",gray)
		gray = cv2.dilate(gray, kernel, iterations=1)
		cv2.imwrite("dialate.jpg",gray) 
		contours2, hierarchy = cv2.findContours(gray,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
		x=0
		clf = load("data.joblib")
		while x<len(contours2):
    			(start_x,start_y,width,height)= cv2.boundingRect(contours2[x])
    			mymat = img[start_y:start_y+height, start_x:start_x+width]
    			dect = tester(clf,checker(mymat))
    			#print(dect)
    			if dect == 'Printed_extended':
    				cv2.rectangle(img, (start_x,start_y),(width+start_x,height+start_y),(255,0 , 0), 2)
    			if dect == 'Handwritten_extended':
    				cv2.rectangle(img, (start_x,start_y),(width+start_x,height+start_y),(0,255 , 0), 2)
    			if dect == 'Mixed_extended':
    				cv2.rectangle(img, (start_x,start_y),(width+start_x,height+start_y),(0,0 ,255), 2)
    			maskROI = fram[start_y:start_y+height, start_x:start_x+width]
    			cv2.imwrite(des2+str(tracker)+".png",maskROI)
    			tracker=tracker+1
    			x=x+1
		ret,gray = cv2.threshold(gray,10,255,cv2.THRESH_BINARY_INV)
		#cv2.imshow('gray',gray)
		cv2.imwrite(des+'res'+s,img)
		print(s)
		#cv2.waitKey(0)
	except:
		print(Exception)
