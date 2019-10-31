from tkinter import *
import tkinter as tk
import tkinter as ttk
from PIL import ImageTk, Image
import os
from tkinter.filedialog import askopenfilename
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
global hubba
#global gg=0

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

def Classify(filename,hubba):
    img = cv2.imread(filename)
    hgt=img.shape[0]
    wdt=img.shape[1]
    hBw=hgt/float(wdt)
    dim = (576, int(576 * hBw))
    fram = img.copy()
    img=cv2.resize(img,dim)
    #if not img.shape():
    #	continue
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #cv2.imwrite("gray.jpg",gray)
    linek = np.zeros((11,11),dtype=np.uint8)
    linek[5,...]=1
    #cv2.imwrite("link.jpg",linek)
    x=cv2.morphologyEx(gray, cv2.MORPH_OPEN, linek ,iterations=1)
    #cv2.imwrite("x.jpg",x)
    gray-=x
    #cv2.imwrite("final.jpg",gray)
    #ret,gray = cv2.threshold(gray,10,255,cv2.THRESH_BINARY_INV)
    #gray = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #            cv2.THRESH_BINARY,11,2)
    kernel = np.ones((5,5), np.uint8)
    ret2,gray = cv2.threshold(gray,10,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    gray = cv2.dilate(gray, kernel, iterations=1) 
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
        if dect == 'Other_extended':
            cv2.rectangle(img, (start_x,start_y),(width+start_x,height+start_y),(0,255 ,255), 2)
        maskROI = fram[start_y:start_y+height, start_x:start_x+width]
        #cv2.imwrite(des2+str(tracker)+".png",maskROI)
        #tracker=tracker+1
        x=x+1
    #ret,gray = cv2.threshold(gray,10,255,cv2.THRESH_BINARY_INV)
    saveImage=img
    cv2.imwrite("temp.png",img)
    #im=hubba
    #gg=ImageTk.PhotoImage(Image.open("temp.png"))
    return hubba

import sys
from tkinter import * #or Tkinter if you're on Python2.7

def button1():
    novi = Toplevel()
    
    canvas = Canvas(novi, width = 300, height = 200)
    canvas.pack(expand = YES, fill = BOTH)
    gif1 = PhotoImage(file = 'image.gif')
                                #image not visual
    canvas.create_image(50, 10, image = gif1, anchor = NW)
    #assigned the gif1 to the canvas object
    canvas.gif1 = gif1

def openfile(saveImage):
    filename=filedialog.askopenfilename()
    print(filename)
    hubba=[[]]
    #global saveImage
    saveImage[0]=Classify(filename,hubba)
    print(hubba)

def savefile(saveImage):
    #global saveImage
    file = filedialog.asksaveasfilename()
    cv2.imwrite(file,cv2.imread("temp.png"))

mGui = Tk()
mGui.geometry("500x500")
mGui.title("Prescription Text Classification")
saveImage = [np.zeros((11,11),dtype=np.uint8)]
button = Button(mGui,text ='Choose an Prescription',command = lambda:openfile(saveImage), height=5, width=20).pack()
print(saveImage[0])
button2 = Button(mGui,text ='Save Image file\n after Classification',command = lambda:savefile(saveImage), height=5, width=20).pack()
mGui.mainloop()
