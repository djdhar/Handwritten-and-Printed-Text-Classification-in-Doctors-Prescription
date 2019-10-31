import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
folder_name='Printed_extended'
onlyfiles = [f for f in listdir('/home/dj/Downloads/Prescription_Samples/B_/'+folder_name) if isfile(join('/home/dj/Downloads/Prescription_Samples/B_/'+folder_name, f))]
for s in onlyfiles:
    img = cv2.imread("/home/dj/Downloads/Prescription_Samples/B_/"+folder_name+"/"+s)
    rows = img.shape[0]
    cols = img.shape[1]
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print(rows)
    print(cols)
    print(rows/cols)
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
    print(myavg)
    change=0
    for xx in range (0,rows):
        mycnt=0
        for yy in range (0,cols-1):
            if bwMask[xx:yy].all()!=bwMask[xx:yy+1].all():
                mycnt=mycnt+1
        change=change+(mycnt*1.0)/cols
    change=change/(rows)
    print(change)
    file = open('res2.csv','a')
    file.write(str(rows)+","+str(cols)+","+str(rows/cols)+","+str(myavg)+","+str(change)+","+folder_name+"\n") 
