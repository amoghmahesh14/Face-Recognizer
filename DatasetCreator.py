#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 17:37:44 2019

@author: amogh
"""

import cv2
import numpy as np
        
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

id = input("Enter the User ID:")
sampleNum = 0
cam = cv2.VideoCapture(0)

while(True):
    ret,img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,1.3,5)
    for(x,y,width,height) in faces:
            sampleNum=sampleNum+1
            cv2.imwrite('Dataset/User.'+id+'.'+str(sampleNum)+'.jpeg',gray)
            cv2.rectangle(img,(x,y),(x+width,y+height),(0,255,0),2)
            cv2.waitKey(100)
    cv2.imshow("Face Recognizer",img)
    cv2.waitKey(1)
    if sampleNum>=10:
        break
cam.release()
cv2.destroyAllWindows()
