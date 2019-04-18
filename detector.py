#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 20:44:19 2019

@author: amogh
"""

import cv2
import numpy as np
#id=0      
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read('trainingData.yml')

fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 0.5
fontcolor = (0, 0, 255)
while(True):
    ret,img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceDetect.detectMultiScale(gray,1.3,5)
    for(x,y,width,height) in faces:
            cv2.rectangle(img,(x,y),(x+width,y+height),(0,255,0),1)
            #(x,y) is the starting point of the circle , (x,y) correspond to (0,0) cell in the matrix
            #(x+width,y+height) is the ending point of the circle , this is (m,n) cell in the matrix
            #(0,255,00) -> (R,G,B) gives color
            # 2 is the thickness
            id,conf=rec.predict(gray[y:y+height,x:x+width])
            #cv2.PutText(cv2.cv.fromarray(img),str(id),(x,y+height),font,255)
            if (id==1):
                id = "Amogh"
      
            cv2.putText(img, str(id), (x,y+height), fontface, fontscale, fontcolor) 
    cv2.imshow("Face Detector",img)
    if(cv2.waitKey(1)==ord('q')):
        break
cam.release()
cv2.destroyAllWindows()
