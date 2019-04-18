#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 19:29:57 2019

@author: amogh
"""

import cv2
import numpy as np
import os
from PIL import *


recognizer = cv2.face.LBPHFaceRecognizer_create()
path='Dataset' #Path of the folder where the Training Images are stored

def getIDsandImages(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    #Gets the complete path of each path
    #listdir() returns a list of files or directories in the path specified
    faces = []
    IDs = []
    
    for imagePath in imagePaths:
        faceImg = Image.open(imagePath).convert('L')
        #convering an image to greyscale if it wasn't previously done
        faceNp = np.array(faceImg,'uint8')
        #convering the imagea into matrix
        ID = int(os.path.split(imagePath)[-1].split('.')[1])
        #Getting the ID from thr image name
        # -1 indicates traversing the name from the back
        # [1] indicates the specific string after splitting the name
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("Trainer",faceNp)
        cv2.waitKey(10)
    return np.array(IDs),faces

IDs,faces = getIDsandImages(path)
recognizer.train(faces,IDs)
recognizer.write('trainingData.yml')
cv2.destroyAllWindows()

        
    