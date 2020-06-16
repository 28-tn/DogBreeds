#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 17:24:49 2020

@author: tn
"""

import cv2
import urllib
import numpy as np

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('../haarcascades/haarcascade_frontalface_alt.xml')

def face_detector(img_path):
    #img = cv2.imread(img_path)
    req = urllib.request.urlopen(img_path)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    img = cv2.imdecode(arr, -1) # 'Load it as it is'
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0