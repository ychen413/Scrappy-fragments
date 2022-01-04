# -*- coding: utf-8 -*-
"""
Created on Wed May 29 18:58:29 2019

@author: deep362
"""

import numpy as np
#import matplotlib.pyplot as plt
#import pandas as pd
import cv2
import glob
import os

def show_imag(imag):
    cv2.imshow('imag', imag)
    cv2.waitKey(0)
    pass

def resized_imag(imag, scale_x = 0.2, scale_y = 0.2):
    return cv2.resize(imag, (0,0), fx = scale_x, fy = scale_y)

def crop_imag(imag, x, y, w, h):
    return imag[y:y+h, x:x+w]

def EKGs_extract(file_path, threshold=400):
    crop_height = 190
    img = cv2.imread(file_path)[crop_height:-1, :]
    h, w = img.shape[0:2]
    mask = np.array([np.sum(img, axis=2) > threshold]).reshape(h, w)
    img[mask, :] = 255
    img_ = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#    show_imag(img_)
    return img_

# =================================
# Delete specific files in many folders:
#for doc in glob.iglob("D:/E/Research_USA/dataset/Pwave_project_sample_EKGs_after/*"):
#    for images in glob.iglob(doc + "/*.png"):
#        os.remove(images)
# =================================

#loc = 'D:/E/Research_USA/dataset/Pwave_project_sample_EKGs/3/8.24.2008_1019.PNG'
#images = [cv2.imread(file) for file in glob.glob("D:/E/Research_USA/dataset/Pwave_project_sample_EKGs/*/*.png")]
#path = glob.glob("D:/E/Research_USA/dataset/Pwave_project_sample_EKGs/*")
#img = EKGs_extract(loc)
#cv2.imwrite('D:/E/Research_USA/dataset/Pwave_project_sample_EKGs/3/test.PNG', img)

for doc in glob.iglob("D:/E/Research_USA/dataset/Pwave_project_sample_EKGs/*"):
    for images in glob.iglob(doc + "/*.png"):
        img = EKGs_extract(images)
        path = images.replace('Pwave_project_sample_EKGs', 'Pwave_project_sample_EKGs_after')
        cv2.imwrite(path, img)
        
