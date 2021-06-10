#!/usr/bin/env python
import cv2
import numpy as np
import glob
from natsort import natsorted
 
img_array = []

all_images = natsorted(glob.glob("/home/Vishwanath/Pictures/kinect_depth_analysis/*.jpg"))
for filename in all_images: 
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
 
out = cv2.VideoWriter('project.mp4',cv2.VideoWriter_fourcc(*'MP4V'), 15, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
