#!/usr/bin/env python

import cv2
import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d
import glob
from natsort import natsorted
import copy

"""
dataset = '/media/pallando/share/FASOD/data/1009/CA-124/8/'

img = dataset + 'imgs/img73148.jpg'

img = cv2.imread(img,0)
img = cv2.medianBlur(img,7)

ret,th1 = cv2.threshold(img,150,255,cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,11,2)
th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)

titles = ['Original Image', 'Global Thresholding (v = 127)',
            'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
images = [img, th1, th2, th3]

for i in xrange(4):
    plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
    plt.title(titles[i])
    plt.xticks([]),plt.yticks([])
plt.show()


# Set up the detector with default parameters.
im=cv2.bitwise_not(img)

params = cv2.SimpleBlobDetector_Params()




detector = cv2.SimpleBlobDetector_create(params)


# Detect blobs.
keypoints = detector.detect(im)
im=cv2.bitwise_not(im)
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)
"""
def getElectrodes(folder_path):
    all_scans = glob.glob(folder_path)
    xyz_points_list = []
    for scan in natsorted(all_scans) :   
        xyz_points = np.load(scan, allow_pickle=True)
        #xyz_points = xyz_points/1000 # dataset is in mm
        xyz_points_list.append(xyz_points)
    return xyz_points_list

if __name__ == "__main__":

    base_dir = "/home/Vishwanath/catkin_ws/src/master_arbeit/src/slam/src/data/CA_124_7/"
    camera =  base_dir + "results/ground_truth_camera_pose_first_frame.npy"
    electrode = base_dir + "depth_variance/14/blob/all_electrodes_list.npy"
    folder_path = base_dir + "depth_variance/14/blob/*.npy"
    

    #print(np.shape(electrode))
    #pcd_1 = o3d.geometry.PointCloud()
    #pcd_1.points = o3d.utility.Vector3dVector(electrode)
    
    ground_truth_camera_poses = np.load(str(camera), allow_pickle=True)
    #ground_truth_electrodes = np.load(str(electrode), allow_pickle=True)
    ground_truth_electrodes = getElectrodes(folder_path)

    electrode_position_list = o3d.geometry.PointCloud()
    electrode_position = o3d.geometry.PointCloud()
    electrode_Transformed = o3d.geometry.PointCloud()
    
    for i in range(0, 100, 1):
    
    #for i in range(len(ground_truth_camera_poses)):
        try:
            point = np.load(base_dir + "depth_variance/14/blob/electrode_{}.npy".format(i), allow_pickle=True)
        except IOError:
            continue
        electrode_position.points = o3d.utility.Vector3dVector(point)
        electrode_Transformed = copy.deepcopy(electrode_position).transform(ground_truth_camera_poses[i])
        electrode_position_list = electrode_position_list + electrode_Transformed
    o3d.visualization.draw_geometries([electrode_position_list]) 

    


