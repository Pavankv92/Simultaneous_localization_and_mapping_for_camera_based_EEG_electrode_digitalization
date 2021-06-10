#!/usr/bin/env python

import numpy as np
import cv2 as cv 
from matplotlib import pyplot as plt
import scipy.misc
import glob
from tqdm import tqdm_notebook as tqdm
import random
import imageio
import string
from matplotlib import pyplot as plt
from matplotlib import patches
from PIL import Image, ImageDraw


def read_images(img_dir, no_images):
    all_image_names = sorted(glob.glob(img_dir))
    # all_image_names = all_image_names[::10]
    all_depth_maps = []
    count = 0
    for img_name in all_image_names:
        image = Image.open(img_name)
        #image_np = np.asarray(image)
        count = count +1
        all_depth_maps.append(image)

        print "depth map " , count
        if count >= no_images:
            break

    return all_depth_maps

def unproject(points_px, intrinsic):
    f_x = intrinsic[0, 0]
    f_y = intrinsic[1, 1]
    c_x = intrinsic[0, 2]
    c_y = intrinsic[1, 2]
    # Step 2. Reproject.
    points_3d = np.zeros_like(points_px)
    for idx in range(points_px.shape[0]):
        z = points_px[idx,2] - 0.025
        x = (points_px[idx, 0] - c_x ) / f_x * z
        y = (points_px[idx, 1] - c_y ) / f_y * z
        points_3d[idx,:] = [x,y,z]
    return points_3d


def read_matrix(file_name):
    with open(file_name) as f:
        return np.loadtxt(f,delimiter=',')

def blobDetector(img, color_img):
    
    cv_image = cv.GaussianBlur(img,(5,5),0)
    cv_image = cv.medianBlur(cv_image,5)

    hsv = cv.cvtColor(color_img, cv.COLOR_BGR2HSV)
    h = hsv[:,:,2]
    h_fake = hsv
    #t = np.full_like(h,100)
    #print("shape",t.shape)
    h_fake[:,:,0] = h
    h_fake[:,:,1] = h
    
    params = cv.SimpleBlobDetector_Params()

    params.minThreshold = 50
    params.maxThreshold = 100
    params.thresholdStep = 5

    params.filterByColor = True
    
    params.filterByArea = True
    params.minArea = 40
    params.maxArea = 600

    params.filterByCircularity = True
    params.minCircularity = 0.6

    params.filterByConvexity = True
    params.minConvexity = 0.50

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv.SimpleBlobDetector_create(params)
    
    # Detect blobs.
    keypoints = detector.detect(cv_image)
    
    im_with_keypoints = cv.drawKeypoints(h_fake, keypoints, np.array([]), (0,0,255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)  
    #cv.imshow("keypoints", im_with_keypoints)
    #cv.waitKey(5000)
    return keypoints, im_with_keypoints         
        
        
def read_labels_yolo(file_name,all_depth_maps, intrinics):
    with open(file_name) as f:
        all_lines = [x.split() for x in f.read().splitlines()]
        point_cloud = []
        extra = 3
        for frame_idx in range(0,len(all_lines),5):
            line = all_lines[frame_idx]
            depth_map = all_depth_maps[frame_idx]
            #camera_pose = all_camera_poses[frame_idx]
            img_name = line[1]
            img_name = string.replace(img_name, "/mnt/", "/media/")
            print(img_name)
            cv_img = cv.imread(img_name, cv.IMREAD_COLOR)
            img =  cv.imread(img_name,cv.IMREAD_GRAYSCALE)
            #print("printing grey scale size:", np.shape(img))
            fig, ax = plt.subplots(1, figsize=(10, 10))
            
            if cv_img is None:
                continue

            mask = np.zeros(img.shape)
            points_bbox = []
            num_labels =  (len(line)-2)/5
            
            for i in range(num_labels):
                x_min = int(line[4+ i*5+1]) -extra
                y_min = int(line[4+ i*5+2]) -extra
                x_max = int(line[4+ i*5+3]) + extra
                y_max = int(line[4+ i*5+4]) + extra
                
                bbox = patches.Rectangle((x_min, y_min), (x_max-x_min), (y_max-y_min), linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(bbox)

                mask[y_min:y_max,x_min:x_max] = 1
                new_img = np.multiply(img,mask)
                new_img = new_img.astype(np.uint8)
                #new_img = new_img.astype(np.uint8)
                keypoints, im_with_keypoints  = blobDetector(new_img,cv_img)

                keypoints_px = np.zeros((len(keypoints),3))

                for i in range(len(keypoints)):
                    u = int(keypoints[i].pt[0])
                    v = int(keypoints[i].pt[1])
                    d =  depth_map.getpixel((u,v))
                    keypoints_px[i,:] = [u,v,d]
                
                keypoints_3d = unproject(keypoints_px, intrinics)
                
                for idx in range(keypoints_3d.shape[0]):
                    u = int(keypoints_px[idx,0])
                    v = int(keypoints_px[idx,1])
                    font = cv.FONT_HERSHEY_SIMPLEX
                    cv.putText(im_with_keypoints, "{:.2f}".format(keypoints_3d[idx,2]), (u,v), font, 1, (0, 255, 0), 1, cv.LINE_AA)
                    

                #plt.imshow(cv_img)
                plt.imshow(im_with_keypoints, alpha = 0.75)

            
            for i in range(len(keypoints_3d)):
                if not keypoints_3d[i,2] == 0:
                        points_bbox.append([ keypoints_3d[i,0],keypoints_3d[i,1], keypoints_3d[i,2] ]) 
            #print(np.shape(points_bbox))  
            #np.save("/home/Vishwanath/catkin_ws/src/master_arbeit/src/slam/src/data/CA_124_7/depth_variance/25/blob/electrode" +'_{}'.format(frame_idx),points_bbox)
            #print(points_bbox)
            #point_cloud.append(points_bbox)
            plt.show()
            #print((np.shape(point_cloud))) 
    return point_cloud               
            

if __name__ == "__main__":
    
    base_dir = "/home/Vishwanath/catkin_ws/src/master_arbeit/src/slam/src/data/robot_trajectory/CS_301_7/"
    dataset = '/media/pallando/share/FASOD/data/1009/CS-301/7/'
    yolo_labels = dataset + 'labels_1.26/yolo_labels.txt' 
    intrinsic_matrix = read_matrix(dataset + 'calibrations/camera_matrix.txt')
    all_depth_maps = read_images('/home/Vishwanath/temp/7/depth_imgs/*.jpg.tif',503)
    #yolo(yolo_labels)
    electrode_position = read_labels_yolo(yolo_labels,all_depth_maps, intrinsic_matrix)
    #np.save(base_dir + "depth_variance/14/blob/electrode_position_YOLO_blob_corrected.npy", electrode_position)


            