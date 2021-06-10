#!/usr/bin/env python

import numpy as np
import cv2
import glob
from PIL import Image, ImageDraw
from time import sleep
import matplotlib.pyplot as plt
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud2, PointField, PointCloud
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from std_msgs.msg import Header
from geometry_msgs.msg import Pose 
from geometry_msgs.msg import TransformStamped, Vector3Stamped
import open3d as o3d 
from open3d_helper import convertCloudFromOpen3dToRos, convertCloudFromRosToOpen3d
import rospy
import tf_conversions.posemath as pm
from conversions import ros2np_Pose
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import struct
from numpy import linalg as LA


def read_matrix(file_name):
    #to read calibration matrix
    with open(file_name) as f:
        return np.loadtxt(f,delimiter=',')


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

def get_3d_points_from_yolo_labels(file_name_yolo_labels, path_to_save_electrode_file, all_depth_maps, intrinics_matrix, depth_offset):
    all_electrode_positions= []
    all_yolo_labels = []
    with open(str(path_to_save_electrode_file), "w") as yolo_write:
        with open(file_name_yolo_labels) as f:
            all_lines = [x.split() for x in f.read().splitlines()]
            for frame_idx in range(len(all_lines)):
                line = all_lines[frame_idx]
                depth_map = all_depth_maps[frame_idx]
                yolo_write.write(str(line[1]))
                yolo_write.write(" ")
                points = []
                yolo_labels=[]
                num_labels =  (len(line)-2)/5
                for i in range(num_labels):
                    x_min = float(line[4+ i*5+1])
                    y_min = float(line[4+ i*5+2])
                    x_max = float(line[4+ i*5+3])
                    y_max = float(line[4+ i*5+4])
                    #print x_min, y_min, x_max, y_max
                    x_mean = (x_min + x_max) / 2.
                    y_mean = (y_min + y_max) /2.
                    #print x_mean, y_mean

                    depth = depth_map.getpixel((x_mean,y_mean))
                    if depth==0 or depth > 0.6:
                        continue
                    depth = depth - depth_offset
                    X = (x_mean-intrinics_matrix[0,2]) / intrinics_matrix[0,0] * depth
                    Y = (y_mean-intrinics_matrix[1,2]) / intrinics_matrix[1,1] * depth
                    Z = depth

                    points.append([X, Y, Z, 1.0])
                    yolo_write.write("{}".format(X))
                    yolo_write.write(" ")
                    yolo_write.write("{}".format(Y))
                    yolo_write.write(" ")
                    yolo_write.write("{}".format(Z))
                    yolo_write.write(" ")
                    yolo_write.write("{}".format(1.0))
                    yolo_write.write(" ")
                    yolo_labels.append([x_mean, y_mean])
                
                all_yolo_labels.append(yolo_labels)
                all_electrode_positions.append(points)
                yolo_write.write("\n")
           
    yolo_write.close()
    return all_electrode_positions, all_yolo_labels


def readElectrodesFromTextFile(path_to_text_file, path_to_save):
    electrode_raw_list = []
    with open(str(path_to_text_file), 'r') as reader:
        for line in reader:
            electrode_raw_list.append(line)
    
    electrode_position_list = []
    #print(len(electrode_raw_list))
    for i in range (len(electrode_raw_list)):
        read_line = electrode_raw_list[i]
        #[2:len(read_line)] for reading from tracking camera file and [1:len(read_line)] for reading yolo labels
        stripped_line = read_line.split()[1:len(read_line)]
        electrode_position = []
        
        for k in range (len(stripped_line)):
            if stripped_line[k] == '1.0':
                electrodes_xyz = [ float(stripped_line[k-3]), float(stripped_line[k-2]), float(stripped_line[k-1]) ]
                #print(electrodes_xyz)
                electrode_position.append(electrodes_xyz)
        
        electrode_position_list.append(electrode_position)
    
    np.save(str(path_to_save), electrode_position_list)


if __name__ == "__main__":
    
    base_dir = '/media/pallando/share/students/Vishwanath/master_arbeit_data/hand_trajectory/static_phantom_head/CA_124/6/'
    
    no_depth_imgs = len(glob.glob(base_dir + 'depth_imgs/*.jpg.tif'))
    print "reading depth maps"
    all_depth_maps = read_images(base_dir + 'depth_imgs/*.jpg.tif', no_depth_imgs)
    print "done reading depth maps"
    #all_imgs = read_images(dataset + 'imgs/*.jpg',1408)
    intrinics_matrix = read_matrix(base_dir + 'calibrations/camera_matrix.txt')
    yolo_labels = base_dir + "yolo_labels.txt"
    
    #path_to_save = base_dir + "/depth_variance/21/yolo_depth_3d_points.txt"
    
    #get_3d_points_from_yolo_labels(yolo_labels, path_to_save, all_depth_maps, intrinics_matrix, 0.025)
    
    #"""
    # forloop stuff
    depth_offset = [x for x in np.arange(0, 30, 1)]

    for i in range(len(depth_offset)):
        depth = depth_offset[i]
        print(float(depth)/1000)
        path_to_save = base_dir + "depth_variance/{}/yolo_depth_3d_points.txt".format(depth)
    
        get_3d_points_from_yolo_labels(yolo_labels, path_to_save, all_depth_maps, intrinics_matrix, (float(depth)/1000))
        electrode_positions_yolo = base_dir + "depth_variance/{}/electrode_position_from_text_file.npy".format(depth)
        readElectrodesFromTextFile(path_to_save, electrode_positions_yolo)
    #"""
    

    

    
    





