#!/usr/bin/env python

import numpy as np
import cv2
import glob
from natsort import natsorted
from PIL import Image, ImageDraw
from time import sleep
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sensor_msgs.point_cloud2 as pcl2
from sensor_msgs.msg import PointCloud2, PointField
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from std_msgs.msg import Header
from geometry_msgs.msg import Pose
from geometry_msgs.msg import TransformStamped, Vector3Stamped
from ICP_open3D import draw_registration_result_corres
import rospy
import tf_conversions.posemath as pm
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import struct
from numpy import linalg as LA
import ros_numpy
import open3d as o3d
from open3d_helper import convertCloudFromOpen3dToRos, convertCloudFromRosToOpen3d
from pcViewSet import *


#def 

def read_image(img_path):

    image = Image.open(img_path)
    image_np = np.asarray(image)

    return image_np

def read_images(img_dir, no_images):
    all_image_names = natsorted(glob.glob(img_dir))
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


def get_camera_poses(file_name, num_poses, T_link78,T_ee_cam):
    with open(file_name) as f:
        traj = []
        for i in range(num_poses):
            line = np.loadtxt(f,delimiter=' ',max_rows=1)
            pose = line[2:]
            pose = np.reshape(pose,[4,4])
            pose = np.matmul(pose, T_link78)
            pose = np.matmul(pose, T_ee_cam)
            T_cam_offset = np.eye(4)
            T_cam_offset[2,3] = -0.00
            pose = np.matmul(pose, T_cam_offset)
            traj.append(pose)
    return traj


def get_3d_points_from_ground_truth(file_name_electrode_positions):
    all_electrode_positions = []
    with open(file_name_electrode_positions) as f:
        all_lines = [x.split() for x in f.read().splitlines()]
        for frame_idx in range(len(all_lines)):
            line = all_lines[frame_idx]
            num_electrodes =  (len(line)-2)/4

            points_in_frame = []
            for i in range(num_electrodes):
                x = float(line[2+ i*4+0])
                y = float(line[2+ i*4+1])
                z = float(line[2+ i*4+2])
                points_in_frame.append([x,y,z])
            all_electrode_positions.append(points_in_frame)
            #if frame_idx >= 0:
                #break
    return all_electrode_positions


def get_3d_points_from_yolo_labels(file_name_yolo_labels, all_depth_maps, intrinics_matrix, depth_offset):
    all_electrode_positions= []
    all_yolo_labels = []
    with open(file_name_yolo_labels) as f:
        all_lines = [x.split() for x in f.read().splitlines()]
        for frame_idx in range(len(all_lines)):
            line = all_lines[frame_idx]
            depth_map = all_depth_maps[frame_idx]
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
                    #points.append([0, 0, 0])
                    continue
                #depth = depth - 0.027
                depth = depth - depth_offset
                X = (x_mean-intrinics_matrix[0,2]) / intrinics_matrix[0,0] * depth
                Y = (y_mean-intrinics_matrix[1,2]) / intrinics_matrix[1,1] * depth
                Z = depth

                points.append([X, Y, Z])
                yolo_labels.append([x_mean, y_mean])
            all_yolo_labels.append(yolo_labels)
            all_electrode_positions.append(points)
            #if frame_idx >= 0:
                #break
    return all_electrode_positions, all_yolo_labels

def transform_pointlist_to_base(pointlist, all_camera_poses):
    all_pc_np = []
    for i in range(len(pointlist)):

        header = Header()
        header.frame_id = '/map'
        header.stamp = rospy.Time.now()
        pointcloud = pcl2.create_cloud_xyz32(header, np.asarray(pointlist[i]))


        camera_pose = all_camera_poses[i]
        a = pm.fromMatrix(camera_pose)
        pose_msg = pm.toMsg(a)
        tf_stamped = TransformStamped()
        tf_stamped.transform.translation = pose_msg.position
        tf_stamped.transform.rotation = pose_msg.orientation
        electrode_pc_base = do_transform_cloud(pointcloud, tf_stamped)

        all_pc_np.append(ros_numpy.point_cloud2.pointcloud2_to_xyz_array(electrode_pc_base))

    stacked_pc_np = np.vstack(all_pc_np)
    #print("shape: ", stacked_pc_np.shape)

    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    ]

    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = "/map"

    pc_base = pcl2.create_cloud(header, fields, stacked_pc_np)
    return pc_base


def read_matrix(file_name):
    with open(file_name) as f:
        return np.loadtxt(f,delimiter=',')

def create_open3d_cloud_from_np_array(np_array):
    open3d_point_cloud_list = []
    for i in range(len(np_array)):
        current_point_cloud = o3d.geometry.PointCloud()
        current_point_cloud.points = o3d.utility.Vector3dVector(np_array[i])
        open3d_point_cloud_list.append(current_point_cloud)
    return open3d_point_cloud_list

def publish_electrodes_yolo_base(path_to_pointCloud2, pub):
    all_scans = glob.glob(path_to_pointCloud2)
    print("publishing....pc_yolo_base")
    rate = rospy.Rate(10)
    while not rospy.is_shutdown() :
        for scan in natsorted(all_scans) :   
            scan = np.load(str(scan), allow_pickle=True)
            pub.publish(scan)
        rate.sleep()


if __name__ == "__main__":
   # rospy.init_node("data_extraction_for_pavan")
    vSet = pcViewSet()
    pc_yolo_pub = rospy.Publisher('electrode_positions_yolo', PointCloud2, queue_size=1)
    pc_ground_truth_pub = rospy.Publisher('electrode_positions_ground_truth', PointCloud2, queue_size=1)

    pc_yolo_base_pub = rospy.Publisher('pc_yolo_base', PointCloud2, queue_size=10)
    tf_pub = rospy.Publisher('camera_pose', Pose, queue_size=10)
    base_dir = "/home/Vishwanath/catkin_ws/src/master_arbeit/src/slam/src/data/robot_trajectory/CS_301_8/"
    dataset = '/media/pallando/share/FASOD/data/1009/CS-301-red/8/'
    T_link7_ee = np.eye(4)
    T_link7_ee[2,3] = 0.045
    #T_link7_ee[2,3] = 0

    T_ee_cam = read_matrix(dataset + 'calibrations/T_ee_kin.txt')
    intrinics_matrix = read_matrix(dataset + 'calibrations/camera_matrix.txt')

    camera_poses = get_camera_poses(dataset + 'robot_poses.txt', 503, T_link7_ee,T_ee_cam)

    print "reading depth maps"
    #all_depth_maps = get_depth_maps(dataset + 'depth_imgs/*.jpg.tif')
    all_depth_maps = read_images(dataset + 'depth_imgs/*.jpg.tif',503)
    print"no of depth maps: ", len(all_depth_maps)
    #all_imgs = read_images(dataset + 'imgs/*.jpg',100)
    print "done reading depth maps"
    print("publishing")
    
    transform_list = []
    RMSE_list = []
    depth_offset = [x for x in np.arange(0.020, 0.030, 0.001)]
    #depth_offset = 0.025
    for i in range(len(depth_offset)):
    
        offset = depth_offset[i]
        #offset = 0.025
        electrode_positions_ground_truth = get_3d_points_from_ground_truth(dataset + 'labels_1.26/electrode_positions.txt')
        electrode_positions_yolo, all_yolo_labels = get_3d_points_from_yolo_labels(dataset + 'labels_1.26/yolo_labels.txt',all_depth_maps,intrinics_matrix,offset)
        
        electrode_positions_ground_truth_base = transform_pointlist_to_base(electrode_positions_ground_truth,camera_poses)
        electrode_positions_yolo_base = transform_pointlist_to_base(electrode_positions_yolo,camera_poses)
        
        #pc_yolo_base_pub.publish(electrode_positions_yolo_base)
        
        electrode_positions_ground_truth_base_open3d = convertCloudFromRosToOpen3d(electrode_positions_ground_truth_base)
        electrode_positions_yolo_base_open3d = convertCloudFromRosToOpen3d(electrode_positions_yolo_base)
        
        
        number_of_clusters =vSet.clusterTheElectrodes_DBSCAN(electrode_positions_yolo_base_open3d)
        print("offset: ", offset)
        vSet.clusterTheElectrodes_KMeans(electrode_positions_yolo_base_open3d, number_of_clusters, show=True)
        #vSet.getClusterMetrics_kMeans(electrode_positions_ground_truth_base_open3d, electrode_positions_yolo_base_open3d,number_of_clusters, show=True)

        
        electrode_positions_ground_truth_open3d = create_open3d_cloud_from_np_array(electrode_positions_ground_truth)
        electrode_positions_yolo_open3d = create_open3d_cloud_from_np_array(electrode_positions_yolo)
        #pc_yolo_pub.publish(convertCloudFromOpen3dToRos(electrode_positions_ground_truth_open3d[1]))
        #pc_yolo_base_pub.publish(convertCloudFromOpen3dToRos(electrode_positions_yolo_open3d[1]))

        """
        threshold = 0.06
        trans_init = np.eye(4)

        for n in range(len(electrode_positions_ground_truth_open3d)):
            reg_p2p = o3d.registration.registration_icp(electrode_positions_ground_truth_open3d[n], electrode_positions_yolo_open3d[n], threshold, trans_init, o3d.registration.TransformationEstimationPointToPoint())
            transform_list.append(reg_p2p.transformation)
            evaluation = o3d.registration.evaluate_registration(electrode_positions_ground_truth_open3d[n], electrode_positions_yolo_open3d[n], threshold, reg_p2p.transformation)
            RMSE_list.append(evaluation.inlier_rmse) 
            #print("offset: ", offset)
            #print("icp RMSE: ", evaluation.inlier_rmse)
            #print(reg_p2p.transformation)
            #draw_registration_result_corres(electrode_positions_ground_truth_open3d[i],electrode_positions_yolo_open3d[i],reg_p2p.transformation, reg_p2p.correspondence_set)
            
        RMSE_mean = np.round(np.mean(RMSE_list),5)
        RMSE_stddev = np.round(np.std(RMSE_list),5)
        print(np.round(offset,4),RMSE_mean, RMSE_stddev)
        #fig2.suptitle( ("mean RMSE mean %s $\pm$ %s " , (RMSE_mean, RMSE_stddev)) )
        #plt.title("translation part of ICP registration between ground turth and yolo estimated 3D electrode positions")
        #plt.show()
        
    #np.save(base_dir + "depth_variance/25/RMSE_list.npy", RMSE_list)
    #np.save(base_dir + "depth_variance/25/transform_list.npy", transform_list)
    #tr = np.load((base_dir + "depth_variance/25/RMSE_list.npy"), allow_pickle=True)
    #print(tr)
    """
    print "done transforming to base"
    rospy.spin()
    
