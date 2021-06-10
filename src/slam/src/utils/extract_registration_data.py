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


def get_camera_poses(file_name, num_poses, T_link78,T_ee_cam):
    #returns camera poses wrt robot frame: base_T_camera
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

def getGroundtruthCameraPosition(camera_poses, path_to_save):
    #camera_poses = Base_T_camera
    #returns all camera poses in first camera frame
    abs_camera_pose_list = []
    for i in range(len(camera_poses)) :
        #base_T_camera = ros2np_Pose(camera_poses[i])
        base_T_camera = camera_poses[i]

        if i == 0:
            firstCameraFrame_T_currentCameraFrame = np.eye(4)
            abs_camera_pose_list.append(firstCameraFrame_T_currentCameraFrame)
            continue

        temp_matrix = np.matmul(firstCameraFrame_T_currentCameraFrame, LA.inv(camera_poses[i-1]))
        firstCameraFrame_T_currentCameraFrame = np.matmul(temp_matrix,base_T_camera)
        abs_camera_pose_list.append(firstCameraFrame_T_currentCameraFrame) 
    np.save(str(path_to_save),abs_camera_pose_list)
    return abs_camera_pose_list

def read_labels_yolo(file_name, all_depth_maps, intrinics_matrix,pc_pub,all_camera_poses,tf_pub):
    with open(file_name) as f:
        all_lines = [x.split() for x in f.read().splitlines()]
        for frame_idx in range(len(all_lines)):
            line = all_lines[frame_idx]
            depth_map = all_depth_maps[frame_idx]
            camera_pose = all_camera_poses[frame_idx]
            points = []
            num_labels =  (len(line)-2)/5
            for i in range(num_labels):
                x_min = line[4+ i*5+1]
                y_min = line[4+ i*5+2]
                x_max = line[4+ i*5+3]
                y_max = line[4+ i*5+4]
                #print x_min, y_min, x_max, y_max
                x_mean = (int(x_min)+int(x_max))/2.
                y_mean = (int(y_min)+int(y_max))/2.
                #print x_mean, y_mean

                depth = depth_map.getpixel((x_mean,y_mean))
                if depth==0 or depth > 0.6: continue
                X_over_Z = (x_mean-intrinics_matrix[0,2]) / intrinics_matrix[0,0]
                Y_over_Z = (y_mean-intrinics_matrix[1,2]) / intrinics_matrix[1,1]
                Z = depth / np.sqrt(1. + X_over_Z**2 + Y_over_Z**2)
                X = X_over_Z * Z
                Y = Y_over_Z * Z
                points.append([X, Y, Z])

            header = Header()
            header.frame_id = 'rgb_camera_link'
            header.stamp = rospy.Time.now()
            electrode_pc = pcl2.create_cloud_xyz32(header, np.asarray(points))
            #pc_pub.publish(electrode_pc)


            a = pm.fromMatrix(camera_pose)
            pose_msg = pm.toMsg(a)
            tf_pose = pm.toTf(a)
            tf_pub.publish(pose_msg)

            tf_stamped = TransformStamped()
            tf_stamped.transform.translation = pose_msg.position
            tf_stamped.transform.rotation = pose_msg.orientation
            electrode_pc_base = do_transform_cloud(electrode_pc, tf_stamped)
            electrode_pc_base.header.frame_id = "base"
            pc_pub.publish(electrode_pc_base)
            print "publishing"
            sleep(0.1)

            #fig, ax = plt.subplots()
            #ax.imshow(depth_map)
            #plt.show()
            #print "drawn"
            #break

def create_pointcloud(all_images, all_depth_maps, intrinics_matrix,pc_dense_pub, all_camera_poses):
    # this is just to create point for visualisation purposes
    all_pointclouds = []
    num_frames = len(all_images)
    for i in range(num_frames):
        rgb_image = all_images[i]
        depth_map = all_depth_maps[i]
        #print(depth_map.size[:])
        points = []
        for v in range(depth_map.size[1]):
            for u in range(depth_map.size[0]):
                color = rgb_image.getpixel((u,v))

                depth = depth_map.getpixel((u,v))
                if depth==0 or depth > 0.6: continue
                X_over_Z = (u-intrinics_matrix[0,2]) / intrinics_matrix[0,0]
                Y_over_Z = (v-intrinics_matrix[1,2]) / intrinics_matrix[1,1]
                Z = depth 
                X = X_over_Z * Z
                Y = Y_over_Z * Z
                #points.append([X, Y, Z])
                a = 255
                rgb = struct.unpack('I', struct.pack('BBBB', color[2], color[1], color[0], a))[0]

                points.append([X,Y,Z,rgb])

        fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('rgb', 12, PointField.UINT32, 1),
                  ]

        header = Header()
        header.stamp = rospy.Time.now()
        header.frame_id = "rgb_camera_link"
        electrode_pc = pcl2.create_cloud(header, fields, points)
        # pc_pub.publish(electrode_pc)

        camera_pose = all_camera_poses[i]
        a = pm.fromMatrix(camera_pose)
        pose_msg = pm.toMsg(a)

        tf_stamped = TransformStamped()
        tf_stamped.transform.translation = pose_msg.position
        tf_stamped.transform.rotation = pose_msg.orientation
        electrode_pc_base = do_transform_cloud(electrode_pc, tf_stamped)
        electrode_pc_base.header.frame_id = "map"
        all_pointclouds.append(electrode_pc_base)
        pc_dense_pub.publish(electrode_pc_base)
        print "pointcloud done", i
    
    #np.save("/home/Vishwanath/catkin_ws/src/master_arbeit/src/slam/src/data/CA_124_1_YOLO/results/CA_124_1_point_cloud_first_camera_frame.npy",all_pointclouds)
    print("pointcloud saved with length %s", len(all_pointclouds))

def publish_pointcloud(path, pc_dense_pub):
    all_pointclouds = np.load(str(path), allow_pickle=True)
    print("pointcloud with length %s", len(all_pointclouds))
   
    electrode_position_list = o3d.geometry.PointCloud()
    for p in range(0,len(all_pointclouds),5):
        temp = convertCloudFromRosToOpen3d(all_pointclouds[p])
        electrode_position_list = electrode_position_list + temp
        print("completed: %s", p)
    
    print("publishing")
    #electrode_position_list = o3d.geometry.voxel_down_sample(electrode_position_list, voxel_size=0.05)
    o3d.visualization.draw_geometries([electrode_position_list])
    electrodes_pointcloud2 = convertCloudFromOpen3dToRos(electrode_position_list)
    pc_dense_pub.publish(electrodes_pointcloud2)
    
def read_matrix(file_name):
    #to read calibration matrix
    with open(file_name) as f:
        return np.loadtxt(f,delimiter=',')


def publish_cluster_centers(path_to_cluster_centers, cluster_pub):
    # for visulaising both point clouds along with ceneters
    cluster_centres = np.load(str(path_to_cluster_centers), allow_pickle=True)
    cluster_centres_o3d = o3d.geometry.PointCloud()
    cluster_centres_o3d.points = o3d.utility.Vector3dVector(cluster_centres)
    cluster_centres_ros = convertCloudFromOpen3dToRos(cluster_centres_o3d)
    cluster_centres_ros.header.frame_id = "map"
    cluster_pub.publish(cluster_centres_ros)


def get_3d_points_from_yolo_labels(file_name_yolo_labels, path_to_save_electrode_file, all_depth_maps, intrinics_matrix, depth_offset):
    all_electrode_positions= []
    all_yolo_labels = []
    with open(path_to_save_electrode_file) as yolo_write:
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
                        continue
                    depth = depth - depth_offset
                    X = (x_mean-intrinics_matrix[0,2]) / intrinics_matrix[0,0] * depth
                    Y = (y_mean-intrinics_matrix[1,2]) / intrinics_matrix[1,1] * depth
                    Z = depth

                    points.append([X, Y, Z, 1.0])
                    yolo_labels.append([x_mean, y_mean])
                all_yolo_labels.append(yolo_labels)
                all_electrode_positions.append(points)
           
    return all_electrode_positions, all_yolo_labels


if __name__ == "__main__":
    
    base_dir = base_dir = "/home/Vishwanath/catkin_ws/src/master_arbeit/src/slam/src/data/CA_124_1/"
    dataset = '/media/pallando/share/FASOD/data/1009/CA-124/1/'

    rospy.init_node("data_extraction_for_pavan")
    #pc_yolo_pub = rospy.Publisher('electrode_positions_yolo_base', PointCloud2, queue_size=10)
    pc_dense_pub = rospy.Publisher('pc_dense_base', PointCloud2, queue_size=10)
    cluster_pub_gt = rospy.Publisher('cluster_centers_gt', PointCloud2, queue_size=10)
    #cluster_pub_est = rospy.Publisher('cluster_centers_est', PointCloud2, queue_size=10)
    map_pub_gt = rospy.Publisher('map_pub_gt', PointCloud2, queue_size=10)
    #map_pub_est = rospy.Publisher('map_pub_est', PointCloud2, queue_size=10)

    #tf_pub = rospy.Publisher('camera_pose', Pose, queue_size=10)
    
    
    T_link7_ee = np.eye(4)
    T_link7_ee[2,3] = 0.045

    T_ee_cam = read_matrix(dataset + 'calibrations/T_ee_kin.txt')
    intrinics_matrix = read_matrix(dataset + 'calibrations/camera_matrix.txt')

    camera_poses = get_camera_poses(dataset + 'robot_poses.txt', 503, T_link7_ee,T_ee_cam)
    ground_truth_camera_pose_path = base_dir + "camera_position/ground_truth_camera_pose_path.npy"
    camera_poses = getGroundtruthCameraPosition(camera_poses, ground_truth_camera_pose_path)

    
    print "reading depth maps"
    all_depth_maps = read_images('/home/Vishwanath/temp/depth_imgs/*.jpg.tif',503)
    print "done reading depth maps"
    all_imgs = read_images(dataset + 'imgs/*.jpg',499)
    
    print("creating point cloud")
    create_pointcloud(all_imgs,all_depth_maps,intrinics_matrix,pc_dense_pub,camera_poses)
    
    """
    publish_pointcloud("/home/Vishwanath/catkin_ws/src/master_arbeit/src/slam/src/data/CA_124_1_YOLO/results/CA_124_1_point_cloud_first_camera_frame.npy", pc_dense_pub)

    #path_cluster_center_est = "/home/Vishwanath/catkin_ws/src/master_arbeit/src/slam/src/data/CA_124_1/results/k_means_cluster_centers_estimated.npy"
    path_cluster_center_gt = base_dir + "results/k_means_cluster_centers_ground_truth.npy"
    #publish_cluster_centers(path_cluster_center_est,cluster_pub_est)
    publish_cluster_centers(path_cluster_center_gt,cluster_pub_gt)

    #path_map_gt = base_dir + "results/ground_truth_electrodes_in_first_camera_frame.npy"
    #path_map_est = "/home/Vishwanath/catkin_ws/src/master_arbeit/src/slam/src/data/CA_124_1/results/estimated_electrodes_in_first_camera_frame.npy"
    #publish_cluster_centers(path_map_est,map_pub_est)
    #publish_cluster_centers(path_map_gt,map_pub_gt)
    #read_labels_yolo(dataset + 'labels_3.2/yolo_labels.txt', all_depth_maps, intrinics_matrix,pc_yolo_pub,camera_poses,tf_pub)
    """
    rospy.spin()
    
    

    
    





