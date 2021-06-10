
import numpy as np
import cv2
import glob
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
from natsort import natsorted
import copy
from pcViewSet import *
from ICP_open3D import draw_registration_result_corres


def getElectrodes_from_array(electrodes_path, path_to_save=None, skip=1):
    electrodes = np.load(str(electrodes_path), allow_pickle=True)
    electrodes_list_dict = []
    for i in range(0, len(electrodes), skip) :   
        points = electrodes[i]
        points_dict = {"time_stamp" : i*33, "points" : points}
        electrodes_list_dict.append(points_dict)
    
    if path_to_save is not None:
        np.save(str(path_to_save), electrodes_list_dict)
    return electrodes_list_dict


def prepare_ground_truth_poses(ground_truth_camera_pose_dict, skip):
    ground_truth_camera_poses = np.load(str(ground_truth_camera_pose_dict), allow_pickle=True)
    ground_truth_camera_bad_poses_removed_dict = []
    for i in range(len(ground_truth_camera_poses)):
        if i < skip:
            continue
        if i == skip :
            temp_pose = ground_truth_camera_poses[i]
            base_time = temp_pose["time_stamp"]
            temp_trajectory = ground_truth_camera_poses[i]
            time_stamp = temp_trajectory["time_stamp"]
            trajectory = temp_trajectory
            trajectory["time_stamp"] = time_stamp-base_time
            ground_truth_camera_bad_poses_removed_dict.append(trajectory)
        else:
            temp_trajectory = ground_truth_camera_poses[i]
            time_stamp = temp_trajectory["time_stamp"]
            trajectory = temp_trajectory
            trajectory["time_stamp"] = time_stamp-base_time
            ground_truth_camera_bad_poses_removed_dict.append(trajectory)
    
    #print("original length", len(ground_truth_camera_poses))
    #print("new_length", len(ground_truth_camera_bad_poses_removed_dict))
    #print(ground_truth_camera_bad_poses_removed_dict[0:10])
    return ground_truth_camera_bad_poses_removed_dict


def synchronize_trajectory(ground_truth_camera_pose_dict, yolo_electrodes_list, path_to_save_ground_truth_camera_poses):
    #using nearest neighbor : correct way to sync trajectories
    #ground_truth_camera_poses = np.load(str(ground_truth_camera_pose_dict), allow_pickle=True)
    ground_truth_camera_poses = ground_truth_camera_pose_dict
    #yolo_electrodes = np.load(str(yolo_electrodes_list), allow_pickle=True)
    
    yolo_electrodes = yolo_electrodes_list
    ground_truth_camera_bad_poses_removed_list = []
    yolo_electrodes_list = []
    for i in range(len(yolo_electrodes)):
        threshold = 20
        #print("i", i)
        gt_ele = yolo_electrodes[i]
        time_ele = gt_ele["time_stamp"]
        yolo_electrodes_list.append(gt_ele["points"])
        #print("time_ele",time_ele)
        for j in range(len(ground_truth_camera_poses)):
            temp_pose = ground_truth_camera_poses[j]
            time_pose_time = temp_pose["time_stamp"]
            time_diff = np.abs(time_ele - time_pose_time)
            #print("time_diff", time_diff)
            if time_diff < threshold:
                threshold = time_diff
                pose_idx = j
                #print("idx", pose_idx)
        camera_pose = ground_truth_camera_poses[pose_idx]["pose"]
        #print("finally picked time stamp", ground_truth_camera_poses[pose_idx]["time_stamp"])
        ground_truth_camera_bad_poses_removed_list.append(camera_pose)
    #print(len(ground_truth_camera_bad_poses_removed_list))
    #print(len(yolo_electrodes_list))
    np.save(str(base_dir_cloud) + "ground_truth_camera_bad_poses_removed_list.npy" , ground_truth_camera_bad_poses_removed_list)
    return ground_truth_camera_bad_poses_removed_list, yolo_electrodes_list


def map_electrodes(camera_pose_list, electrodes_list):
    electrode_position_list = o3d.geometry.PointCloud()
    electrodes = o3d.geometry.PointCloud()
    electrode_Transformed = o3d.geometry.PointCloud()
    #print(len(camera_pose_list))
    #print(len(electrodes_list))
    for i in range(len(electrodes_list)):
        if (camera_pose_list[i][2,3])  < 0.001 :
            #print(camera_pose_list[i])
            #print(camera_pose_list[i][2,3])
            continue
        else: 
            electrodes.points = o3d.utility.Vector3dVector(electrodes_list[i])
            electrode_Transformed = copy.deepcopy(electrodes).transform(camera_pose_list[i])
            electrode_position_list = electrode_position_list + electrode_Transformed
    o3d.visualization.draw_geometries([electrode_position_list]) 
    return electrode_position_list

def get_tcam_T_kcam(base_dir_cloud):
    tcam_T_kcam_list = np.load(str(base_dir_cloud) + "tcam_T_kcam_dict.npy", allow_pickle=True)
    ground_truth_camera_bad_poses_removed_dict = np.load(str(base_dir_cloud) + "ground_truth_camera_bad_poses_removed_dict.npy", allow_pickle=True)
    tcam_T_kcam_bad_poses_removed_list = []

    for i in range (len(ground_truth_camera_bad_poses_removed_dict)):
        idx = ground_truth_camera_bad_poses_removed_dict[i]["idx"]
        print("i",i)
        print("idx",idx)
        tcam_T_kcam_bad_poses_removed_list.append(tcam_T_kcam_list[idx]["pose"])
    np.save(str(base_dir_cloud) + "tcam_T_kcam_bad_pose_removed_list.npy" , tcam_T_kcam_bad_poses_removed_list)

def create_open3d_cloud_from_np_array(np_array):
    np_points = np.load(str(np_array), allow_pickle=True)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(np_points)
    return point_cloud

def plot(inertia, x_label, y_label):
    
    fig, ax = plt.subplots(1,1)
    for i, point in enumerate(inertia):
        ax.scatter(i,np.round(inertia[i],3), c="k", s=100, marker='.')
    ax.set_ylabel(str(y_label))
    ax.set_xlabel(str(x_label))
    plt.show()


if __name__ == "__main__":
    
    vSet = pcViewSet()
    pc_yolo_base_pub = rospy.Publisher('pc_yolo_base', PointCloud2, queue_size=10)
    base_dir_cloud = '/media/pallando/share/students/Vishwanath/master_arbeit_data/hand_trajectory/static_phantom_head/CS_301_RED/14/'
    base_dir = "/home/Vishwanath/catkin_ws/src/master_arbeit/src/slam/src/data/hand_trajectory/static_phantom_head/CS_301_RED_14/"
    #ground_truth_electrodes_path = base_dir + "results/electrode_dictionary.npy"
    ground_truth_camera_pose_path = base_dir_cloud + "camera_poses_in_first_frame_unseen_dictionary.npy"
    #folder_path = base_dir + "electrode_position/*.npy"
    ground_truth_camera_bad_poses_removed = base_dir_cloud + "ground_truth_camera_bad_poses_removed"
    #ground_truth_electrodes_path = base_dir + "results/ground_truth_electrodes_in_first_camera_frame.npy"
    ground_truth_electrodes_path = base_dir + 'results/dbsc_cluster_centers_ground_truth_no_order.npy'
    
    #saves
    ground_truth_electrodes_dict = base_dir_cloud + "ground_truth_electrodes_dict.npy"



    #code

    electrode_positions_yolo = base_dir_cloud + "depth_variance/0/electrode_position_from_text_file.npy"
    electrodes_dict = getElectrodes_from_array(electrode_positions_yolo, skip=1)
    ground_truth_camera = prepare_ground_truth_poses(ground_truth_camera_pose_path,48)
    ground_truth_camera_bad_poses_removed_list, electrodes_list = synchronize_trajectory(ground_truth_camera, electrodes_dict, ground_truth_camera_bad_poses_removed)
    estimated_electrodes_in_first_camera_frame = map_electrodes(ground_truth_camera_bad_poses_removed_list, electrodes_list)
    







    """
    sync_skip = [x for x in np.arange(0, 60, 1)]
    for i in range(len(sync_skip)):
        skip = sync_skip[i]
        print("skip", skip)
        electrode_positions_yolo = base_dir_cloud + "depth_variance/0/electrode_position_from_text_file.npy"
        electrodes_dict = getElectrodes_from_array(electrode_positions_yolo, skip=1)
        ground_truth_camera = prepare_ground_truth_poses(ground_truth_camera_pose_path,skip)
        ground_truth_camera_bad_poses_removed_list, electrodes_list = synchronize_trajectory(ground_truth_camera, electrodes_dict, ground_truth_camera_bad_poses_removed)
        estimated_electrodes_in_first_camera_frame = map_electrodes(ground_truth_camera_bad_poses_removed_list, electrodes_list)
        pc_yolo_base_pub.publish(convertCloudFromOpen3dToRos(estimated_electrodes_in_first_camera_frame))
    
    #this is for cluster center comparision
    """
    """
    depth_offset = [x for x in np.arange(0, 2, 1)]
    threshold = 0.06
    trans_init = np.eye(4)
    inertia_list = []
    inlier_rmse_list = []
    
    for i in range(len(depth_offset)):
        depth = depth_offset[i]
        print(depth)
        electrode_positions_yolo = base_dir_cloud + "depth_variance/{}/electrode_position_from_text_file.npy".format(depth)
        path_to_save = base_dir_cloud + "depth_variance/{}/k_means_cluster_centers_ground_truth.npy".format(depth)
        
        electrodes_dict = getElectrodes_from_array(electrode_positions_yolo, skip=1)
        ground_truth_camera_bad_poses_removed_list, electrodes_list = synchronize_trajectory(ground_truth_camera_pose_path, electrodes_dict, ground_truth_camera_bad_poses_removed)
        estimated_electrodes_in_first_camera_frame = map_electrodes(ground_truth_camera_bad_poses_removed_list, electrodes_list)
    
        
        electrodes_dict = getElectrodes_from_array(electrode_positions_yolo, skip=1)
        ground_truth_camera = prepare_ground_truth_poses(ground_truth_camera_pose_path,48)
        ground_truth_camera_bad_poses_removed_list, electrodes_list = synchronize_trajectory(ground_truth_camera, electrodes_dict, ground_truth_camera_bad_poses_removed)
        estimated_electrodes_in_first_camera_frame = map_electrodes(ground_truth_camera_bad_poses_removed_list, electrodes_list)


        
        
        #inertia
        cluster_centers, inertia = vSet.clusterTheElectrodes_KMeans_depth(estimated_electrodes_in_first_camera_frame,47,path_to_save,show=False)
        inertia_list.append(inertia)
        pc_yolo_base_pub.publish(convertCloudFromOpen3dToRos(estimated_electrodes_in_first_camera_frame))
       
        
        # #cluster center comparision
        # cluster_centers_open3d = create_open3d_cloud_from_np_array(path_to_save)
        # ground_truth_electrodes_in_first_camera_frame = vSet.get_ground_truth_electrodes_in_first_camera_frame(ground_truth_electrodes_path)
        # reg_p2p = o3d.registration.registration_icp(cluster_centers_open3d, ground_truth_electrodes_in_first_camera_frame,  threshold, trans_init, o3d.registration.TransformationEstimationPointToPoint())

        # evaluation = o3d.registration.evaluate_registration(cluster_centers_open3d, ground_truth_electrodes_in_first_camera_frame,  threshold, reg_p2p.transformation)
        # #draw_registration_result_corres(cluster_centers_open3d, ground_truth_electrodes_in_first_camera_frame, reg_p2p.transformation, reg_p2p.correspondence_set)
        # #print("icp RMSE: ", evaluation.inlier_rmse)
        # inlier_rmse_list.append(evaluation.inlier_rmse)
        

        # #visualization
        # #vSet.cluster_visualization(ground_truth_electrodes_in_first_camera_frame, estimated_electrodes_in_first_camera_frame,47,show=True)

    plot(inertia_list, "depth(mm)", "inertia(m.sq)")
    plot(inlier_rmse_list, "depth(mm)", "RMSE")
    
    #"""
    
        
    #get_tcam_T_kcam(base_dir_cloud)

