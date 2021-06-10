#!/usr/bin/env python


import rospy
import ros_numpy 
import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
from geometry_msgs.msg import Pose
from nav_msgs.msg import Path
from conversions import np2ros_poseStamped, ros2np_Pose
import open3d as o3d


points = []
xyz_points = []
camera_pose_list = []
camera_pose = Pose()
abs_camera_pose_list = []
abs_camera_pose = Path()


def callback(msg):
    xyz_array = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)
    xyz_points.append(xyz_array)

def callbackCameraPose(msg):
    camera_pose = msg
    camera_pose_list.append(camera_pose)

def callbackAbsolutePose(msg):
    abs_camera_pose = msg
    abs_camera_pose_list.append(abs_camera_pose)


def listener():
    rospy.init_node('listner', anonymous=True)
    rospy.Subscriber('/electrode_positions', PointCloud2, callback)
    #rospy.Subscriber('/camera_pose', Pose, callbackCameraPose)
    #rospy.Subscriber('/abs_pose_list', Path, callbackAbsolutePose)
    rospy.spin()

def createArray(array_path, save_path, save_name):
    common_array = np.load(str(array_path),allow_pickle = True)
    print(len(common_array))
    for i in range(len(common_array)):
        if save_name == "electrode" :
            if (np.shape(common_array[i])[0] >= 3): 
                np.save(str(save_path) + str(save_name) +'_{}'.format(i),common_array[i])
        else:
            np.save(str(save_path) + str(save_name) +'_{}'.format(i),common_array[i])
    print("array created")

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
    

def removeBadPoses(path_to_bad_poses, path_to_save):
        camera_pose_list = np.load(str(path_to_bad_poses), allow_pickle=True)
        camera_pose_list_1 = []
        camera_pose_list_2 = []
        camera_pose_list_3 = []
        camera_pose_list_4 = []
        camera_pose_list_5 = []
        camera_pose_list_6 = []
        camera_pose_list_1 [:92] = camera_pose_list [:92]
        camera_pose_list_2 [107:153] = camera_pose_list [107:153]
        camera_pose_list_3 [168:213] = camera_pose_list [168:213]
        camera_pose_list_4 [229:275] = camera_pose_list [229:275]
        camera_pose_list_5 [290:336] = camera_pose_list [290:336]
        camera_pose_list_6 [351:442] = camera_pose_list [351:442]

        conc_array = camera_pose_list_1 + camera_pose_list_2 +camera_pose_list_3 + camera_pose_list_4 + camera_pose_list_5 + camera_pose_list_6
        np.save(str(path_to_save), conc_array)        

 
def get_groundTruth_relative_cameraPose(path_to_camera_pose, path_to_save):
    # calculates the relative pose between 2 consecative poses as ICP does
    # this will be compared to ICP results : ICP performance
    pose_array = np.load(str(path_to_camera_pose),allow_pickle = True)
    relative_pose_path_list = []
    for i in range(len(pose_array)):
        if i == 0:
            continue
        if np.shape(pose_array[i]) == (4,4):
            curr_camera_pose = pose_array[i]
            prev_camera_pose = pose_array[i-1]
        else:
            curr_camera_pose = ros2np_Pose(pose_array[i])
            prev_camera_pose = ros2np_Pose(pose_array[i-1])
        
        #relative_pose_path_list.append(np.matmul(np.linalg.inv((pose_array[i-1])), (pose_array[i])))
        relative_pose_path_list.append(np.matmul(np.linalg.inv(prev_camera_pose), curr_camera_pose))
    
    np.save(str(path_to_save), relative_pose_path_list)

def get_groundTruth_relative_cameraPose_skip(path_to_camera_pose, path_to_save, step_to_skip):
    # calculates the relative pose between 2 consecative poses as ICP does
    # this will be compared to ICP results : ICP performance
    pose_array = np.load(str(path_to_camera_pose),allow_pickle = True)
    relative_pose_path_list = []
    for i in range(0, len(pose_array), step_to_skip):
        if i == 0:
            continue
        if np.shape(pose_array[i]) == (4,4):
            curr_camera_pose = pose_array[i]
            prev_camera_pose = pose_array[i-step_to_skip]
        else:
            curr_camera_pose = ros2np_Pose(pose_array[i])
            prev_camera_pose = ros2np_Pose(pose_array[i-step_to_skip])
        
        #relative_pose_path_list.append(np.matmul(np.linalg.inv((pose_array[i-1])), (pose_array[i])))
        relative_pose_path_list.append(np.matmul(np.linalg.inv(prev_camera_pose), curr_camera_pose))
    
    np.save(str(path_to_save), relative_pose_path_list)

def vis_electrodes(electrodes_path):
    electrodes = np.load(str(electrodes_path), allow_pickle=True)
    electrode_position_list = o3d.geometry.PointCloud()
    electrode_position = o3d.geometry.PointCloud()
    for i in range(len(electrodes)):
        electrode_position.points = o3d.utility.Vector3dVector(electrodes[i])
        electrode_position_list = electrode_position_list + electrode_position
    o3d.visualization.draw_geometries([electrode_position_list])



if __name__ == "__main__":
    
    base_dir = "/home/Vishwanath/catkin_ws/src/master_arbeit/src/slam/src/data/robot_trajectory/CS_301_RED_8/"
    
    """
    listener()
    #np.save(base_dir+"electrodes_from_blob_detection", xyz_points)
    #np.save(base_dir+"camera_pose_from_bag_file", camera_pose_list)
    """

    #electrodes
    """
    path_to_text_file = base_dir + "depth_variance/25/CS-301-red_8.txt"
    path_to_save = base_dir + "electrode_position_from_text_file.npy"
    readElectrodesFromTextFile(path_to_text_file, path_to_save)
    
    
    path_to_bad_electrodes = base_dir + "electrode_position_from_text_file.npy"
    path_to_save_electrodes = base_dir + "electrode_position_from_text_file_bad_pose_removed.npy"
    removeBadPoses(path_to_bad_electrodes, path_to_save_electrodes)
   
    electrode_array_path = base_dir + "electrode_position_from_text_file_bad_pose_removed.npy"
    electrode_save_path =  base_dir + "electrode_position/"
    electrode_save_name = "electrode"
    createArray(electrode_array_path, electrode_save_path, electrode_save_name)
    """
    #"""
    #ground truth camera
    #path_to_bad_poses_camera = base_dir + "camera_position/ground_truth_camera_pose_path.npy"
    #path_to_save_camera = base_dir + "camera_position/ground_truth_camera_pose_path_bad_pose_removed.npy"
    #removeBadPoses(path_to_bad_poses_camera, path_to_save_camera)
    
    ground_truth_camera_pose_path = base_dir + "camera_position/ground_truth_camera_pose_path_bad_pose_removed.npy"
    #groundTruth_relative_cameraPose = base_dir + "results/groundTruth_relative_cameraPose.npy"
    groundTruth_relative_cameraPose_skip = base_dir + "results/groundTruth_relative_cameraPose_skip.npy"

    #get_groundTruth_relative_cameraPose(ground_truth_camera_pose_path, groundTruth_relative_cameraPose)
    get_groundTruth_relative_cameraPose_skip(ground_truth_camera_pose_path, groundTruth_relative_cameraPose_skip,2)
    temp = np.load(groundTruth_relative_cameraPose_skip, allow_pickle=True)
    print(len(temp))
    #"""
    
    
    """
    #tracking camera
    dataset = '/media/pallando/share/FASOD/data/1009/CA-124/8/labels_1.26/'
    path_to_text_file = dataset + "electrode_positions.txt"
    path_to_save = base_dir + "tracking_camera/ground_truth_electrodes_from_tracking_camera.npy"
    readElectrodesFromTextFile(path_to_text_file, path_to_save)
    
    path_to_bad_electrodes = base_dir + "tracking_camera/ground_truth_electrodes_from_tracking_camera.npy"
    path_to_save_electrodes = base_dir + "tracking_camera/ground_truth_electrodes_from_tracking_camera_bad_pose_removed.npy"
    removeBadPoses(path_to_bad_electrodes, path_to_save_electrodes)
    
    
    #electrode_array_path = base_dir + "tracking_camera/ground_truth_electrodes_from_tracking_camera_bad_pose_removed.npy"
    #electrode_save_path =  base_dir + "electrode_position/"
    #electrode_save_name = "electrode"
    #createArray(electrode_array_path, electrode_save_path, electrode_save_name)
    """
    