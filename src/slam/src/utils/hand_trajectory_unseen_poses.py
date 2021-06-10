import rospy
import numpy as np
import numpy.linalg as la
from rt_msgs.msg import TransformRTStampedWithHeader
from conversions import ros2np_RTStampedWithHeader, np2ros_poseStamped, ros2np_Pose
from geometry_msgs.msg import PoseStamped 
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Path
import open3d as o3d
from open3d_helper import convertCloudFromOpen3dToRos
from ICP_open3D import draw_registration_result
import copy


#tcam : tracking camera
#kcam : kinect camera
#headband : reflective marker attached to either phantom head or human head
#bkMarker: blue_kinect_marker : reflective marker attaced to kinect 

headband_pose_list = []
blue_kinect_marker_pose_list = []

def headband_callback(msg):
    headband_pose_list.append(msg)

def kinect_callback(msg):
    blue_kinect_marker_pose_list.append(msg)

def rosbag_subscriber():
    #rospy.init_node('hand_trajectory', anonymous=True)
    rospy.Subscriber('/kinect_rt_tf',  TransformRTStampedWithHeader, kinect_callback)
    rospy.Subscriber('/head_rt_tf', TransformRTStampedWithHeader, headband_callback)
    rospy.spin()

def get_hand_trajectory(headband_pose_list, blue_kinect_marker_pose_list):
    head_t_blue_ros_list = []
    for i in range(len(blue_kinect_marker_pose_list)):
        blue_temp_np = ros2np_RTStampedWithHeader(blue_kinect_marker_pose_list[i])
        head_temp_np = ros2np_RTStampedWithHeader(headband_pose_list[i])
        head_t_blue = np.matmul(la.inv(head_temp_np), blue_temp_np)
        head_t_blue_ros = np2ros_poseStamped(head_t_blue)
        head_t_blue_ros_list.append(head_t_blue_ros)
    return head_t_blue_ros_list

def publish_hand_trajectory_as_path(hand_trajectory_list, pub):
    hand_path = Path()
    rate = rospy.Rate(10)
    print("publishing")
    while not rospy.is_shutdown():
        for i in range (len(hand_trajectory_list)):
            hand_path.header.stamp = rospy.Time.now()
            hand_path.header.frame_id = "/map"
            hand_path.poses.append(hand_trajectory_list[i])
            pub.publish(hand_path)
        rate.sleep()

def publish_hand_trajectory_as_poseStamped(hand_trajectory_list, pub):
    hand_path = PoseStamped()
    rate = rospy.Rate(10)
    print("publishing")
    while not rospy.is_shutdown():
            for i in range (len(hand_trajectory_list)):
                hand_path.header.frame_id = "/map"
                pub.publish(hand_trajectory_list[i])
            rate.sleep()
def pose_to_poseStamped(poses):
    pose_list = np.load(str(poses), allow_pickle=True)
    poseStamped_list = []
    for i in range(len(pose_list)):
        poseStamped_list.append(np2ros_poseStamped(pose_list[i]))
    return poseStamped_list

def create_np_array(trans, rot):
    temp = np.eye(4)
    temp[0,0] = float(rot[0])
    temp[0,1] = float(rot[1])
    temp[0,2] = float(rot[2])
    temp[1,0] = float(rot[3])
    temp[1,1] = float(rot[4])
    temp[1,2] = float(rot[5])
    temp[2,0] = float(rot[6])
    temp[2,1] = float(rot[7])
    temp[2,2] = float(rot[8])
    temp[0,3] = float(trans[0])/1000
    temp[1,3] = float(trans[1])/1000
    temp[2,3] = float(trans[2])/1000
    return temp


def read_cambar_text_file(path_to_text_file):
    #headband is a primary marker and blue kinect marker is defined wrt head band
    # h = headband
    # k = marker placed on the kinect camera
    # tcam = tracking camera = cambar

    with open(path_to_text_file) as f:
        all_lines = [x.split() for x in f.read().splitlines()]
        #print(len(all_lines))
        trajectory_list = []
        for i in range(len(all_lines)):
            if i == 0:
                continue
            h_temp = np.eye(4)
            #k_temp = np.eye(4)
            line = all_lines[i]

            #head marker stuff
            frame_id = line[0]
            time_stamp = np.int(line[1])
            h_trans = line[6:9]     
            h_rot = line[9:18]
            if i == 1:
                base_time = time_stamp
            #k_trans = line[25:28]
            #k_rot = line[28:37] 
            
            h_temp = create_np_array(h_trans,h_rot)

            #trajectory = {"frame_id" : frame_id, "time_stamp" : (time_stamp - base_time), "pose": h_temp}
            trajectory = {"frame_id" : frame_id, "time_stamp" : time_stamp, "pose": h_temp}

            trajectory_list.append(trajectory)
        print(trajectory_list)
        return trajectory_list

def get_hand_eye_matrix():
    # returns kinect pose wrt to blue marker attached to kinect : bkMarker_T_kcam
    blueMarker_T_kcam = np.array([[0.971156, 0.00774746,  -0.238264,  0.0249159], 
                                 [0.233779,  -0.226615,   0.945494,   0.115563],
                                 [-0.0466573,  -0.973938,  -0.221892,  -0.058718],
                                 [0.0,        0.0,         0.0,       1.0]],dtype=np.float32)
    return blueMarker_T_kcam

def get_tcam_T_kcam(tcam_T_bkMarker, path_to_save):
    # bkMarker_T_kcam : hand_eye
    bkMarker_T_kcam = get_hand_eye_matrix()
    tcam_T_kcam_list = []
    for i in range(len(tcam_T_bkMarker)):
        temp_traj = tcam_T_bkMarker[i]
        temp_pose = temp_traj["pose"]
        if np.array_equal(temp_pose, np.eye(4)) :
            tcam_T_kcam = {"frame_id" : temp_traj["frame_id"], "time_stamp" : temp_traj["time_stamp"], "pose": np.eye(4)}
            tcam_T_kcam_list.append(tcam_T_kcam)
        else:
            pose_calc = np.matmul(temp_pose,bkMarker_T_kcam)
            tcam_T_kcam = {"frame_id" : temp_traj["frame_id"], "time_stamp" : temp_traj["time_stamp"], "pose": pose_calc}
            tcam_T_kcam_list.append(tcam_T_kcam)
    np.save(str(path_to_save) ,tcam_T_kcam_list)
    return tcam_T_kcam_list

def get_camera_poses_in_first_frame(camera_poses, path_to_save):
    #tracking camera (tcam) acts like robot base here, bkMarker is attached to kcam
    #what we know = tcam_T_kcam : hand trajectory trajectory, bkMarker_T_kcam (through hand eye)  
    #camera_poses = tcam_T_kcam 
    #hand_eye : bkMarker_T_kcam
    #returns camera poses in first camera frame
    abs_camera_pose_list = []
    abs_camera_poseStamped_list  = []
    for i in range(len(camera_poses)) :
        #base_T_camera = ros2np_Pose(camera_poses[i])
        
        pose_temp = camera_poses[i]
        base_T_camera = pose_temp["pose"]

        if i == 0:
            firstCameraFrame_T_currentCameraFrame = np.eye(4)
            abs_camera_pose = {"frame_id" : pose_temp["frame_id"], "time_stamp" : pose_temp["time_stamp"], "pose": firstCameraFrame_T_currentCameraFrame}
            abs_camera_pose_list.append(abs_camera_pose)
            continue

        temp_matrix = np.matmul(firstCameraFrame_T_currentCameraFrame, la.inv(camera_poses[i-1]["pose"]))
        firstCameraFrame_T_currentCameraFrame = np.matmul(temp_matrix,base_T_camera)
        abs_camera_pose = {"frame_id" : pose_temp["frame_id"], "time_stamp" : pose_temp["time_stamp"], "pose": firstCameraFrame_T_currentCameraFrame}
        
        abs_camera_pose_list.append(abs_camera_pose)
        
        abs_camera_poseStamped_list .append(np2ros_poseStamped(firstCameraFrame_T_currentCameraFrame)) 
    np.save(str(path_to_save) ,abs_camera_pose_list)
    print(abs_camera_pose_list)
    return abs_camera_pose_list , abs_camera_poseStamped_list   


def get_groundTruth_relative_cameraPose(path_to_camera_pose, path_to_save):
    # calculates the relative pose between 2 consecative poses as ICP does
    # this will be compared to ICP results : ICP performance
    pose_array = np.load(str(path_to_camera_pose),allow_pickle = True)
    relative_pose_path_list = []
    for i in range(0,len(pose_array),1):
        if i == 0:
            continue
        if np.shape(pose_array[i]) == (4,4):
            curr_camera_pose = pose_array[i]
            prev_camera_pose = pose_array[i-1]
        else:
            curr_camera_pose = ros2np_Pose(pose_array[i])
            prev_camera_pose = ros2np_Pose(pose_array[i-1])
        
        #relative_pose_path_list.append(np.matmul(np.linalg.inv((pose_array[i-1])), (pose_array[i])))
        rel_pose = np.matmul(np.linalg.inv(prev_camera_pose), curr_camera_pose)
        relative_pose_path_list.append(rel_pose)
        #print(i)
        #print(rel_pose)
        #print("---------------------")
    #print(len(relative_pose_path_list))
    np.save(str(path_to_save), relative_pose_path_list)


def get_groundTruth_relative_cameraPose_skip(path_to_camera_pose, path_to_save):
    # calculates the relative pose between 2 consecative poses as ICP does
    # this will be compared to ICP results : ICP performance
    pose_array = np.load(str(path_to_camera_pose),allow_pickle = True)
    relative_pose_path_list = []
    print (len(pose_array))
    for i in range(0,len(pose_array),2):
        if i <=4 :
            continue 
        
        if np.shape(pose_array[i]) == (4,4):
            curr_camera_pose = pose_array[i]
            prev_camera_pose = pose_array[i-1]
        else:
            curr_camera_pose = ros2np_Pose(pose_array[i])
            prev_camera_pose = ros2np_Pose(pose_array[i-1])
        
        #relative_pose_path_list.append(np.matmul(np.linalg.inv((pose_array[i-1])), (pose_array[i])))
        rel_pose = np.matmul(np.linalg.inv(prev_camera_pose), curr_camera_pose)
        relative_pose_path_list.append(rel_pose)
        #print(i)
        #print(rel_pose)
    
    np.save(str(path_to_save), relative_pose_path_list)


def map_electrodes_groundtruth_trajectory(ground_truth_camera_pose_path, ground_truth_electrodes_path):
        ground_truth_camera_poses = np.load(str(ground_truth_camera_pose_path), allow_pickle=True)
        ground_truth_electrodes = np.load(str(ground_truth_electrodes_path), allow_pickle=True)
        electrode_position_list = o3d.geometry.PointCloud()
        electrode_position = o3d.geometry.PointCloud()
        electrode_Transformed = o3d.geometry.PointCloud()
        rospy.loginfo("mapping ground truth")
        
        for i in range(len(ground_truth_electrodes)):
            electrode_position.points = o3d.utility.Vector3dVector(ground_truth_electrodes[i])
            camera_pose = ground_truth_camera_poses[2*i]
            if np.array_equal(camera_pose, np.eye(4)):
                continue
            else:
                electrode_Transformed = copy.deepcopy(electrode_position).transform(camera_pose)
                electrode_position_list = electrode_position_list + electrode_Transformed
        o3d.visualization.draw_geometries([electrode_position_list]) 
        #np.save("/home/Vishwanath/catkin_ws/src/master_arbeit/src/slam/src/data/CA_124_1/results/ground_truth_electrodes_in_first_camera_frame.npy",electrode_position_list)
        #np.save(self.base_dir + "results/ground_truth_camera_pose_first_frame.npy", ground_truth_camera_poses)
        return electrode_position_list

    
if __name__ == "__main__" :
    rospy.init_node("hand_trajectory",anonymous=True)
    base_dir = "/media/pallando/share/students/Vishwanath/master_arbeit_data/hand_trajectory/static_phantom_head/CA_124/6/"
    pub_hand_trajectory_Path = rospy.Publisher('pub_hand_trajectory_path', Path, queue_size=10)
    pub_gt_electrodes = rospy.Publisher('gt_electrodes', PointCloud2, queue_size=10)
    #pub_hand_trajectory_poseStamped = rospy.Publisher('pub_hand_trajectory_poseStamped', PoseStamped, queue_size=10)

    #paths
    #camera_poses_in_first_frame_path = base_dir + "camera_poses_in_first_frame_unseen.npy"
    #camera_poses_in_first_frame_path = base_dir + "camera_poses_in_first_frame_unseen_dictionary.npy"
    camera_poses_in_first_frame_path = base_dir + "ground_truth_camera_bad_poses_removed.npy"
    tcam_T_kcam_path = base_dir + "tcam_T_kcam_dict"
    cambar_data = base_dir + "CA_124_6_processed.txt"
    tcam_T_ele_list_path = base_dir + "recorded_caps/tcam_T_ele_list.npy"
    ground_truth_electrodes_in_first_camera_frame_path = base_dir + "ground_truth_electrodes_in_first_camera_frame.npy"
    path_cluster_estimated =  base_dir + "k_means_cluster_centers_estimated.npy"
    path_to_cluster_gt = base_dir + "k_means_cluster_centers_ground_truth.npy"
    groundTruth_relative_cameraPose_path = base_dir + "groundTruth_relative_cameraPose.npy"
    groundTruth_relative_cameraPose_path_skip = base_dir + "groundTruth_relative_cameraPose_skip.npy"
    electrode_path = base_dir + "electrode_position_from_text_file"
    
    
    
    #tcam_T_bkMarker = read_cambar_text_file(cambar_data)

    #tcam_T_kcam_list = get_tcam_T_kcam(tcam_T_bkMarker, tcam_T_kcam_path)

    #pose_list,poseStamped_list = get_camera_poses_in_first_frame(tcam_T_kcam_list, camera_poses_in_first_frame_path)
    
    #poseStamped_list = pose_to_poseStamped(camera_poses_in_first_frame_path)
    #publish_hand_trajectory_as_path(poseStamped_list, pub_hand_trajectory_Path)
    #publish_hand_trajectory_as_poseStamped(poseStamped_list, pub_hand_trajectory_poseStamped)
    #map_electrodes_groundtruth_trajectory(pose_list, electrode_path)

    #ground_truth_electrodes_in_first_camera_frame = get_ground_truth_electrodes_in_first_camera_frame(tcam_T_kcam_path,tcam_T_ele_list_path,ground_truth_electrodes_in_first_camera_frame_path)
    #ground_truth_electrodes_in_first_camera_frame = get_cluster_transform(path_cluster_estimated,path_to_cluster_gt,ground_truth_electrodes_in_first_camera_frame,ground_truth_electrodes_in_first_camera_frame_path)
    #publish_gt_electrodes(ground_truth_electrodes_in_first_camera_frame,pub_gt_electrodes)
    
    # trajectory comparison stuff
    get_groundTruth_relative_cameraPose_skip(camera_poses_in_first_frame_path,groundTruth_relative_cameraPose_path_skip)
    get_groundTruth_relative_cameraPose(camera_poses_in_first_frame_path, groundTruth_relative_cameraPose_path)

    

