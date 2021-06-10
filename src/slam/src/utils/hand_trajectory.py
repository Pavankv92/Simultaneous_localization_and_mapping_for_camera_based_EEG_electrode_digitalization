import rospy
import numpy as np
import numpy.linalg as la
from rt_msgs.msg import TransformRTStampedWithHeader
from conversions import ros2np_RTStampedWithHeader, np2ros_poseStamped
from geometry_msgs.msg import PoseStamped 
from sensor_msgs.msg import PointCloud2
from nav_msgs.msg import Path
import open3d as o3d
from open3d_helper import convertCloudFromOpen3dToRos
from ICP_open3D import draw_registration_result, draw_registration_result_corres
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

def get_timestamped_array(pose_list):
    pass

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
        print(len(all_lines))
        trajectory_list = []
        for i in range(len(all_lines)):
            
            
            if i == 0:
                continue
            h_temp = np.eye(4)
            #k_temp = np.eye(4)
            line = all_lines[i]

            #head marker stuff
            frame_id = line[0]
            time_stamp = line[1]
            h_trans = line[6:9]     
            h_rot = line[9:18]
            
            #k_trans = line[25:28]
            #k_rot = line[28:37] 
            
            h_temp = create_np_array(h_trans,h_rot)

            if np.array_equal(h_temp, np.eye(4)):
                continue
            else :
                trajectory = {"frame_id" : frame_id, "time_stamp" : time_stamp, "tcam_T_h": h_temp}
                trajectory_list.append(trajectory)
        
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
        temp_pose = temp_traj["tcam_T_h"]
        if np.array_equal(temp_pose, np.eye(4)) :
            continue
        tcam_T_kcam_list.append(np.matmul(temp_pose,bkMarker_T_kcam))
    np.save(str(path_to_save),tcam_T_kcam_list)
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
        base_T_camera = camera_poses[i]

        if i == 0:
            firstCameraFrame_T_currentCameraFrame = np.eye(4)
            abs_camera_pose_list.append(firstCameraFrame_T_currentCameraFrame)
            continue

        temp_matrix = np.matmul(firstCameraFrame_T_currentCameraFrame, la.inv(camera_poses[i-1]))
        firstCameraFrame_T_currentCameraFrame = np.matmul(temp_matrix,base_T_camera)
        abs_camera_pose_list.append(firstCameraFrame_T_currentCameraFrame)
        abs_camera_poseStamped_list .append(np2ros_poseStamped(firstCameraFrame_T_currentCameraFrame)) 
    np.save(str(path_to_save),abs_camera_pose_list)
    return abs_camera_pose_list , abs_camera_poseStamped_list   

def get_ground_truth_electrodes_in_first_camera_frame(tcam_T_kcam_list_path, tcam_T_ele_list_path, path_to_save):
    tcam_T_kcam_list = np.load(str(tcam_T_kcam_list_path), allow_pickle=True)
    tcam_T_ele_list = np.load(str(tcam_T_ele_list_path), allow_pickle=True)
    first_camera_frame_T_ele = []
    for i in range(len(tcam_T_ele_list)):
        temp = np.matmul(la.inv(tcam_T_kcam_list[12]), tcam_T_ele_list[i])
        first_camera_frame_T_ele.append(temp[0:3,3])
    
    np.save(str(path_to_save),first_camera_frame_T_ele)
    return first_camera_frame_T_ele

def publish_gt_electrodes(first_camera_frame_T_ele, pub_electrode):
        """
        gt_electrodes = np.load(str(first_camera_frame_T_ele), allow_pickle=True)
        electrode_position = o3d.geometry.PointCloud()
        electrode_position.points = o3d.utility.Vector3dVector(gt_electrodes)
        """
        electrode_position = first_camera_frame_T_ele
        print("Publishing gt eletrodes")
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            pub_electrode.publish(convertCloudFromOpen3dToRos(electrode_position))
            rate.sleep()

def get_cluster_transform(path_cluster_estimated, path_to_cluster_gt, ground_truth_electrodes_in_first_camera_frame, path_to_save):
    est = np.load(str(path_cluster_estimated))
    gt = np.load(str(path_to_cluster_gt))
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(gt)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(est)
    gt_ele = o3d.geometry.PointCloud()
    gt_ele.points = o3d.utility.Vector3dVector(ground_truth_electrodes_in_first_camera_frame)
    
    
    threshold = 2
    trans_init = np.eye(4)
    reg_p2p = o3d.registration.registration_icp(source, target, threshold, trans_init, o3d.registration.TransformationEstimationPointToPoint())
    print("Transformation is:")
    print(reg_p2p.transformation)
    print("")
    evaluation = o3d.registration.evaluate_registration(source, target, threshold, reg_p2p.transformation)
    draw_registration_result_corres(source, target, reg_p2p.transformation, reg_p2p.correspondence_set)
    print(evaluation.inlier_rmse)
    gt_transformed = o3d.geometry.PointCloud()
    gt_transformed = copy.deepcopy(gt_ele).transform(reg_p2p.transformation)
    
    np.save(str(path_to_save),np.asarray(gt_transformed.points))
    return gt_transformed


        

if __name__ == "__main__" :
    rospy.init_node("hand_trajectory",anonymous=True)
    base_dir = "/media/pallando/share/students/Vishwanath/master_arbeit_data/hand_trajectory/static_phantom_head/CA_124/6/"
    pub_hand_trajectory_Path = rospy.Publisher('pub_hand_trajectory_path', Path, queue_size=10)
    pub_gt_electrodes = rospy.Publisher('gt_electrodes', PointCloud2, queue_size=10)
    #pub_hand_trajectory_poseStamped = rospy.Publisher('pub_hand_trajectory_poseStamped', PoseStamped, queue_size=10)

    #paths
    camera_poses_in_first_frame_path = base_dir + "camera_poses_in_first_frame.npy"
    tcam_T_kcam_path = base_dir + "tcam_T_kcam_list.npy"
    cambar_data = base_dir + "CS_301_7_processed.txt"
    tcam_T_ele_list_path = base_dir + "recorded_caps/tcam_T_ele_list.npy"
    ground_truth_electrodes_in_first_camera_frame_path = base_dir + "ground_truth_electrodes_in_first_camera_frame_sync.npy"
    path_cluster_estimated =  base_dir + "k_means_cluster_centers_estimated.npy"
    path_to_cluster_gt = base_dir + "k_means_cluster_centers_ground_truth.npy"
    
     
    
    """
    rosbag_subscriber()
    np.save(base_dir+"head_band_pose_list", headband_pose_list)
    np.save(base_dir+"blue_kinect_marker_pose_list", blue_kinect_marker_pose_list)
    headband_pose_list = np.load(base_dir + "head_band_pose_list.npy", allow_pickle = True)
    blue_kinect_marker_pose_list = np.load(base_dir + "blue_kinect_marker_pose_list.npy", allow_pickle = True)
    print("kinect list:" , len(blue_kinect_marker_pose_list))
    print("head list:" , len(headband_pose_list))
    hand_trajectory_list = get_hand_trajectory(headband_pose_list,blue_kinect_marker_pose_list)
    publish_hand_trajectory(hand_trajectory_list, pub_hand_trajectory)
    """
    
    #tcam_T_bkMarker = read_cambar_text_file(cambar_data)

    #tcam_T_kcam_list = get_tcam_T_kcam(tcam_T_bkMarker, tcam_T_kcam_path)

    #pose_list,poseStamped_list = get_camera_poses_in_first_frame(tcam_T_kcam_list, camera_poses_in_first_frame_path)
    #publish_hand_trajectory_as_path(poseStamped_list, pub_hand_trajectory_Path)
    #publish_hand_trajectory_as_poseStamped(poseStamped_list, pub_hand_trajectory_poseStamped)

    ground_truth_electrodes_in_first_camera_frame = get_ground_truth_electrodes_in_first_camera_frame(tcam_T_kcam_path,tcam_T_ele_list_path,ground_truth_electrodes_in_first_camera_frame_path)
    #ground_truth_electrodes_in_first_camera_frame = get_cluster_transform(path_cluster_estimated,path_to_cluster_gt,ground_truth_electrodes_in_first_camera_frame,ground_truth_electrodes_in_first_camera_frame_path)
    #publish_gt_electrodes(ground_truth_electrodes_in_first_camera_frame,pub_gt_electrodes)
    
    # trajectory comparison stuff
    #print(len(pose_list))


    

