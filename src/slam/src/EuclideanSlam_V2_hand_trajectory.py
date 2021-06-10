import open3d as o3d
import glob
from natsort import natsorted
import numpy.linalg as ln
import numpy as np
import rospy
from utils.pcViewSet import *
from utils.ICP_open3D import draw_registration_result_corres
from utils.conversions import np2ros_poseStamped
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped
import random


# global var
max_correspondence_distance = 0.006
number_of_scans_to_ingnore = 200 # this is to avoid local loops
norm_threshold = 0.10 #CA_124_7 : 0.040
icp_fitness_threshold = 0.80
icp_inlier_threshold = 0.010
vSet = pcViewSet()
icp_convergence_criteria = o3d.registration.ICPConvergenceCriteria()
icp_convergence_criteria.max_iteration = 30
random.seed(10)


def getElectrodes(folder_path, skip):
    all_scans = glob.glob(folder_path)
    all_scans = natsorted(all_scans)
    xyz_points_list = []
    for i in range(0, len(all_scans), skip) :   
        scan = all_scans[i]
        xyz_points = np.load(scan, allow_pickle=True)
        xyz_points_list.append(xyz_points)
    return xyz_points_list

def slam(num_frames, step_to_skip, xyz_points_list, pub_pointCloud2, pub_poseStamped, path_to_save_ICP_relative_transform):
    absTform = np.eye(4)
    relTform = np.eye(4)
    initTform = np.eye(4)
    view_id = 0
    relTform_list = []
    current_point_cloud = o3d.geometry.PointCloud()
    previous_point_cloud = o3d.geometry.PointCloud()

    for n in range(0, num_frames, step_to_skip):

        current_point_cloud.points = o3d.utility.Vector3dVector(xyz_points_list[n])
        if n == 0 :
            vSet.addView(view_id, absTform=absTform, pointCloud=o3d.utility.Vector3dVector(xyz_points_list[n]))
            vSet.addPriorFactor(curr_viewID=view_id, prev_viewID=view_id,absTform=absTform)
            view_id = view_id + 1
            previous_point_cloud.points = current_point_cloud.points
            initTform = np.eye(4)
            continue
        #random.seed(10)
        relTform = o3d.registration.registration_icp(current_point_cloud, previous_point_cloud, max_correspondence_distance, initTform, o3d.registration.TransformationEstimationPointToPoint(), criteria=icp_convergence_criteria)
        evaluation = o3d.registration.evaluate_registration(current_point_cloud, previous_point_cloud, max_correspondence_distance,relTform.transformation)
        
        #print(evaluation)
        #draw_registration_result(current_point_cloud, previous_point_cloud, relTform.transformation)
        #print(n)
        relTform_list.append(relTform.transformation)
        absTform = np.matmul(absTform,relTform.transformation)
        vSet.addView(view_id, absTform=absTform, pointCloud=o3d.utility.Vector3dVector(xyz_points_list[n]))
        vSet.addOdometryFactor(view_id, view_id-1, absTform, relTform.transformation)
        
        previous_point_cloud.points = current_point_cloud.points
        initTform = relTform.transformation
        loop_closed = False
        
        #if n == num_frames-1:
        
        loop_closed, loop_view_id, loop_transform = loopClosure(n, view_id, vSet ,current_point_cloud, absTform, relTform)
        
        if loop_closed == True:
            vSet.addLoopFactor(view_id,loop_view_id, loop_transform)
            graph_optimised = vSet.optimizePoseGraph()
            vSet.updatePointCloudViewSetWithOptimisedPose(graph_optimised)
        else :
            graph_optimised = vSet.pcViewList
                

        vSet.publish_estimated_electrodes(n,vSet.pcViewList,pub_pointCloud2,pub_poseStamped, loop_closed)
        #vSet.publish_estimated_electrodes(n,vSet.pcViewList,pub_pointCloud2,pub_poseStamped)
        view_id = view_id + 1


    #graph_optimised = vSet.optimizePoseGraph()
    #vSet.updatePointCloudViewSetWithOptimisedPose(graph_optimised)
    #estimated_electrodes_in_first_camera_frame = vSet.mapElectrodes(vSet.pcViewList)
    #vSet.publishUnOptimisedpose()
    if path_to_save_ICP_relative_transform :
        np.save(str(path_to_save_ICP_relative_transform), relTform_list)
    return graph_optimised

    

        
def loopClosure(n, view_id, vSet, current_point_cloud, absTform, relTform):
            loop_view_id = None
            loop_closed = False
            loop_transform = None
            loop_fitness = 0
            #max_correspondence_distance = 0.010
            if n > number_of_scans_to_ingnore :
                for i in range( (len(vSet.pcViewList) - number_of_scans_to_ingnore)):    
                    view_temp = vSet.pcViewList[i]
                    absTform_temp = np.matmul(ln.inv(absTform),view_temp["absTform"])
                    absTform_temp_norm = ln.norm(absTform_temp[0:3, 3])
                    if absTform_temp_norm  < norm_threshold and absTform_temp_norm > 0:
                        point_cloud_temp = o3d.geometry.PointCloud()
                        point_cloud_temp.points = view_temp["pointCloud"]
                        relTform_temp = o3d.registration.registration_icp(current_point_cloud, point_cloud_temp, max_correspondence_distance, relTform.transformation, o3d.registration.TransformationEstimationPointToPoint())
                        evaluation = o3d.registration.evaluate_registration(current_point_cloud, point_cloud_temp, max_correspondence_distance,relTform_temp.transformation)
                        #draw_registration_result(current_point_cloud, point_cloud_temp, relTform_temp.transformation)
                        if evaluation.fitness > icp_fitness_threshold:
                            if evaluation.fitness > loop_fitness:
                                loop_fitness = evaluation.fitness
                                loop_view_id =  view_temp["viewID"]
                                loop_transform = relTform_temp.transformation
                                print("n: " +  str(n) +" current view id: " + str(view_id) + " pose list view id: " + str(loop_view_id) + " loop norm: "+ str(absTform_temp_norm) + " fitness: " + str(evaluation.fitness))
                                print("----------------------")
                                loop_closed = True
                                draw_registration_result_corres(current_point_cloud, point_cloud_temp, relTform_temp.transformation, relTform_temp.correspondence_set)
        
                                                         
            return loop_closed, loop_view_id, loop_transform    


if __name__ == "__main__":
    
    #rospy.init_node("slam")
    base_dir = "/home/Vishwanath/catkin_ws/src/master_arbeit/src/slam/src/data/hand_trajectory/static_phantom_head/beta_10/CA_124_6/"
    
    pub_electrodes = rospy.Publisher('map_pub_est', PointCloud2, queue_size=10)
    pub_cluster_center = rospy.Publisher('cluster_centers', PointCloud2, queue_size=10)
    pub_poseStamped = rospy.Publisher('camera_pose', PoseStamped, queue_size=10)
    step_to_skip = 1
    folder_path = base_dir + "electrode_position/*.npy"
    
    #paths
    #ground_truth_electrodes_path = base_dir + 'results/k_means_cluster_centers_ground_truth_transformed.npy'
    ground_truth_electrodes_path = base_dir + "results/ground_truth_electrodes_in_first_camera_frame_sync.npy"
    path_cluster_center_est = base_dir + "results/k_means_cluster_centers_estimated.npy"
    path_to_save_icp_relative_transform = base_dir + "results/icp_relative_transform.npy"
    #ground_truth_camera_pose_path = base_dir + "results/camera_poses_in_first_frame_unseen.npy"
    ground_truth_camera_pose_path = base_dir + "results/ground_truth_camera_bad_poses_removed.npy"
    un_optimised_camera_trajectory_path = base_dir + "results/un_optimised_camera_poses.npy"
    groundTruth_relative_cameraPose = base_dir + "results/groundTruth_relative_cameraPose.npy"
    icp_relative_transform = base_dir + "results/icp_relative_transform.npy"
    vis = base_dir + "results/ground_truth_electrodes_in_first_camera_frame_sync.npy"
    
    #Slam starts here
    xyz_points_list = getElectrodes(folder_path, step_to_skip)

    
    graph_optimised = slam(len(xyz_points_list), 1, xyz_points_list, pub_electrodes, pub_poseStamped, path_to_save_icp_relative_transform)
    
    vSet.publishUnOptimisedpose()
    un_optimised_camera_trajectory = vSet.saveUnOptimisedPoseList(un_optimised_camera_trajectory_path)
    estimated_electrodes_in_first_camera_frame = vSet.mapElectrodes(vSet.pcViewList)
    #number_of_cluster_centers = vSet.clusterTheElectrodes_DBSCAN(estimated_electrodes_in_first_camera_frame)
    #cluster_centers = vSet.clusterTheElectrodes_KMeans(estimated_electrodes_in_first_camera_frame,50,show=True)
    ground_truth_electrodes_in_first_camera_frame = vSet.get_ground_truth_electrodes_in_first_camera_frame(ground_truth_electrodes_path)
    
    vSet.getClusterMetrics_DBSCAN(ground_truth_electrodes_in_first_camera_frame, estimated_electrodes_in_first_camera_frame)

    vis = vSet.get_ground_truth_electrodes_in_first_camera_frame(vis)
    vSet.cluster_visualization(vis, estimated_electrodes_in_first_camera_frame,50,show=True)
    vSet.getClusterMetrics_kMeans(ground_truth_electrodes_in_first_camera_frame, estimated_electrodes_in_first_camera_frame,50,show=True)

    #trajectory metrics
    
    #print(len(un_optimised_camera_trajectory))
    #pose = np.load(ground_truth_camera_pose_path, allow_pickle=True)
    #print("pose length",len(pose))

    vSet.getErrorMetrics_hand_trajectory(un_optimised_camera_trajectory_path, ground_truth_camera_pose_path,1,plot=True)
    
    vSet.getErrorMetrics_hand_trajectory_ICP(icp_relative_transform, groundTruth_relative_cameraPose,1,plot=False)
    rospy.spin()
 

 

 