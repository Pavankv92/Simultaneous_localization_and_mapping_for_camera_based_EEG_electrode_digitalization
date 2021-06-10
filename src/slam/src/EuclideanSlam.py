import open3d as o3d
import glob
from natsort import natsorted
import numpy.linalg as ln
import numpy as np
import rospy

from utils.pcViewSet import *
from utils.ICP_open3D import draw_registration_result
from utils.conversions import np2ros_poseStamped


file_path = "/home/Vishwanath/catkin_ws/src/master_arbeit/src/slam/src/data/CA_124_1/electrode_position/*.npy"
ground_truth_camera_pose_path = "/home/Vishwanath/catkin_ws/src/master_arbeit/src/slam/src/data/CA_124_1/camera_position_final.npy"
ground_truth_electrode_path = "/home/Vishwanath/catkin_ws/src/master_arbeit/src/slam/src/data/CA_124_1/electrode_position_from_text_file_final.npy"
all_scans = glob.glob(file_path)
xyz_points_list = []
for scan in natsorted(all_scans) :   
    xyz_points = np.load(scan, allow_pickle=True)
    #xyz_points = xyz_points/1000 # dataset is in mm
    xyz_points_list.append(xyz_points)

#print(xyz_points_list)

# objects initiations
vSet = pcViewSet()
absTform = np.eye(4)
relTform = np.eye(4)
initTform = np.eye(4)
view_id = 0
max_correspondence_distance = 0.002
number_of_scans_to_ingnore = 100 # this is to avoid local loops
norm_threshold = 0.2
icp_fitness_threshold = 0.80
current_point_cloud = o3d.geometry.PointCloud()
previous_point_cloud = o3d.geometry.PointCloud()
point_cloud_temp_2 = o3d.geometry.PointCloud()

num_frames = len(xyz_points_list)
step_to_skip = 1

#actual loop starts here
for n in range(0, (num_frames), step_to_skip):

    # create point cloud object
    current_point_cloud.points = o3d.utility.Vector3dVector(xyz_points_list[n])
    if n == 0 :
        vSet.addView(view_id, absTform=absTform, pointCloud=o3d.utility.Vector3dVector(xyz_points_list[n]))
        vSet.addPriorFactor(curr_viewID=view_id, prev_viewID=view_id,absTform=absTform)
        view_id = view_id + 1
        previous_point_cloud.points = current_point_cloud.points
        initTform = np.eye(4)
        continue

    relTform = o3d.registration.registration_icp(current_point_cloud, previous_point_cloud, max_correspondence_distance, initTform, o3d.registration.TransformationEstimationPointToPoint())
    evaluation = o3d.registration.evaluate_registration(current_point_cloud, previous_point_cloud, max_correspondence_distance,relTform.transformation)
   
    # for visulising the transformation
    #draw_registration_result(current_point_cloud, previous_point_cloud,relTform.transformation)
    
    # Calculate the absolute transform
    absTform = np.matmul(absTform,relTform.transformation)
    
    vSet.addView(view_id, absTform=absTform, pointCloud=o3d.utility.Vector3dVector(xyz_points_list[n]))
    vSet.addOdometryFactor(view_id, view_id-1, absTform, relTform.transformation)
    
    previous_point_cloud.points = current_point_cloud.points
    initTform = relTform.transformation

    def loopClosure():
        loop_view_id = None
        loop_closed = False
        loop_transform = None
        loop_fitness = 0
        if n > number_of_scans_to_ingnore :
            for i in range( (len(vSet.pcViewList) - number_of_scans_to_ingnore)):    
                view_temp = vSet.pcViewList[i]
                #print(view_temp["absTform"])
                absTform_temp = np.matmul(ln.inv(absTform),view_temp["absTform"])
                absTform_temp_norm = ln.norm(absTform_temp[0:3, 3])
                #print(absTform_temp_norm)
                if absTform_temp_norm  < norm_threshold and absTform_temp_norm > 0:
                    #print("norm: ")
                    #print(absTform_temp_norm)
                    point_cloud_temp = o3d.geometry.PointCloud()
                    point_cloud_temp.points = view_temp["pointCloud"]
                    relTform_temp = o3d.registration.registration_icp(current_point_cloud, point_cloud_temp, max_correspondence_distance, relTform.transformation, o3d.registration.TransformationEstimationPointToPoint())
                    evaluation = o3d.registration.evaluate_registration(current_point_cloud, point_cloud_temp, max_correspondence_distance,relTform_temp.transformation)
                    #draw_registration_result(current_point_cloud, point_cloud_temp, relTform_temp.transformation)
                    if evaluation.fitness > icp_fitness_threshold:
                        #draw_registration_result(current_point_cloud, point_cloud_temp, relTform_temp.transformation)
                        
                        if evaluation.fitness > loop_fitness:
                            loop_fitness = evaluation.fitness
                            loop_view_id =  view_temp["viewID"]
                            loop_transform = relTform_temp.transformation
                            print(n)
                            print("n: " +  str(n) +" current view id: " + str(view_id) + " pose list view id: " + str(loop_view_id) + 
                        " loop norm: "+ str(absTform_temp_norm) + " fitness: " + str(evaluation.fitness))
                            print("----------------------")
                        loop_closed = True
                        """
                        loop_view_id =  view_temp["viewID"]
                        loop_transform = relTform_temp.transformation
                        loop_closed = True
                        print("n: " +  str(n) +" current view id: " + str(view_id) + " pose list view id: " + str(loop_view_id) + 
                        " loop norm: "+ str(absTform_temp_norm) + " fitness: " + str(evaluation.fitness))
                        print("----------------------")
                        break
                        """
        return loop_closed, loop_view_id, loop_transform
    
    loop_closed, loop_view_id, loop_transform = loopClosure()
    print("after the loop closure")
    print(" pose list view id: " + str(loop_view_id))
    if loop_closed == True:
        vSet.addLoopFactor(view_id,loop_view_id, loop_transform)
        graph_optimised = vSet.optimizePoseGraph()
        vSet_opt = vSet.updatePointCloudViewSetWithOptimisedPose(graph_optimised)
    view_id = view_id + 1
    


vSet.publishGroundtruthAndOptimisedposes(ground_truth_camera_pose_path, graph_optimised)

#mapping

ground_truth_electrodes_in_first_camera_frame = vSet.mapGroundTruthElectrodes(ground_truth_camera_pose_path, ground_truth_electrode_path)
estimated_electrodes_in_first_camera_frame = vSet.mapElectrodes(vSet_opt)

vSet.clusterTheElectrodes_KMeans(ground_truth_electrodes_in_first_camera_frame,63, show=True)
vSet.clusterTheElectrodes_KMeans(estimated_electrodes_in_first_camera_frame,63,show=True)

vSet.getClusterMetrics(ground_truth_electrodes_in_first_camera_frame, estimated_electrodes_in_first_camera_frame,63,show=True)

#trajectory metrics
#opt_pose_list = vSet.getoptimisedPoseList(graph_optimised)
#vSet.getErrorMetrics(opt_pose_list, ground_truth_camera_pose_path,step_to_skip)

 