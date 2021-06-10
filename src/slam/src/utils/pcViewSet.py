#!/usr/bin/env python
import numpy as np
np.set_printoptions(precision=4)
import numpy.linalg as la
from scipy.linalg import qr
import csv
import minisam
import rospy
from geometry_msgs.msg import PoseStamped, TransformStamped
from nav_msgs.msg import Path, Odometry
from conversions import np2ros_poseStamped, ros2np_Pose
from pyquaternion import Quaternion
import tf.transformations as tf
import open3d as o3d
import copy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import DBSCAN, KMeans
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances_argmin, silhouette_score, davies_bouldin_score, homogeneity_completeness_v_measure, adjusted_rand_score
import time
from open3d_helper import convertCloudFromOpen3dToRos
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt



class pcViewSet(object):
    def __init__(self):
        #point cloud view set stuff
        self.pcViewList = []
        self.view = {"viewID" : 0 , "absTform" : np.eye(4), "pointCloud" : np.eye(4) }
        self.base_dir = "/home/Vishwanath/catkin_ws/src/master_arbeit/src/slam/src/data/robot_trajectory/CA_124_7/"
        # pose graph stuff
        self.prior_cov = minisam.DiagonalLoss.Sigmas(np.array([1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4])) 
        self.const_cov = np.array([0.5, 0.5, 0.5, 0.1, 0.1, 0.1])
        self.odom_cov = minisam.DiagonalLoss.Sigmas(self.const_cov)
        self.loop_cov = minisam.DiagonalLoss.Sigmas(self.const_cov)

        self.graph_factors = minisam.FactorGraph()
        self.graph_initials = minisam.Variables()

        self.opt_param = minisam.LevenbergMarquardtOptimizerParams()
        self.opt = minisam.LevenbergMarquardtOptimizer(self.opt_param)
        
        self.graph_optimized = None

        # ros stuff
        rospy.init_node('abs_pose_list', anonymous=True)
        self.pub_optimized = rospy.Publisher('optimized_path', Path, queue_size=1)
        self.pub_unoptimized = rospy.Publisher('un_optimized_path', Path, queue_size=1)
        self.pub_camera_ground_truth = rospy.Publisher('camera_ground_truth_path', Path, queue_size=1)
        self.Path = Path()
        self.Path_unoptimized = Path()
        self.Path_optimized = Path()
        self.Path_camera = Path()
        self.Transform_stamped_msg = TransformStamped()
        self.odom_msg = Odometry()
        self.electrode_position_list = o3d.geometry.PointCloud()

    # methods related to pc view set 

    def addView(self, viewID , absTform, pointCloud):
        view_temp = {"viewID" : viewID , "absTform" : absTform, "pointCloud" : pointCloud }
        self.pcViewList.append(view_temp)

    def deleteView(self, viewID):
        is_viewID_deleted = False
        idx = 0
        for i in range(len(self.pcViewList)):
            view_temp = self.pcViewList[i] 
            if view_temp["viewID"] == viewID :
                idx = i
        if (idx != 0):
            self.pcViewList.pop(idx)
            is_viewID_deleted = True   
            print("viewID: " + str(viewID) +" deleted")
        if not is_viewID_deleted :
            print("ViewID cannot be found")
    
    def plot(self):
        pose_list = []
        with open('/home/Vishwanath/catkin_ws/src/master_arbeit/src/slam/src/data/poses.csv', 'w') as file:
            writer = csv.writer(file)
            for i in range(len(self.pcViewList)):
                view_temp = self.pcViewList[i]
                absTform_temp = view_temp["absTform"]
                pose = absTform_temp[0:3, 0:4]
                pose_r = np.reshape(pose, (1,12))
                writer.writerows(pose_r)

    
    
    # methods related to pose graph

    def addPriorFactor(self, curr_viewID, prev_viewID, absTform):
        self.graph_initials.add(minisam.key('x', curr_viewID), minisam.SE3(absTform))
        self.graph_factors.add(minisam.PriorFactor(
                                                minisam.key('x', curr_viewID), 
                                                minisam.SE3(absTform), 
                                                self.prior_cov))

    def addOdometryFactor(self, curr_viewID, prev_viewID, absTform, odom_transform):
        self.graph_initials.add(minisam.key('x', curr_viewID), minisam.SE3(absTform))
        self.graph_factors.add(minisam.BetweenFactor(
                                                minisam.key('x', prev_viewID), 
                                                minisam.key('x', curr_viewID), 
                                                minisam.SE3(odom_transform), 
                                                self.odom_cov))

    def addLoopFactor(self, curr_viewID, loop_viewID,loop_transform):
        self.graph_factors.add(minisam.BetweenFactor(
                                            minisam.key('x', loop_viewID), 
                                            minisam.key('x', curr_viewID),  
                                            minisam.SE3(loop_transform), 
                                            self.loop_cov))
    def getGraphNodePose(self, graph, viewID):
        pose = np.eye(4)
        graph_pose = graph.at(minisam.key('x', viewID))
        pose [0:3,3] = graph_pose.translation()
        pose [0:3, 0:3] = graph_pose.so3().matrix()
        return pose
    
    def optimizePoseGraph(self):
        self.graph_optimized = minisam.Variables()
        status = self.opt.optimize(self.graph_factors, self.graph_initials, self.graph_optimized)
        if status != minisam.NonlinearOptimizationStatus.SUCCESS:
            print("optimization error: ", status)
        else:
            print("SUCCESS: Pose Graph optimised!")
            return self.graph_optimized 
    
    # un optimised stuff
    def getUnOptimisedPoseList(self):
        pose_list = []
        for i in range(len(self.pcViewList)):
            view_temp = self.pcViewList[i]
            absTform_temp = view_temp["absTform"]
            pose_list.append(absTform_temp)
        return pose_list

    def saveUnOptimisedPoseList(self, path):
        np.save(str(path), self.getUnOptimisedPoseList())
        #print("Unoptimised pose List saved at: " + path)
    
    def getUnOptimisedPoseStampedList(self):
        unOptimised_poseStamped_list = []
        pose_list = self.getUnOptimisedPoseList()
        for i in range(len(pose_list)):
            pose = pose_list[i]
            unOptimised_poseStamped_list.append(np2ros_poseStamped(pose))
        return unOptimised_poseStamped_list
    
    def publishUnOptimisedpose(self):
        unOptimised_poseStamped_list = self.getUnOptimisedPoseStampedList()
        rate = rospy.Rate(10)
        rospy.loginfo("publishing un-optimised poses...")
        while not rospy.is_shutdown():
            for i in range (len(unOptimised_poseStamped_list)):
                self.Path_unoptimized.header.stamp = rospy.Time.now()
                self.Path_unoptimized.header.frame_id = "/map"
                self.Path_unoptimized.poses.append(unOptimised_poseStamped_list[i])
                self.pub_unoptimized.publish(self.Path_unoptimized)
            rate.sleep()

    
    # optimised stuff
    def getoptimisedPoseList(self, graph_optimised):
        optimised_pose_list = []
        for i in range(len(self.pcViewList)):
            pose = self.getGraphNodePose(graph_optimised, i)
            optimised_pose_list.append(pose)
        
        return optimised_pose_list
    
    def saveOptimisedPoseList(self, graph_optimised, path):
        optimised_pose_list = self.getoptimisedPoseList(graph_optimised)
        np.save(str(path), optimised_pose_list)
        #print("Optimised pose List saved at: " + path)    

    def getOptimisedPoseStampedList(self, graph_optimised):
        optimised_poseStamped_list = []
        for i in range(len(self.pcViewList)):
            pose = self.getGraphNodePose(graph_optimised, i)
            optimised_poseStamped_list.append(np2ros_poseStamped(pose))
        return optimised_poseStamped_list

    def publishOptimisedpose(self, graph_optimised):
        optimised_poseStamped_list = self.getOptimisedPoseStampedList(graph_optimised=graph_optimised)
        rate = rospy.Rate(10)
        rospy.loginfo("publishing optimised poses...")
        while not rospy.is_shutdown():
            for i in range (len(optimised_poseStamped_list)):
                self.Path_optimized.header.stamp = rospy.Time.now()
                self.Path_optimized.header.frame_id = "/map"
                self.Path_optimized.poses.append(optimised_poseStamped_list[i])
                self.pub_optimized.publish(self.Path_optimized)
            rate.sleep()
   
    def updatePointCloudViewSetWithOptimisedPose(self, graph_optimised):
        optimised_pose_list = self.getoptimisedPoseList(graph_optimised)
        for i in range(len(self.pcViewList)):
           view_temp = self.pcViewList[i]
           #print("before update")
           #print(view_temp["absTform"])
           view_temp["absTform"] = optimised_pose_list[i]
           self.pcViewList[i] = view_temp
           #view_temp = self.pcViewList[i]
           #print("after update")
           #print(view_temp["absTform"])
        return self.pcViewList 

    
    def mapElectrodes(self, pcViewList):
        electrode_position_list = o3d.geometry.PointCloud()
        electrode_position = o3d.geometry.PointCloud()
        electrode_Transformed = o3d.geometry.PointCloud()
        for i in range(len(pcViewList)):
            view_temp = pcViewList[i]
            electrode_position.points = view_temp["pointCloud"]
            electrode_Transformed = copy.deepcopy(electrode_position).transform(view_temp["absTform"])
            electrode_position_list = electrode_position_list + electrode_Transformed
        #o3d.visualization.draw_geometries([electrode_position_list])
        np.save(self.base_dir + "results/estimated_electrodes_in_first_camera_frame.npy", electrode_position_list)
        return electrode_position_list

    def concatenate_electrodes(self, pcViewList):
        electrode_position_list = o3d.geometry.PointCloud()
        electrode_position = o3d.geometry.PointCloud()
        electrode_Transformed = o3d.geometry.PointCloud()
        for i in range(len(pcViewList)):
            view_temp = pcViewList[i]
            electrode_position.points = view_temp["pointCloud"]
            electrode_Transformed = copy.deepcopy(electrode_position).transform(view_temp["absTform"])
            electrode_position_list = electrode_position_list + electrode_Transformed
        all_electrodes_pointcloud2 = convertCloudFromOpen3dToRos(electrode_position_list)
        return all_electrodes_pointcloud2

    def publish_estimated_electrodes (self, index, pcViewList, pub_electrode, pub_pose, loop_closed):
        
        if loop_closed == True:
            self.electrode_position_list = o3d.geometry.PointCloud()
            electrode_position = o3d.geometry.PointCloud()
            electrode_Transformed = o3d.geometry.PointCloud()
            for i in range(len(pcViewList)):
                view_temp = pcViewList[i]
                electrode_position.points = view_temp["pointCloud"]
                electrode_Transformed = copy.deepcopy(electrode_position).transform(view_temp["absTform"])
                self.electrode_position_list = self.electrode_position_list + electrode_Transformed
            #pub_electrode.publish(convertCloudFromOpen3dToRos(self.electrode_position_list))
            #pose_temp = np2ros_poseStamped(view_temp["absTform"])
            #pose_temp.header.frame_id = "/map" 
            #pub_pose.publish(pose_temp)
            rospy.loginfo("remapping due to loop closure")
        electrode_position = o3d.geometry.PointCloud()
        view_temp = pcViewList[index]
        electrode_position.points = view_temp["pointCloud"]
        electrode_Transformed = copy.deepcopy(electrode_position).transform(view_temp["absTform"])
        self.electrode_position_list = self.electrode_position_list + electrode_Transformed
        pub_electrode.publish(convertCloudFromOpen3dToRos(self.electrode_position_list))
        pose_temp = np2ros_poseStamped(view_temp["absTform"])
        pose_temp.header.frame_id = "/map" 
        pub_pose.publish(pose_temp)
        
    def errorMetrics(self, pose_source, pose_target):
        u,s,v = la.svd(pose_source[0:3,0:3])
        mat_unitary = np.matmul(u,v)
        pose_source[0:3,0:3] = mat_unitary[0:3,0:3]

        u,s,v = la.svd(pose_target[0:3,0:3])
        mat_unitary = np.matmul(u, v)
        pose_target[0:3,0:3] = mat_unitary[0:3,0:3]

        pose_source_qt = Quaternion(matrix=pose_source)
        pose_target_qt = Quaternion(matrix=pose_target)
        error_qt = pose_source_qt * pose_target_qt.inverse
        rot_error_qt_angle = np.abs(error_qt.degrees)
        rot_error_qt_axis = error_qt.get_axis
        #translation_error = la.norm(pose_source[0:3,3] - pose_target[0:3,3])
        translation_error = (la.norm(pose_source[0:3,3]) - la.norm(pose_target[0:3,3]) )

        return rot_error_qt_angle , translation_error
    
    def errorMetrics_only_translation(self, pose_source, pose_target):
        translation_error = (la.norm(pose_source[0:3,3]) - la.norm(pose_target[0:3,3]) )
        return translation_error
    

    def getGroundtruthCameraPosition(self, path_to_camera_poses):
        'ground truth camera pose in first camera position'
        abs_camera_pose_list = []
        camera_poses = np.load(str(path_to_camera_poses),allow_pickle=True)
        for i in range(len(camera_poses)) :
            if i == 0:
                firstCameraFrame_T_currentCameraFrame = np.eye(4)
                abs_camera_pose_list.append(firstCameraFrame_T_currentCameraFrame)
                continue
            
            if np.shape(camera_poses[i]) == (4,4):
                base_T_camera = camera_poses[i]
                prev_camera_pose = camera_poses[i-1]
            else:
                base_T_camera = ros2np_Pose(camera_poses[i])
                prev_camera_pose = ros2np_Pose(camera_poses[i-1])

            temp_matrix = np.matmul(firstCameraFrame_T_currentCameraFrame, la.inv(prev_camera_pose))
            firstCameraFrame_T_currentCameraFrame = np.matmul(temp_matrix,base_T_camera)
            abs_camera_pose_list.append(firstCameraFrame_T_currentCameraFrame) 
        return abs_camera_pose_list 
    
    def publishGroundTruthCameraPosition(self, path_to_camera_poses):
        #camera_pose_list = self.getGroundtruthCameraPosition(path_to_camera_poses)
        camera_pose_list = np.load(str(path_to_camera_poses), allow_pickle=True)
        poseStamped_list = []
        for i in range (len(camera_pose_list)):
            poseStamped_list.append(np2ros_poseStamped(camera_pose_list[i]))
        rate = rospy.Rate(10)
        rospy.loginfo("publishing ground truth poses...")
        while not rospy.is_shutdown():
            for i in range (len(camera_pose_list)):
                self.Path_camera.header.stamp = rospy.Time.now()
                self.Path_camera.header.frame_id = "/map"
                self.Path_camera.poses.append(poseStamped_list[i])
                self.pub_camera_ground_truth.publish(self.Path_camera)
            rate.sleep()


    def getErrorMetrics(self, calc_pose_list, ground_truth_pose_path, step_to_skip, plot=True):
        #TODO make is consistant (generalise this function) either with the pose list directly or with the pose paths
        #ground_truth_pose_list = self.getGroundtruthCameraPosition(ground_truth_pose_path)
        ground_truth_pose_list = np.load(str(ground_truth_pose_path), allow_pickle = True)
        #TODO remove this or make it right
        calc_pose_list = np.load(str(calc_pose_list), allow_pickle = True)
        rot_error_list = []
        trans_error_list = []
        pose_index_list = []
        print(len(calc_pose_list))
        print(len(ground_truth_pose_list))
        for i in range (0,len(calc_pose_list), step_to_skip):
            rot_error, trans_error = self.errorMetrics(calc_pose_list[i], ground_truth_pose_list[i])
            rot_error_list.append(rot_error)
            trans_error_list.append(trans_error)
            pose_index_list.append(i)
        trans_mean = np.round(np.mean(trans_error_list),4)
        trans_stddev = np.round(np.std(trans_error_list),4)
        rot_mean = np.round(np.mean(rot_error_list),4)
        rot_stddev = np.round(np.std(rot_error_list),4)
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        
        if plot == False:
            fig.suptitle("ICP vs Ground Truth_signed ")
            ax1.plot(pose_index_list, trans_error_list, 'o', color = "black" , label = "Translational Error: mean %s $\pm$ %s " %(trans_mean, trans_stddev))
            ax1.legend()
            ax2.plot(pose_index_list, rot_error_list, 'v', color = "blue", label = "Rotational Error: mean %s $\pm$ %s " %(rot_mean, rot_stddev))
            ax2.legend()
            plt.savefig(self.base_dir + "results/ICP vs Ground Truth_signed.png") 
        if plot == True:
            fig.suptitle("absolute vs Ground Truth_signed ")
            ax1.plot(pose_index_list, trans_error_list, 'o', color = "black" , label = "Translational Error: mean %s $\pm$ %s " %(trans_mean, trans_stddev))
            ax1.legend()
            ax2.plot(pose_index_list, rot_error_list, 'v', color = "blue", label = "Rotational Error: mean %s $\pm$ %s " %(rot_mean, rot_stddev))
            ax2.legend()
            plt.savefig(self.base_dir + "results/absolute vs Ground Truth_signed.png") 
        
        plt.show()
    
    def getErrorMetrics_hand_trajectory(self, calc_pose_list, ground_truth_pose_path, step_to_skip, plot=True):
        #TODO make is consistant (generalise this function) either with the pose list directly or with the pose paths
        #ground_truth_pose_list = self.getGroundtruthCameraPosition(ground_truth_pose_path)
        ground_truth_pose_list = np.load(str(ground_truth_pose_path), allow_pickle = True)
        #TODO remove this or make it right
        calc_pose_list = np.load(str(calc_pose_list), allow_pickle = True)
        rot_error_list = []
        trans_error_list = []
        pose_index_list = []
        for i in range (0,len(calc_pose_list), step_to_skip):
            calc_temp = calc_pose_list[i]
            #gt_temp = ground_truth_pose_list[np.int(2.12*i)]
            gt_temp = ground_truth_pose_list[i]
            
            rot_error, trans_error = self.errorMetrics(calc_temp, gt_temp)
            if(gt_temp[2,3])  < 0.001 or  np.abs(rot_error) > 50:
                continue
            else:
                rot_error_list.append(rot_error)
                trans_error_list.append(trans_error)
                pose_index_list.append(i)
        #print(trans_error_list)
        #print(rot_error_list)
        trans_mean = np.round(np.mean(trans_error_list),4)
        trans_stddev = np.round(np.std(trans_error_list),4)
        rot_mean = np.round(np.mean(rot_error_list),4)
        rot_stddev = np.round(np.std(rot_error_list),4)
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        
        if plot == False:
            fig.suptitle("ICP vs Ground Truth_signed ")
            ax1.plot(pose_index_list, trans_error_list, 'o', color = "black" , label = "Translational Error: mean %s $\pm$ %s " %(trans_mean, trans_stddev))
            ax1.legend()
            ax2.plot(pose_index_list, rot_error_list, 'v', color = "blue", label = "Rotational Error: mean %s $\pm$ %s " %(rot_mean, rot_stddev))
            ax2.legend()
            plt.savefig(self.base_dir + "results/ICP vs Ground Truth_signed.png") 
        if plot == True:
            fig.suptitle("absolute vs Ground Truth_signed ")
            ax1.plot(pose_index_list, trans_error_list, 'o', color = "black" , label = "Translational Error: mean %s $\pm$ %s " %(trans_mean, trans_stddev))
            ax1.legend()
            ax2.plot(pose_index_list, rot_error_list, 'v', color = "blue", label = "Rotational Error: mean %s $\pm$ %s " %(rot_mean, rot_stddev))
            ax2.legend()
            plt.savefig(self.base_dir + "results/absolute vs Ground Truth_signed.png") 
        
        plt.show()
    
    def getErrorMetrics_hand_trajectory_ICP(self, calc_pose_list, ground_truth_pose_path, step_to_skip, plot=True):
        #TODO make is consistant (generalise this function) either with the pose list directly or with the pose paths
        #ground_truth_pose_list = self.getGroundtruthCameraPosition(ground_truth_pose_path)
        ground_truth_pose_list = np.load(str(ground_truth_pose_path), allow_pickle = True)
        #TODO remove this or make it right
        calc_pose_list = np.load(str(calc_pose_list), allow_pickle = True)
        rot_error_list = []
        trans_error_list = []
        pose_index_list = []
        print(len(calc_pose_list))
        print(len(ground_truth_pose_list))
        for i in range (0,len(calc_pose_list), step_to_skip):
            calc_temp = calc_pose_list[i]
            gt_temp = ground_truth_pose_list[i]
            rot_error, trans_error = self.errorMetrics(calc_temp, gt_temp)
            if np.abs(gt_temp[2,3]) == 0.0 or np.abs(gt_temp[2,3]) > 1.0 or np.abs(rot_error) > 50:
                #print(np.abs(gt_temp[2,3]))
                continue
            else:
                rot_error_list.append(rot_error)
                trans_error_list.append(trans_error)
                pose_index_list.append(i)
        #print(trans_error_list)
        #print(rot_error_list)
        trans_mean = np.round(np.mean(trans_error_list),4)
        trans_stddev = np.round(np.std(trans_error_list),4)
        rot_mean = np.round(np.mean(rot_error_list),4)
        rot_stddev = np.round(np.std(rot_error_list),4)
        fig, (ax1, ax2) = plt.subplots(2, sharex=True)
        
        if plot == False:
            fig.suptitle("ICP vs Ground Truth_signed ")
            ax1.plot(pose_index_list, trans_error_list, 'o', color = "black" , label = "Translational Error: mean %s $\pm$ %s " %(trans_mean, trans_stddev))
            ax1.legend()
            ax2.plot(pose_index_list, rot_error_list, 'v', color = "blue", label = "Rotational Error: mean %s $\pm$ %s " %(rot_mean, rot_stddev))
            ax2.legend()
            plt.savefig(self.base_dir + "results/ICP vs Ground Truth_signed.png") 
        if plot == True:
            fig.suptitle("absolute vs Ground Truth_signed ")
            ax1.plot(pose_index_list, trans_error_list, 'o', color = "black" , label = "Translational Error: mean %s $\pm$ %s " %(trans_mean, trans_stddev))
            ax1.legend()
            ax2.plot(pose_index_list, rot_error_list, 'v', color = "blue", label = "Rotational Error: mean %s $\pm$ %s " %(rot_mean, rot_stddev))
            ax2.legend()
            plt.savefig(self.base_dir + "results/absolute vs Ground Truth_signed.png") 
        
        plt.show()
        
    
    def publishGroundtruthAndOptimisedposes(self, path_to_camera_poses, graph_optimised):
        camera_pose_list = self.getGroundtruthCameraPosition(path_to_camera_poses)
        optimised_poseStamped_list = self.getOptimisedPoseStampedList(graph_optimised=graph_optimised)
        poseStamped_list = []
        for i in range (len(camera_pose_list)):
            poseStamped_list.append(np2ros_poseStamped(camera_pose_list[i]))
        rate = rospy.Rate(10)
        rospy.loginfo("publishing ground truth and optimised poses...")
        while not rospy.is_shutdown():
            for i in range (len(optimised_poseStamped_list)):
                self.Path_optimized.header.stamp = rospy.Time.now()
                self.Path_optimized.header.frame_id = "/map"
                self.Path_optimized.poses.append(optimised_poseStamped_list[i])
                
                self.Path_camera.header.stamp = rospy.Time.now()
                self.Path_camera.header.frame_id = "/map"
                self.Path_camera.poses.append(poseStamped_list[i])
                
                self.pub_optimized.publish(self.Path_optimized)
                self.pub_camera_ground_truth.publish(self.Path_camera)
            rate.sleep()       

    def mapGroundTruthElectrodes(self, ground_truth_camera_pose_path, ground_truth_electrodes_path):
        ground_truth_camera_poses = self.getGroundtruthCameraPosition(ground_truth_camera_pose_path)
        ground_truth_electrodes = np.load(str(ground_truth_electrodes_path), allow_pickle=True)
        electrode_position_list = o3d.geometry.PointCloud()
        electrode_position = o3d.geometry.PointCloud()
        electrode_Transformed = o3d.geometry.PointCloud()
        rospy.loginfo("mapping ground truth")
        
        #for i in range(0, 100, 1):
        
        for i in range(len(ground_truth_camera_poses)):
            electrode_position.points = o3d.utility.Vector3dVector(ground_truth_electrodes[i])
            electrode_Transformed = copy.deepcopy(electrode_position).transform(ground_truth_camera_poses[i])
            electrode_position_list = electrode_position_list + electrode_Transformed
        o3d.visualization.draw_geometries([electrode_position_list]) 
        #np.save("/home/Vishwanath/catkin_ws/src/master_arbeit/src/slam/src/data/CA_124_1/results/ground_truth_electrodes_in_first_camera_frame.npy",electrode_position_list)
        np.save(self.base_dir + "results/ground_truth_camera_pose_first_frame.npy", ground_truth_camera_poses)
        return electrode_position_list  


    def get_centroid(self, cluster):
        """calculate the centroid of a cluster of geographic coordinate points
        Args:
        cluster coordinates, nx2 array-like (array, list of lists, etc)
        n is the number of points(latitude, longitude)in the cluster.
        Return:
        geometry centroid of the cluster

        """
        cluster_ary = np.asarray(cluster)
        centroid = cluster_ary.mean(axis=0)
        return centroid
    
    def clusterTheElectrodes_DBSCAN(self, all_elecrodes_in_first_camera_frame):
        dataset = np.asarray(all_elecrodes_in_first_camera_frame.points)
        # Compute DBSCAN
        #scaler = StandardScaler()
        #dataset_scaled = scaler.fit_transform(dataset)
        
        db = DBSCAN(eps=0.010, min_samples=20).fit(dataset) # 30 samples is correct
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print('Estimated number of clusters: %d' % n_clusters_)
        #print('Estimated number of noise points: %d' % n_noise_)

        dbsc_clusters = pd.Series([dataset[labels==n] for n in  range(n_clusters_)])
        cluster_centroids = dbsc_clusters.map(self.get_centroid)

        """
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
        print("Adjusted Rand Index: %0.3f"
            % metrics.adjusted_rand_score(labels_true, labels))
        print("Adjusted Mutual Information: %0.3f"
            % metrics.adjusted_mutual_info_score(labels_true, labels))
        print("Silhouette Coefficient: %0.3f"
            % metrics.silhouette_score(X, labels))
        """
        # #############################################################################
        #Plot result
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=[100,50])
        ax = fig.add_subplot(111, projection='3d')
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0,1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = [0,0,0,1]
            class_member_mask = (labels == k)
            xyz = dataset[class_member_mask & core_samples_mask]

            ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c = col, marker="." )
            
        for i in range(n_clusters_):
            center = cluster_centroids[i]
            ax.scatter(center[0], center[1], center[2], c = col, s=200, marker="." )
        """
        cluster_centroids_gt = np.load(self.base_dir + "results/k_means_cluster_centers_ground_truth.npy", allow_pickle=True)

        for i in range(len(cluster_centroids_gt)):
            center = cluster_centroids_gt[i]
            ax.scatter(center[0], center[1], center[2], c = col, marker="+" )
        """
        plt.title('Estimated number of cluster: %d' %n_clusters_)
        #plt.show()
        return n_clusters_

        
    def clusterTheElectrodes_KMeans(self, all_elecrodes_in_first_camera_frame, num_cluster, show=False):
        X = np.asarray(all_elecrodes_in_first_camera_frame.points)

        k_means = KMeans(init='k-means++', n_clusters=num_cluster, n_init=10)
        t0 = time.time()
        k_means.fit(X)
        t_batch = time.time() - t0
        
        k_means_cluster_centers = k_means.cluster_centers_
        print(k_means.inertia_)

        k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)

        computed_silhouette_score = silhouette_score(X, k_means_labels, metric='euclidean')
        computed_davies_bouldin_index = davies_bouldin_score(X,k_means_labels)

        # #############################################################################
        # Plot result
        if show == True:
            import matplotlib.pyplot as plt
            
            fig = plt.figure(figsize=[100,50])
            ax = fig.add_subplot(111)
            unique_labels = set(k_means_labels)
            colors = [plt.cm.Spectral(each) for each in np.linspace(0,1, len(unique_labels))]
            
            for k, col in zip(range(num_cluster), colors):
                my_members = (k_means_labels == k)
        
                cluster_center = k_means_cluster_centers[k]
                
                ax.scatter(X[my_members, 0], X[my_members, 1],  c=col, s=100, marker='.')
                ax.scatter(cluster_center[0], cluster_center[1], c=col , s=250, marker='+')

            #plt.title('Clustering and its centers Silhouette score: %s (perfect : 1) , Davies Bouldin score: %s (perfect : 0)'  %( np.round(computed_silhouette_score,2) , np.round(computed_davies_bouldin_index,2) ))
            
            cluster_centroids_gt = np.load(self.base_dir + "results/dbsc_cluster_centers_ground_truth_no_order.npy", allow_pickle =True)
            #"""
            for i in range(len(cluster_centroids_gt)):
                center = cluster_centroids_gt[i]
                ax.scatter(center[0], center[1], c = "k", s=200, marker="1" )
            #"""
            plt.show()
        
        return k_means_cluster_centers, k_means_labels
    
    def clusterTheElectrodes_KMeans_depth(self, all_elecrodes_in_first_camera_frame, num_cluster, path_to_save, show=False):
        X = np.asarray(all_elecrodes_in_first_camera_frame.points)

        k_means = KMeans(init='k-means++', n_clusters=num_cluster, n_init=10)
        t0 = time.time()
        k_means.fit(X)
        t_batch = time.time() - t0
        
        k_means_cluster_centers = k_means.cluster_centers_
        #print(k_means.inertia_)
        np.save(str(path_to_save), k_means_cluster_centers)
        k_means_labels = pairwise_distances_argmin(X, k_means_cluster_centers)

        computed_silhouette_score = silhouette_score(X, k_means_labels, metric='euclidean')
        computed_davies_bouldin_index = davies_bouldin_score(X,k_means_labels)

        # #############################################################################
        # Plot result
        if show == True:
            import matplotlib.pyplot as plt
            
            fig = plt.figure(figsize=[100,50])
            ax = fig.add_subplot(111, projection='3d')
            unique_labels = set(k_means_labels)
            colors = [plt.cm.Spectral(each) for each in np.linspace(0,1, len(unique_labels))]
            
            for k, col in zip(range(num_cluster), colors):
                my_members = (k_means_labels == k)
        
                cluster_center = k_means_cluster_centers[k]
                
                ax.scatter(X[my_members, 0], X[my_members, 1], X[my_members, 2], c=col, s=100, marker='.')
                ax.scatter(cluster_center[0], cluster_center[1], cluster_center[2], c=col , s=250, marker='+')

            #plt.title('Clustering and its centers Silhouette score: %s (perfect : 1) , Davies Bouldin score: %s (perfect : 0)'  %( np.round(computed_silhouette_score,2) , np.round(computed_davies_bouldin_index,2) ))
            
            #cluster_centroids_gt = np.load(self.base_dir + "results/k_means_cluster_centers_ground_truth_transformed.npy", allow_pickle =True)
            """
            for i in range(len(cluster_centroids_gt)):
                center = cluster_centroids_gt[i]
                ax.scatter(center[0], center[1], center[2], c = "k", s=200, marker="1" )
            """
            plt.show()
        
        return k_means_cluster_centers, k_means.inertia_

    def publish_cluster_centers(self, cluster_centers, pub_pointCloud2):
        cluster = o3d.geometry.PointCloud()
        cluster.points = o3d.utility.Vector3dVector(np.asarray(cluster_centers))
        cluster = convertCloudFromOpen3dToRos(cluster)
        pub_pointCloud2.publish(cluster)

    def getClusterMetrics_kMeans(self,ground_truth_electrodes_in_first_camera_frame, estimated_electrodes_in_first_camera_frame, num_cluster, method ="KMeans", show=False):
        ground_truth_data = np.asarray(ground_truth_electrodes_in_first_camera_frame.points)
        estimated_data = np.asarray(estimated_electrodes_in_first_camera_frame.points)

        k_means_ground_truth = KMeans(init='k-means++', n_clusters=63, n_init=15)
        k_means_ground_truth.fit(ground_truth_data)

        k_means_estimated = KMeans(init='k-means++', n_clusters=num_cluster, n_init=15)
        k_means_estimated.fit(estimated_data)
        
        k_means_cluster_centers_estimated = k_means_estimated.cluster_centers_
        k_means_cluster_centers_ground_truth = k_means_ground_truth.cluster_centers_
        np.save(self.base_dir + "results/k_means_cluster_centers_ground_truth_no_order.npy", k_means_cluster_centers_ground_truth)

        order = pairwise_distances_argmin(k_means_cluster_centers_estimated, k_means_cluster_centers_ground_truth)
        k_means_cluster_centers_ground_truth = k_means_ground_truth.cluster_centers_[order]
        
        
        #print(k_means_cluster_centers_estimated[c>1,:])
        np.save(self.base_dir + "results/k_means_cluster_centers_estimated.npy", k_means_cluster_centers_estimated)
        np.save(self.base_dir + "results/k_means_cluster_centers_ground_truth.npy", k_means_cluster_centers_ground_truth)

        k_means_labels_ground_truth = pairwise_distances_argmin(ground_truth_data, k_means_cluster_centers_ground_truth)
        k_means_labels_estimated = pairwise_distances_argmin(estimated_data, k_means_cluster_centers_estimated)
        #computed_HCV = homogeneity_completeness_v_measure(k_means_labels_ground_truth, k_means_labels_estimated)


        if show == True:
            import matplotlib.pyplot as plt
                
            fig = plt.figure(figsize=[100,50])
            ax = fig.add_subplot(111)
            unique_labels = set(k_means_labels_estimated)
            colors = [plt.cm.Spectral(each) for each in np.linspace(0,1, len(unique_labels))]
            error_list = []
            for k, col in zip(range(num_cluster), colors):
                cluster_center_ground_truth = k_means_cluster_centers_ground_truth[k]
                cluster_center_estimated = k_means_cluster_centers_estimated[k]
                my_members = (k_means_labels_estimated == k)

                
                ax.scatter(estimated_data[my_members, 0], estimated_data[my_members, 1], c=col, s=100,marker='.')
                error_list.append(  la.norm(cluster_center_ground_truth[:]- cluster_center_estimated[:])   )
                #print('ground: %s' %cluster_center_ground_truth[:])
                #print(k)
                #print('estimated: %s' %cluster_center_estimated[:])

                #print('norm: %s ' %la.norm(cluster_center_ground_truth[:]-cluster_center_estimated[:]) )

            ax.scatter(k_means_cluster_centers_ground_truth[:,0], k_means_cluster_centers_ground_truth[:,1], marker='+', s=100,label="Ground Truth")
            ax.scatter(k_means_cluster_centers_estimated[:,0],k_means_cluster_centers_estimated[:,1], marker='1', s=100,label="Estimated")
            
            error_mean = np.round(np.mean(error_list),5)
            error_stddev = np.round(np.std(error_list),3)

            #plt.title('mean %s $\pm$ %s , Homogeneity : %s , completeness : %s , Vmeasure score : %s'  %(error_mean, error_stddev, computed_HCV[0], computed_HCV[1], computed_HCV[2]) )
            plt.title('ground truth vs estimated cluster centers, mean %s $\pm$ %s ' %(error_mean, error_stddev) )
            ax.legend()
            plt.legend()
            fig.savefig(self.base_dir + "results/ground truth vs estimated cluster centers.png")
            plt.show()
            #print(error_mean, error_stddev)
            

        else:
            print('Cluster centers of both ground truth and estimated electrode position, mean %s $\pm$ %s'  %(error_mean, error_stddev))
        
    
    
    def cluster_visualization(self,ground_truth_electrodes_in_first_camera_frame, estimated_electrodes_in_first_camera_frame, num_cluster, method ="KMeans", show=False):
        ground_truth_data = np.asarray(ground_truth_electrodes_in_first_camera_frame.points)
        estimated_data = np.asarray(estimated_electrodes_in_first_camera_frame.points)

        k_means_ground_truth = KMeans(init='k-means++', n_clusters=63, n_init=15)
        k_means_ground_truth.fit(ground_truth_data)

        k_means_estimated = KMeans(init='k-means++', n_clusters=num_cluster, n_init=15)
        k_means_estimated.fit(estimated_data)
        
        k_means_cluster_centers_estimated = k_means_estimated.cluster_centers_
        k_means_cluster_centers_ground_truth = k_means_ground_truth.cluster_centers_

        order = pairwise_distances_argmin(k_means_cluster_centers_estimated, k_means_cluster_centers_ground_truth)
        k_means_cluster_centers_ground_truth = k_means_ground_truth.cluster_centers_[order]
        
        #np.save(self.base_dir + "depth_variance/k_means_cluster_centers_estimated.npy", k_means_cluster_centers_estimated)

        k_means_labels_ground_truth = pairwise_distances_argmin(ground_truth_data, k_means_cluster_centers_ground_truth)
        k_means_labels_estimated = pairwise_distances_argmin(estimated_data, k_means_cluster_centers_estimated)
        #computed_HCV = homogeneity_completeness_v_measure(k_means_labels_ground_truth, k_means_labels_estimated)


        if show == True:
            import matplotlib.pyplot as plt
                
            fig = plt.figure(figsize=[100,50])
            ax = fig.add_subplot(111)
            unique_labels = set(k_means_labels_estimated)
            colors = [plt.cm.Spectral(each) for each in np.linspace(0,1, len(unique_labels))]
            error_list = []
            for k, col in zip(range(num_cluster), colors):
                cluster_center_ground_truth = k_means_cluster_centers_ground_truth[k]
                cluster_center_estimated = k_means_cluster_centers_estimated[k]
                my_members = (k_means_labels_estimated == k)

                ax.scatter(estimated_data[my_members, 0], estimated_data[my_members, 1],c=col, s=100,marker='.')
                error_list.append(  la.norm(cluster_center_ground_truth[:]- cluster_center_estimated[:])   )


            ax.scatter(k_means_cluster_centers_ground_truth[:,0], k_means_cluster_centers_ground_truth[:,1],  marker='+', s=200,label="Ground Truth")
            ax.scatter(k_means_cluster_centers_estimated[:,0],k_means_cluster_centers_estimated[:,1], marker='1', s=200,label="Estimated")
            
            error_mean = np.round(np.mean(error_list),5)
            error_stddev = np.round(np.std(error_list),3)

            #plt.title('mean %s $\pm$ %s , Homogeneity : %s , completeness : %s , Vmeasure score : %s'  %(error_mean, error_stddev, computed_HCV[0], computed_HCV[1], computed_HCV[2]) )
            plt.title('ground truth vs estimated cluster centers, mean %s $\pm$ %s ' %(error_mean, error_stddev) )
            ax.legend()
            plt.legend()
            
            plt.show()
            print(error_mean, error_stddev)
            

        else:
            print('Cluster centers of both ground truth and estimated electrode position, mean %s $\pm$ %s'  %(error_mean, error_stddev))
    
    
    
    

    def getClusterMetrics_DBSCAN(self,ground_truth_electrodes_in_first_camera_frame, estimated_electrodes_in_first_camera_frame, show=False):
        ground_truth_data = np.asarray(ground_truth_electrodes_in_first_camera_frame.points)
        estimated_data = np.asarray(estimated_electrodes_in_first_camera_frame.points)
        
        
        db_estimated = DBSCAN(eps=0.010, min_samples=5).fit(estimated_data)
        core_samples_mask = np.zeros_like(db_estimated.labels_, dtype=bool)
        core_samples_mask[db_estimated.core_sample_indices_] = True
        labels_estimated = db_estimated.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_estimated = len(set(labels_estimated)) - (1 if -1 in labels_estimated else 0)
        n_noise_ = list(labels_estimated).count(-1)

        # print('Estimated number of clusters: %d' % n_clusters_estimated)
        # print('Estimated number of noise points: %d' % n_noise_)

        dbsc_clusters_estimated = pd.Series([estimated_data[labels_estimated==n] for n in  range(n_clusters_estimated)])
        cluster_centroids_estimated = dbsc_clusters_estimated.map(self.get_centroid)

        #gt
        db_gt = DBSCAN(eps=0.010, min_samples=1).fit(ground_truth_data)
        core_samples_mask_gt= np.zeros_like(db_gt.labels_, dtype=bool)
        core_samples_mask_gt[db_gt.core_sample_indices_] = True
        labels_gt = db_gt.labels_


        n_clusters_gt = len(set(labels_gt)) - (1 if -1 in labels_gt else 0)

        dbsc_clusters_gt = pd.Series([ground_truth_data[labels_gt==n] for n in  range(n_clusters_gt)])
        cluster_centroids_gt = dbsc_clusters_gt.map(self.get_centroid)
        np.save(self.base_dir + "results/dbsc_cluster_centers_ground_truth_no_order.npy", cluster_centroids_gt)

        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=[100,50])
        ax = fig.add_subplot(111, projection='3d')
        unique_labels = set(labels_estimated)
        colors = [plt.cm.Spectral(each) for each in np.linspace(0,1, len(unique_labels))]
        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = [0,0,0,1]
            class_member_mask = (labels_estimated == k)
            xyz = estimated_data[class_member_mask & core_samples_mask]

            ax.scatter(xyz[:,0], xyz[:,1], xyz[:,2], c = col, marker="." )
             
        for i in range(n_clusters_estimated):
            center = cluster_centroids_estimated[i]
            ax.scatter(center[0], center[1], center[2], c = col, s=200, marker="." )
        
        for i in range(n_clusters_gt):
            center = cluster_centroids_gt[i]
            ax.scatter(center[0], center[1], center[2], c = col, s=200, marker="+" )
        
        plt.title('Estimated number of cluster: %d' %n_clusters_estimated)
        plt.show()
    
    
    # hand trajectory stuff
    def get_ground_truth_electrodes_in_first_camera_frame(self, path_to_np_array):
        gt_electrodes = np.load(str(path_to_np_array), allow_pickle=True)
        gt_electrode_o3d = o3d.geometry.PointCloud()
        gt_electrode_o3d.points = o3d.utility.Vector3dVector(gt_electrodes)
        return gt_electrode_o3d


if __name__ == "__main__":
    pcViewSet = pcViewSet()
    """
    pcViewSet.addView(0,np.eye(4),np.eye(4))
    pcViewSet.addView(1,2*np.eye(4),2*np.eye(4))
    pcViewSet.addView(2,3*np.eye(4),3*np.eye(4))
    print(pcViewSet.pcViewList)
    pcViewSet.deleteView(2)
    print(pcViewSet.pcViewList)
   
    R1 = tf.random_rotation_matrix()
    R2 = tf.random_rotation_matrix()
    #R1 = np.eye(4)
    #R2 = np.eye(4)
    
    q1 = tf.quaternion_from_matrix(R1)
    q2 = tf.quaternion_from_matrix(R2)
    angle = np.arccos(np.abs(np.dot(q1,q2)))
    angle = np.rad2deg(2*angle)
    print(angle)

    
    rot, trans = pcViewSet.computeErrorMetrics(R1, R2)
    
    print(rot)
    print(trans)

    pose_list = pcViewSet.getGroundtruthCameraPosition("/home/Vishwanath/catkin_ws/src/master_arbeit/src/slam/src/data_matlab/camera_position.npy")
    print(pose_list)
    """

    

