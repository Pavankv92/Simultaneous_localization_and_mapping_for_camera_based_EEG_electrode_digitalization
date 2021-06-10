import open3d as o3d
import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
import copy
import tf.transformations as tf
import matplotlib.pyplot as plt
import numpy.linalg as la
from utils.open3d_helper import convertCloudFromOpen3dToRos
from utils.pcViewSet import *

base_dir = "/home/Vishwanath/catkin_ws/src/master_arbeit/src/slam/src/data/robot_trajectory/CA_124_7/"
#file_1 = np.load("/home/Vishwanath/catkin_ws/src/master_arbeit/src/slam/src/data/hand_trajectory/static_phantom_head/CS_301_7/electrode_position/electrode_1.npy", allow_pickle=True)
#file_2 = np.load("/home/Vishwanath/catkin_ws/src/master_arbeit/src/slam/src/data/hand_trajectory/static_phantom_head/CS_301_7/electrode_position/electrode_50.npy", allow_pickle=True)


file_1 = np.load(base_dir + "results/k_means_cluster_centers_estimated.npy")
pcd_1 = o3d.geometry.PointCloud()
pcd_1.points = o3d.utility.Vector3dVector(file_1)

if __name__ == "__main__":
    vSet = pcViewSet()
    pub_electrodes = rospy.Publisher('map_pub_est', PointCloud2, queue_size=10)
    source = pcd_1
    
    o3d.visualization.draw_geometries([pcd_1])
    #pub_electrodes.publish(convertCloudFromOpen3dToRos(pcd_1))
    rospy.spin()



    


    