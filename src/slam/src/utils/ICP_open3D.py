import open3d as o3d
import numpy as np
import copy
import tf.transformations as tf
import matplotlib.pyplot as plt
import numpy.linalg as la

base_dir = "/home/Vishwanath/catkin_ws/src/master_arbeit/src/slam/src/data/hand_trajectory/static_phantom_head/beta_10/CA_124_6/"
#file_1 = np.load("/home/Vishwanath/catkin_ws/src/master_arbeit/src/slam/src/data/hand_trajectory/static_phantom_head/CS_301_7/electrode_position/electrode_1.npy", allow_pickle=True)
#file_2 = np.load("/home/Vishwanath/catkin_ws/src/master_arbeit/src/slam/src/data/hand_trajectory/static_phantom_head/CS_301_7/electrode_position/electrode_50.npy", allow_pickle=True)


file_1 = np.load(base_dir + 'results/k_means_cluster_centers_estimated.npy')
file_2 = np.load(base_dir + "results/ground_truth_electrodes_in_first_camera_frame_sync.npy", allow_pickle = True)
print(len(file_2))
pcd_1 = o3d.geometry.PointCloud()
pcd_1.points = o3d.utility.Vector3dVector(file_1)
pcd_2 = o3d.geometry.PointCloud()
pcd_2.points = o3d.utility.Vector3dVector(file_2)
"""
#print(np.asarray(pcd_1.points))
#pcd_2.cluster_dbscan(eps=1e7, min_points = 10, print_progress =True)
"""
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0, 0])
    target_temp.paint_uniform_color([0, 0, 1])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])
    #o3d.geometry.LineSet.create_from_point_cloud_correspondences()

def draw_registration_result_corres(source, target, transformation, correspondence_set):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0, 0])
    target_temp.paint_uniform_color([0, 0, 1])
    source_temp.transform(transformation)
    line_set = o3d.geometry.LineSet.create_from_point_cloud_correspondences(source_temp, target_temp, correspondence_set)
    o3d.visualization.draw_geometries([source_temp, target_temp, line_set])



if __name__ == "__main__":
    
    
    source = pcd_1
    target = pcd_2
    threshold = 0.060
    trans_init = np.eye(4)
    transform_init = np.array([[9.99573796e-01, -2.66190262e-02,  1.19846035e-02, -3.75437710e-03], 
                                 [2.72267145e-02,  9.98181745e-01, -5.37761413e-02,  2.06973677e-02],
                                 [-1.05313473e-02,  5.40795160e-02,  9.98481110e-01, -3.82126097e-04],
                                 [0.0,        0.0,         0.0,       1.0]],dtype=np.float32)
    reg_p2p = o3d.registration.registration_icp(source, target, threshold, trans_init, o3d.registration.TransformationEstimationPointToPoint())
    print(reg_p2p.correspondence_set)
    print("Transformation is:")
    print(reg_p2p.transformation)
    print("")
    evaluation = o3d.registration.evaluate_registration(source, target, threshold, reg_p2p.transformation)
    draw_registration_result_corres(source, target, reg_p2p.transformation, reg_p2p.correspondence_set)
    print(evaluation)
    print("RMSE and fitness :" , evaluation.inlier_rmse, evaluation.fitness)
    result = [] 
    result.append(evaluation.inlier_rmse)

    #gt_transformed = o3d.geometry.PointCloud()
    #gt_transformed = copy.deepcopy(target).transform(la.inv(reg_p2p.transformation))
    
    #path_to_save = base_dir + 'results/ground_truth_transformed_vis.npy'
    #np.save(str(path_to_save),np.asarray(gt_transformed.points))

    o3d.visualization.draw_geometries([pcd_1, pcd_2])



    


    