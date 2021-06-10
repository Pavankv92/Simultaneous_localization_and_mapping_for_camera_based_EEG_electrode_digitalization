from chkbd_pose_estimation import poseEstimation
from natsort import natsorted
from PIL import Image, ImageDraw
import glob
import cv2 as cv
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy.linalg as la


def get_obj_pts_camera_frame(obj_pts, chkbd_pose):
    obj_pts_camera_frame = []
    for i in range(len(obj_pts)):
        obj = obj_pts[i]
        obj_pts_camera_frame.append(np.dot(chkbd_pose, [obj[0], obj[1], obj[2], 1.0]))
    return obj_pts_camera_frame

def calibrate_depth(color_image_dir, pe):
    
    cam_mtx = pe.cam_mtx
    all_color_images = natsorted(glob.glob(color_image_dir))
    c = 0
    norm_list = []
    for color_img_name in all_color_images:
        #cv_image = cv.imread(str(color_img_name))
        #cv.imshow("image", cv_image)
        #cv.waitKey(1000)
        norm_per_image_list = []
        try:
            chkbd_pose, img_pts, rvecs, tvecs = pe.transform_from_image(str(color_img_name))
            
            #"""
            img_pts = img_pts.reshape((40,2))
            if not np.array_equal(np.eye(4), chkbd_pose):
                #fig = plt.figure(figsize=[100,50])
                #ax = fig.add_subplot(111, projection="3d")
                depth_image_name = color_img_name + ".tif"
                depth_image_name = depth_image_name.replace("imgs", "depth_imgs")
                depth_image = Image.open(depth_image_name)
                #print(depth_image.format, depth_image.size, depth_image.mode)
                obj_pts_gt_camera_frame_list = get_obj_pts_camera_frame(pe.objp, chkbd_pose)
                obj_pts_est_camera_frame_list = []
                for idx in range(len(img_pts)):
                    img_pt = img_pts[idx]
                    obj_pt = obj_pts_gt_camera_frame_list[idx]
                    depth = depth_image.getpixel((float(img_pt[0]), float(img_pt[1])))
                    
                    X_c = (float(img_pt[0]) - cam_mtx[0,2]) / cam_mtx[0,0] * depth
                    Y_c = (float(img_pt[1]) - cam_mtx[1,2]) / cam_mtx[1,1] * depth
                    Z_c = depth
                    obj_pt_est = np.asarray([X_c, Y_c, Z_c])
                    obj_pts_est_camera_frame_list.append(obj_pt_est)
                    #print("est:", obj_pt_est)
                    #print("gt:", obj_pt[0:3])
                    #print(la.norm(obj_pt_est-obj_pt[0:3]))
                    norm_per_image_list.append(la.norm(obj_pt_est-obj_pt[0:3]))
                    #ax.scatter(obj_pt[0], obj_pt[1], obj_pt[2] , marker='o')
                    #ax.scatter(X_c, Y_c, Z_c, marker='1')
                
                #ax.scatter(obj_pt[0], obj_pt[1], obj_pt[2] , marker='o',label="Ground Truth")
                #ax.scatter(X_c, Y_c, Z_c, marker='1',label="Estimated")
                #ax.legend()
                mean_value_per_image = np.round(np.mean(norm_per_image_list, axis=0),4)
                #plt.title("mean of L2 norm per image: {} (mm)".format(mean_value_per_image*1000))
                #plt.show()
            #"""
        except ValueError as v:
            print(v)
            print(color_img_name)
            continue    
        
        norm_list.append(mean_value_per_image)
        #break
        
        """
        #cv_image = cv.imread(str(color_img_name))
        img_pts, _ =  cv.projectPoints(pe.objp, rvecs, tvecs, pe.cam_mtx, pe.cam_dist)
        cv.drawChessboardCorners(cv_image, (8,5), img_pts, True)
        cv.imshow("img", cv_image)
        cv.waitKey(10000)
        c = c + 1
        if c == 5:
            break
        """
    mean_value = np.round(np.mean(norm_list, axis=0),4)  
    print("mean of L2 norm of all images: {} (mm)".format(mean_value*1000)) 
        
        

if __name__ == "__main__":
    pe = poseEstimation()
    base_dir = "/media/pallando/share/students/Vishwanath/master_arbeit_data/hand_trajectory/static_phantom_head/depth_analysis/depth_6/"
    color_image_dir = base_dir + "imgs/*.jpg"
    calibrate_depth(color_image_dir,pe)
    