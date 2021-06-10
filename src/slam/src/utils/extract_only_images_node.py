#!/usr/bin/env python

# once the trajectory is recorded in to the rosbag file. extract rgb and depth images to the imgs & depth_imgs folder
# create a single .txt file with just the rgb image names in it. yolo will read this file and access the rgb and depth images
# yolo(.....path to the image_names.txt, path_to_save_yolo_detected_images, path_to_save_predictions)
# Yolo outputs .txt file for each image with predictions @  path_to_save_predictions
# but to generate point cloud we need to have a single file with image name and yolo predications create_single_txt_file_from.....function creates
# this single file
# 


import rospy
import os
import time
import shutil
from tf_sync import TFSynchronizer
from natsort import natsorted
import glob


def create_single_txt_file_from_multiple_txt_files(folder_path, path_to_save_txt_file):
    all_txt_files = glob.glob(folder_path + "*.txt")

    with open(path_to_save_txt_file,"w") as txt_write :
        for file in natsorted(all_txt_files):
            with open(file, "r+") as txt_read:
                lines = txt_read.read().splitlines()
                #img_name = file.split("/")
                #txt_write.write("{} ".format(str(img_name[-1:])))
                txt_write.write("0.0 ")
                txt_write.write(str(file))
                txt_write.write(" ")
                txt_write.write("1280 ")
                txt_write.write("720 ")
                txt_write.write("0.0 ")
                for i in range(len(lines)):
                    line = lines[i].split()
                    for j in range(len(line)):
                        if j == 0 :
                            continue
                        if j == 1 :
                            continue
                        
                        #if j%3 == 0:
                            #txt_write.write("0 ")
                            #continue
                        txt_write.write(str(round(float(line[j]))))
                        txt_write.write(" ")
                    txt_write.write("0.0 ")
                txt_write.write("\n")
        txt_write.close()


if __name__ == '__main__':

    print('Starting node.')
    raw_input("start the ros bag and hit enter to extract rgb and depth images")

    try:
        rospy.init_node("extract_images_node")
        default_path = '/media/pallando/share/students/Vishwanath/master_arbeit_data/hand_trajectory/static_phantom_head/CA_124/dummy/'
        #default_path = '/home/Vishwanath/master_arbeit_data/hand_trajectory/static_phantom_head/CA_124_2/'
        img_name = rospy.get_param("~img_name", "/k4a/rgb/image_rect_color")
        depth_name = rospy.get_param("~depth_name", "/k4a/depth_to_rgb/image_rect")
        img_fns = rospy.get_param("~img_fns", default_path + "imgs/img__id__.jpg")
        depth_img_fns = rospy.get_param("~depth_img_fns", default_path + "depth_imgs/img__id__.jpg")

        stop_topic = rospy.get_param("~stop_topic", "/shoot_flag")
        tf_name = rospy.get_param("~ee_link", "panda_link7")
        base_link = rospy.get_param("~base_link", "panda_link0")

        if os.path.exists(os.path.dirname(depth_img_fns)):
            print("Removing old depth imgs at {}...".format(os.path.dirname(depth_img_fns)))
            shutil.rmtree(os.path.dirname(depth_img_fns))
        if os.path.exists(os.path.dirname(img_fns)):
            print("Removing old imgs at {}...".format(os.path.dirname(img_fns)))
            shutil.rmtree(os.path.dirname(img_fns))
        print("Creating directory {}".format(os.path.dirname(depth_img_fns)))
        os.mkdir(os.path.dirname(depth_img_fns))
        print("Creating directory {}".format(os.path.dirname(img_fns)))
        os.mkdir(os.path.dirname(img_fns))

        sync = TFSynchronizer(tf_name, base_link, img_name, img_fns, stop_topic, depth_name, depth_img_fns, use_shoot_flag=False)
        print("Starting to record img and TFs.")

        print("Waiting for user input to process and save all")
        raw_input("completed hit enter")

    except rospy.ROSInterruptException:
        pass

    print("writing all images to a text file. yolo reads this file to access rgb and depth images")

    base_dir = default_path 
    image_dir = base_dir + "imgs/"
    all_image_names = glob.glob(image_dir + "*.jpg")

    with open(base_dir + "all_image_names.txt" , 'w') as w:
        for image in natsorted(all_image_names):
            w.write(str(image))
            w.write("\n")
        w.close()
    print("no of images: {}".format(len(all_image_names)))

    """
    print("creating a single file with image names and yolo predictions.")
    
    folder_path = default_path + "yolo_prediction/"
    path_to_save_text_file = default_path + "yolo_labels.txt"
    create_single_txt_file_from_multiple_txt_files(folder_path, path_to_save_text_file)

    print("completed: proceed towards creating the point cloud for slam")
    """