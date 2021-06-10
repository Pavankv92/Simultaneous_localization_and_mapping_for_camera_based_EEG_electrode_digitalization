import rospy
import glob
from sensor_msgs.msg import Image
from natsort import natsorted
import cv2
import numpy as np



def create_single_txt_file_from_multiple_txt_files(folder_path, path_to_save_txt_file):
    all_txt_files = glob.glob(folder_path + "*.txt")

    with open(path_to_save_txt_file,"w") as txt_write :
        for file in natsorted(all_txt_files):
            with open(file, "r+") as txt_read:
                lines = txt_read.read().splitlines()
                img_name = file.split("img")
                img_name = img_name[-1].split(".")
                txt_write.write("{} ".format(str(img_name[0])))
                #txt_write.write(" ")
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

if __name__ =="__main__":

    """
    base_dir = "/home/Vishwanath/data/data/"
    image_dir = base_dir + "imgs/"
    all_image_names = glob.glob(image_dir + "*.jpg")
    
    with open(base_dir + "all_image_names.txt" , 'w') as w:
        for image in natsorted(all_image_names):
            w.write(str(image))
            w.write("\n")
        w.close()
    
    img = cv2.imread("/media/pallando/share/students/Vishwanath/master_arbeit_data/hand_trajectory/static_phantom_head/CA_124_2/data/imgs/img5936.jpg")
    cv2.imshow("image", img)
    cv2.waitKey(1000)
    """
    
    base_dir = "/media/pallando/share/students/Vishwanath/master_arbeit_data/hand_trajectory/static_phantom_head/CA_124/12/"
    folder_path = base_dir + "yolo_prediction/"
    path_to_save_text_file = base_dir + "yolo_labels.txt"
    create_single_txt_file_from_multiple_txt_files(folder_path, path_to_save_text_file)
    
