#!/usr/bin/env python
# electrode mapping before using hand held trajectory with static phanton head
# head band is querried only once as phantom doesn't move
# head band can be eliminated while recording the kinect hand trajectoy also. 
# both electrode position and kinect hand trajectory will be mapped in tracking camera frame 
# everything will be calculated wrt to first camera frame. 

import tf.transformations as tf
import numpy as np
import scipy.linalg as las
from atracsys_client import attracsysClient
import datetime , time
import csv
import rospy
import collections
from itertools import chain
from nav_msgs.msg import Path
from conversions import np2ros_poseStamped, ros2np_Pose
from geometry_msgs.msg import PoseStamped


class electrodeMapping(object):

    def __init__(self):
        self.stylus = attracsysClient('134.28.45.17',5001,'bluestylustip','FORMAT_MATRIXROWWISE')
        self.headmarker = attracsysClient('134.28.45.17',5001,'headband','FORMAT_MATRIXROWWISE')
        self.base_dir = '/media/pallando/share/students/Vishwanath/master_arbeit_data/hand_trajectory/static_phantom_head/CA_124/1/'
        self.Path = Path()
    def record_electrodes(self, eeg_cap_type ):
        print('Lets start the electrode mapping\n')
        electrode_labels = self.read_labels_file_from_csv(eeg_cap_type)
        print("Electrode labels: ", electrode_labels)
        electrode_labels_dict = collections.OrderedDict.fromkeys(electrode_labels , np.zeros(3,))
        print("Number of electrode labels: ", len(electrode_labels_dict))

        T_trk_ele_list = []
        response = ''
        while not response == 'q':
            for i in range(len(electrode_labels_dict)):
                electrode_name = electrode_labels_dict.keys()[i]
                response = raw_input('Place the stylus at '+ '-' + electrode_name + '-' + ' and hit enter. Press q to quit')
                if response == 'q':
                    return 0
    
                T_trk_ele = self.stylus.record_marker()
                T_trk_ele_list.append(T_trk_ele)
                t_trk_ele = T_trk_ele[0:3,3]
                
                electrode_labels_dict[electrode_name] = t_trk_ele

            #print("recording the headband now")
            #T_trk_hm = self.headmarker.record_marker()
            #print("T_trk_hm: ", T_trk_hm)
            response = raw_input('All electrodes were measured. press q to continue')
            if response == 'q':
                break

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(T_trk_ele_list)
        self.save_to_csv(electrode_labels_dict, eeg_cap_type, timestamp)
        np.save(self.base_dir + "recorded_caps/T_trk_ele_list_{}.npy".format(timestamp),T_trk_ele_list)
        #np.save(self.base_dir + "recorded_caps/T_trk_hm_{}.npy".format(timestamp),T_trk_hm)


    
    def save_to_csv(self, electrode_labels_dict, eeg_cap_type, timestamp):
        filename = eeg_cap_type + "_" + timestamp
        file_dir = self.base_dir + "recorded_caps/" + filename
        with open(file_dir, 'w') as f:
            for key in electrode_labels_dict.keys():
                f.write("%s,%s,%s,%s\n"%(key,electrode_labels_dict[key][0],electrode_labels_dict[key][1],electrode_labels_dict[key][2]))

    def read_labels_file_from_csv(self, eeg_cap_type):
        labels = []
        file_dir = "/home/Vishwanath/catkin_ws/src/master_arbeit/src/slam/src/data/hand_trajectory/common_data/electrode_labels/" + eeg_cap_type + ".csv"
        with open(file_dir) as csvfile:
            csv_reader = csv.reader(csvfile, delimiter=',')
            for row in csv_reader:
                labels.append(row)
        return list(chain.from_iterable(labels)) # removes inner list
    
    def track_marker_continuously(self, locator_name, path_to_save):
        pose_list = []
        if str(locator_name) == "bluestylustip" :
            locator = self.stylus
        while True:
            matrix = locator.record_marker_ignore_visibility()
            pose_list.append(matrix)
        np.save(str(path_to_save), pose_list)
            
    

if __name__ == "__main__":
    em = electrodeMapping()
    # read rosparams
    rospy.init_node('electrode_mapping')
    pub_poseStamped = rospy.Publisher('camera_pose_kinect', Path, queue_size=10)
    eeg_cap_type = "CA-124"
    em.record_electrodes(eeg_cap_type)
    #path_to_save = "/home/Vishwanath/catkin_ws/src/master_arbeit/src/slam/src/data/test/pose_list.npy"
    #em.track_marker_continuously("bluestylustip", path_to_save)
    


