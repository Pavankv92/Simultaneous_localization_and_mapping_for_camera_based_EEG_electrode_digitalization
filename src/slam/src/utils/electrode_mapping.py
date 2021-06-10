#!/usr/bin/env python

import tf.transformations as tf
import numpy as np
import scipy.linalg as las
from atracsys_client import attracsysClient
import datetime , time
import csv
import rospy
import collections
from itertools import chain


class electrodeMapping(object):

    def __init__(self):
        self.stylus = attracsysClient('134.28.45.17',5001,'bluestylustip','FORMAT_MATRIXROWWISE')
        self.headmarker = attracsysClient('134.28.45.17',5001,'headband','FORMAT_MATRIXROWWISE')
        self.base_dir = '/media/pallando/share/students/Vishwanath/master_arbeit_data/hand_trajectory/dynamic_human_head/CS_301_RED/14/'


    def record_head_coordinate(self):
        T_trk_face_point = self.stylus.record_marker_ignore_visibility()
        T_trk_hm = self.headmarker.record_marker_ignore_visibility()
        return T_trk_face_point, T_trk_hm

    def record(self, electrode_name):
        response = raw_input('Place the stylus at '+ '-' + electrode_name + '-' + ' and hit enter. Press q to quit')
        if response == 'q':
            return 0

        T_trk_ele = self.stylus.record_marker_ignore_visibility()
        T_trk_hm = self.headmarker.record_marker_ignore_visibility()
        
        return T_trk_ele, T_trk_hm

    def record_electrodes(self, eeg_cap_type ):
        print('Lets start the electrode mapping\n')
        electrode_labels = self.read_labels_file_from_csv(eeg_cap_type)
        print("Electrode labels: ", electrode_labels)
        electrode_labels_dict = collections.OrderedDict.fromkeys(electrode_labels , np.zeros(3,))
        print("Number of electrode labels: ", len(electrode_labels_dict))

        response = ''
        T_hm_ele_list = []

        while not response == 'q':
            for i in range(len(electrode_labels_dict)):
                electrode_name = electrode_labels_dict.keys()[i]
                """
                response = raw_input('Place the stylus at '+ '-' + electrode_name + '-' + ' and hit enter. Press q to quit')
                if response == 'q':
                    return 0
                T_trk_ele = self.stylus.record_marker()
                T_trk_hm = self.headmarker.record_marker()
                T_hm_ele = np.matmul(np.linalg.inv(T_trk_hm),T_trk_ele)
                t_hm_ele = T_hm_ele[0:3,3]
                """
                recorded = False
                while not recorded: 
                    T_trk_ele, T_trk_hm = self.record(electrode_name)
                    if np.array_equal(np.eye(4),T_trk_ele) or np.array_equal(np.eye(4),T_trk_hm) :
                        #print("did not capture either one of the marker, feel free to move the head")
                        recorded = False 
                    else: 
                        recorded = True 
              
                T_hm_ele = np.matmul(np.linalg.inv(T_trk_hm),T_trk_ele)
                T_hm_ele_list.append(T_hm_ele)
                t_hm_ele = T_hm_ele[0:3,3]
                
                electrode_labels_dict[electrode_name] = t_hm_ele

            response = raw_input('All electrodes were measured. Press q to continue')
            if response == 'q':
                break

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.save_to_csv(electrode_labels_dict, eeg_cap_type, timestamp)
        np.save(self.base_dir + "recorded_caps/T_hm_ele_list_{}.npy".format(timestamp),T_hm_ele_list)

    
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

if __name__ == "__main__":
    em = electrodeMapping()

    # read rosparams
    rospy.init_node('electrode_mapping')
    #eeg_cap_type = rospy.get_param("/eeg_cap_type")
    eeg_cap_type = "CS-301-red"
    em.record_electrodes(eeg_cap_type)
    

