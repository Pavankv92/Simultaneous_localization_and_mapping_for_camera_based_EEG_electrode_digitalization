#!/usr/bin/env python

import rospy
import numpy as np
from tf_sync import TFSynchronizer as TFSync
import cv2
import os
import time
import shutil
from pcl_transform.pcl_transformer_hand import PCLTransformer


if __name__ == '__main__':
    print('Starting pcl transformer node.')
    # input()
    try:
        rospy.init_node("pcl_transformer")
        default_folder =  "/media/pallando/share/students/Vishwanath/master_arbeit_data/hand_trajectory/static_phantom_head/CA_124/7_c/"
        cap_fn = default_folder + 'recorded_caps/CA-124_2021-02-19 14:23:54'
        rob_fn = default_folder + 'ground_truth_camera_bad_poses_removed.npy'
        img_fns = default_folder + 'imgs/img__id__.jpg'
        example_imgs_fn = default_folder + 'example_imgs/img__id__.jpg'
        output_fn = default_folder + 'yolo_labels.txt'
        camera_matrix_fn = default_folder + 'calibrations/camera_matrix.txt'
        electrode_positions_fn = default_folder + 'electrode_positions.txt'
        box_size = 0.02
        model_bloat = 0.015
        max_angle = np.pi
        transformer = PCLTransformer(markerFilename=cap_fn,
                                     src_img_size=[1280, 720],
                                     dest_img_size=[1280, 720],
                                     camera_matrix=np.loadtxt(camera_matrix_fn, delimiter=',', dtype=np.float),
                                     box_size=box_size,
                                    )
        print("Reading transformations...")
        poses, idx = transformer.read_robot_poses(rob_fn)

        print("Processing poses...")
        all_boxes = transformer.process_poses(output_fn, electrode_positions_fn, poses, idx, img_fns, model_bloat, max_angle, head_tfs_fn =None)
        print("Saving example images...")
        example_img_dir = os.path.dirname(example_imgs_fn)
        if os.path.exists(example_img_dir):
            shutil.rmtree(example_img_dir)
            time.sleep(5)
        os.mkdir(example_img_dir)

        for i in range(0, len(all_boxes), 10):
            img_ori = cv2.imread(img_fns.replace('__id__', str(idx[i])))
            cv2.imshow("img", img_ori)
            cv2.waitKey(10)
            for box in all_boxes[i]:
                PCLTransformer.plot_one_box(img_ori, box)
            cv2.imwrite(example_imgs_fn.replace('__id__', str(idx[i])), img_ori)

        # print("Showing results...")
        # print("Press 'n' to go forward and 'b' to go backward. Press c to quit")
        # i = 0
        # while 0<=i<len(all_boxes):
        #     img_ori = cv2.imread(img_fns.replace('__id__', str(idx[i])))
        #     for box in all_boxes[i]:
        #         PCLTransformer.plot_one_box(img_ori, box)
        #     #img_ori = cv2.resize(img_ori, (516,516))
        #     cv2.imshow('Detection result', img_ori)
        #     key = cv2.waitKey(0)
        #     if key == ord('c'):
        #         break
        #     elif key == ord('n') and i+1 < len(all_boxes):
        #         i += 1
        #     elif key == ord('b') and i > 0:
        #         i -= 1

    except rospy.ROSInterruptException:
        pass
