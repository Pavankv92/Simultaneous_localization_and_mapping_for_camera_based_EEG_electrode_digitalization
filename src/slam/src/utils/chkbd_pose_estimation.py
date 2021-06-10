#!/usr/bin/env python
"""Class provides methods to estimate the checkerboard pose on the fly using opencv
***no need to undistort the images*** 
Raises:
    ValueError: when the 2D image points weren't found on the checkerboard
                *** it is important to have the emough illumination on the checkerboard***
Returns:
    array -- checkerboard pose in RTSTampedWuthHeader format, check the msg type for more info.
"""

import numpy as np
import cv2
from numpy import linalg as la


class poseEstimation(object):
    def __init__(self): 

        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-16)
        np.set_printoptions(precision=16)
        
        #"""
        # my calibration value : for larger data set
        self.cam_mtx= np.array([[601.06,   0.        , 636.54],
                                [0.        , 601.33, 360.29],
                                [0, 0, 1]], np.float32)
        self.cam_dist= np.array([-0.00745334,  0.0103613 ,  0.00081139, -0.00163488, -0.00378353], np.float32)
        
        #"""
        
        """
        # matlab value : for larger data set
        self.cam_mtx= np.array([[604.5848,   0.        , 639.250],
                                [0.        , 604.4832, 358.5420],
                                [0, 0, 1]], np.float32)
        self.cam_dist= np.array([ -0.00911066,  0.01196246,  0.00057368, -0.00148552, -0.00455759], np.float32)
        
        """
        """
        # martin's value
        self.cam_mtx= np.array([[607.4978637695312, 0.0, 638.925537109375],
                                [0.0, 607.4415893554688, 364.3547058105469],
                                [0, 0, 1]], np.float32)

        self.cam_dist= np.array([ 0.00361788, -0.01384572, -0.00146016,  0.00374483,  0.01151012], np.float32)
        """

        self.objp = self.createObjectpoints(5, 8, 0.04)

    def createObjectpoints(self, height, width, size):
        objp=np.zeros((height*width, 3), np.float32)
        objp[:, :2]=np.indices((width, height)).T.reshape(-1, 2)
        objp *= size
        return np.around(objp, 3)

    def cv2_pose(self, rvecs, tvecs):
        rot3X3=cv2.Rodrigues(rvecs)[0]
        transformation = np.array([[rot3X3[0, 0], rot3X3[0, 1], rot3X3[0, 2], tvecs[0]],
                                    [rot3X3[1, 0], rot3X3[1, 1],rot3X3[1, 2], tvecs[1]],
                                    [rot3X3[2, 0], rot3X3[2, 1],rot3X3[2, 2], tvecs[2]],
                                    [0, 0, 0, 1]], np.float32)
        return transformation
   
    def transform_from_image(self, img):
        
        cv_image=cv2.imread(img)
        #cv2.imshow('image', cv_image)
        #cv2.waitKey(250)
        gray_image=cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        found, corners=cv2.findChessboardCorners(gray_image, (8, 5), None)

        if found:
            cv2.cornerSubPix(gray_image, corners, (11, 11),(-1, -1), self.criteria)
            # debug
            img = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
            cv2.drawChessboardCorners(img, (8,5), corners, found)
            cv2.imshow('image', img)
            cv2.waitKey(1)

            # solvePnP is used to get the position and orientation of the object
            _, rvecs, tvecs=cv2.solvePnP(self.objp, corners, self.cam_mtx, self.cam_dist)

            return self.cv2_pose(rvecs, tvecs), corners, rvecs, tvecs

        else:
            print("pattern not found")
            raise ValueError("Could not find the pattern in the image")

        return np.eye(4), corners


if __name__ == "__main__":
   pose=poseEstimation()