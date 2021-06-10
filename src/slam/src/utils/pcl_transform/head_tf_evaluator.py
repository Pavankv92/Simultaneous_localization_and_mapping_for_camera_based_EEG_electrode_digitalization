
import rospy
from typing import Dict, List
import numpy as np
from geometry_msgs.msg import PoseStamped
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import tf
import csv
from threading import Timer

Matrix4d = np.ndarray
PoseList = List[Matrix4d]


class HeadTFEvaluator:

    def __init__(self, img_filenames: str, depth_fns: str, head_tfs: Dict[int, Matrix4d], head_tf_topic: str, img_topic: str, depth_topic: str):
        self.img_fns = img_filenames
        self.true_head_tfs = head_tfs
        self.head_tf_topic = head_tf_topic
        self.depth_img_fns = depth_fns
        self.head_tf_sub = rospy.Subscriber(head_tf_topic, PoseStamped, self.pose_res_cb, queue_size=len(head_tfs))
        self.img_publisher = rospy.Publisher(img_topic, Image, queue_size=len(head_tfs))
        self.depth_publisher = rospy.Publisher(depth_topic, Image, queue_size=len(head_tfs))
        self.estimated_tfs: Dict[int, Matrix4d] = {}
        self.bridge = CvBridge()
        self.timer = Timer(50, self.processing_done)
        self.timer.start()
        self.done = False

    @staticmethod
    def load_poses(filename: str) -> Dict[int, Matrix4d]:
        poses: Dict[int, Matrix4d] = {}
        with open(filename, 'r') as f:
            for line in f.readlines():
                splits = line.strip('\n').split(" ")
                if len(splits) >= 17:
                    pose = []
                    for e in splits[-16:]:
                        pose.append(float(e))
                    poses[int(splits[0])] = np.asarray(pose).reshape((4,4))

        return poses

    def run(self) -> None:

        rate = rospy.Rate(5)
        for k, v in self.true_head_tfs.items():
            fn = self.img_fns.replace("__id__", str(k))
            img = cv2.imread(fn, cv2.IMREAD_UNCHANGED)
            depth_data = cv2.imread(self.depth_img_fns.replace("__id__", str(k)) + ".tif", cv2.IMREAD_UNCHANGED)
            stamp = rospy.Time.now()
            msg_rgb = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            msg_rgb.header.frame_id = str(k)
            msg_rgb.header.stamp = stamp
            #msg_rgb.data = np.array(img).tostring()

            msg_depth = self.bridge.cv2_to_imgmsg(depth_data)
            msg_depth.header.frame_id = str(k)
            msg_depth.header.stamp = stamp
            #msg_depth.data = np.array(depth_data).tostring()

            self.img_publisher.publish(msg_rgb)
            self.depth_publisher.publish(msg_depth)

            rate.sleep()
            # input('')

        while not self.done:
            rate.sleep()

    def pose_res_cb(self, msg: PoseStamped) -> None:
        self.timer.cancel()
        self.timer = Timer(50, self.processing_done)
        self.timer.start()
        pose = tf.transformations.quaternion_matrix(np.array([msg.pose.orientation.x, msg.pose.orientation.y, msg.pose.orientation.z, msg.pose.orientation.w]))
        pose[0,3] = msg.pose.position.x
        pose[1,3] = msg.pose.position.y
        pose[2,3] = msg.pose.position.z
        self.estimated_tfs[int(msg.header.frame_id)] = pose
        print(msg.header.frame_id)
        print(pose)
        print(self.true_head_tfs[int(msg.header.frame_id)])
        print('\n\n')

    def processing_done(self):
        self.done = True

    def save(self, filename):
        rows = []
        for k, v in self.estimated_tfs.items():
            trans_err = v[0:3, 3] - np.asarray(self.true_head_tfs[k])[0:3, 3]
            t_err_norm = np.linalg.norm(trans_err, 2)
            E = np.matmul(np.asarray(self.true_head_tfs[k])[0:3,0:3], np.transpose(v[0:3,0:3]))
            d: np.ndarray = [E[1, 2] - E[2, 1], E[2, 0] - E[0, 2], E[0, 1] - E[1, 0]]

            dmag = np.sqrt(d[0]*d[0] + d[1]*d[1] + d[2]*d[2])
            phi = np.arcsin(dmag/2)
            m = str(self.true_head_tfs[k].flatten()).replace('[', '').replace(']','').replace('\n','')
            rows.append([k, t_err_norm, phi, ' '.join(m.split())])

        with open(filename, 'w+', newline='') as csvfile:
            for row in rows:
                spamwriter = csv.writer(csvfile, delimiter=',',
                                        quotechar='|', quoting=csv.QUOTE_MINIMAL)
                spamwriter.writerow(row)


