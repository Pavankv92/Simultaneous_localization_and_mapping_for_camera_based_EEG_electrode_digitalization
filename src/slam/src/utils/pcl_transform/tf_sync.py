import tf
import tf2_ros
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
import numpy as np
import imageio
from cv_bridge import CvBridge
import os
from threading import Timer
import message_filters


class TFSynchronizer:

    def __init__(self, tf_name, base_link, img_name, img_fns, stop_topic, depth_name, depth_img_fns, use_shoot_flag=True):
        self.tf_name = tf_name
        self.depth_img_fns = depth_img_fns
        img_sub = message_filters.Subscriber(img_name, Image, queue_size=None, buff_size=65536*1024*20)
        depth_img_sub = message_filters.Subscriber(depth_name, Image, queue_size=None, buff_size=65536*1024*20)
        self.ts = message_filters.ApproximateTimeSynchronizer([img_sub, depth_img_sub], 1000, 0.02)
        self.ts.registerCallback(self.img_callback)
        self.stop_sub = rospy.Subscriber(stop_topic, Bool, self.shoot_callback, queue_size=1)
        self.img_timestamps = []
        self.img_ids = []
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(50*60))
        self.transformer = tf2_ros.TransformListener(self.tf_buffer)
        self.transforms = []
        self.img_fns = img_fns
        self.cv_bridge = CvBridge()
        self.base_link = base_link
        self.use_shoot_flag = use_shoot_flag
        self.save_next_img = not self.use_shoot_flag # if no shoot flag -> set save_next_img true from the beginning
        self.stop_processing = False
        self.timer = Timer(40, self.processing_done)
        self.done = False

    def processing_done(self):
        self.done = True

    def shoot_callback(self, data):
        # Usually would need mutex or synchronize callbacks
        # but shoot happens with enough spacing in time
        self.save_next_img = True and not self.stop_processing

    def img_callback(self, msg_img, msg_depth):
        self.timer.cancel()
        self.timer = Timer(40, self.processing_done)
        self.timer.start()
        if self.save_next_img:
            self.img_timestamps.append(msg_img.header.stamp)
            self.img_ids.append(msg_img.header.seq)
            dtype = np.dtype(np.uint8)
            dtype = dtype.newbyteorder('>' if msg_img.is_bigendian else '<')
            dtype_depth = np.dtype(np.float32)
            dtype_depth = dtype_depth.newbyteorder('>' if msg_depth.is_bigendian else '<')
            shape = (msg_img.height, msg_img.width, 4)
            data = np.fromstring(msg_img.data, dtype=dtype).reshape(shape)
            data.strides = (
                msg_img.step,
                dtype.itemsize * 4,
                dtype.itemsize
            )
            depth_data = np.fromstring(msg_depth.data, dtype=dtype_depth).reshape((msg_depth.height, msg_depth.width, 1))
            depth_data.strides = (
                msg_depth.step,
                dtype_depth.itemsize,
                dtype_depth.itemsize
            )
            print("Processing img seq {}".format(msg_img.header.seq))
            imageio.imwrite(self.img_fns.replace("__id__", str(msg_img.header.seq)), np.flip(data[:, :, 0:3], 2))

            imageio.imwrite(self.depth_img_fns.replace("__id__", str(msg_img.header.seq)) + ".tif", depth_data)
            imageio.imwrite(self.depth_img_fns.replace("__id__", str(msg_img.header.seq)), np.clip(depth_data, 0., 2.))
            if self.use_shoot_flag:
                self.save_next_img = False

    def process_transforms(self):
        self.transforms = []
        ids_to_remove = []
        for idx, tp in enumerate(self.img_timestamps):
            try:
                trans = self.tf_buffer.lookup_transform(self.base_link, self.tf_name, tp)
                transl = trans.transform.translation
                quat = trans.transform.rotation
                rot = tf.transformations.quaternion_matrix((quat.x, quat.y, quat.z, quat.w))
                transform = rot
                transform[:3, 3] = (transl.x, transl.y, transl.z)
                transform[3, :] = [0, 0, 0, 1]
                self.transforms.append(transform)
            except (tf2_ros.LookupException, tf2_ros.ExtrapolationException):
                print("Removing image with id {} because no valid TF was found".format(self.img_ids[idx]))
                ids_to_remove.append(idx)
                os.remove(self.img_fns.replace("__id__", str(self.img_ids[idx])))

        self.img_ids = [val for idx, val in enumerate(self.img_ids) if idx not in ids_to_remove]
        self.img_timestamps = [val for idx, val in enumerate(self.img_timestamps) if idx not in ids_to_remove]

    def save_transformations(self, robot_fn):
        print("Saving {} poses".format(len(self.transforms)))
        with open(robot_fn, 'w') as file:
            for id, tp, transform in zip(self.img_ids, self.img_timestamps, self.transforms):
                strings = [str(id), str(tp)]
                for e in np.asarray(transform).flatten():
                    strings.append(str(e))
                file.write(" ".join(strings) + "\n")


    @staticmethod
    def read_transformations(filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
            arrays = []
            idx = []
            time_stamps = []
            for line in lines:
                flattened_arr = line.strip().split(" ")[2:]
                idx.append(line.split(" ")[0])
                time_stamps.append(line.split(" ")[1])
                shaped_arr = np.asarray(flattened_arr, dtype=np.float).reshape((4, 4))
                arrays.append(shaped_arr)
            return arrays, idx, time_stamps
