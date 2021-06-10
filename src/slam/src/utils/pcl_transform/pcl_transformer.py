import typing
import numpy as np
import csv
#from mayavi import mlab
#from scipy.spatial import Delaunay
#import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Delaunay
import os
import cv2
import random

# List of 3 floats
Vector4d = np.ndarray

PointList = typing.List[Vector4d]

Matrix4d = np.ndarray
PoseList = typing.List[Matrix4d]

# List of triangles
Model = typing.List[PointList]
Box = np.ndarray # xmin ymin xmax ymax


class PCLTransformer:

    def __init__(self, rob_T_atrac: Matrix4d , ee_T_kin: Matrix4d, atrac_T_head: Matrix4d, markerFilename: typing.AnyStr,
                 static_ee_offset: float, hm_T_head: Matrix4d, src_img_size: typing.List[int] = None, dest_img_size: typing.List[int] = None,
                 camera_matrix=None, box_size=0.02):
        self.rob_T_atrac = rob_T_atrac
        self.ee_T_kin = ee_T_kin
        self.panda_link8_tf = np.eye(4)
        self.panda_link8_tf[2,3] = static_ee_offset
        self.atrac_T_head = atrac_T_head # head marker
        self.read_marker_in_head_from_file(markerFilename)
        self.src_img_size = src_img_size
        self.dest_img_size = dest_img_size
        self.camera_matrix = camera_matrix
        self.box_size = box_size
        self.head_tf = hm_T_head


    #kin_T_tf, tf :4x4 matrix (electrode)
    def transform_tf_head_to_kin(self, tf: Matrix4d, robPose: Matrix4d) -> Matrix4d:
        # in:point in head coordinate frame
        rob_T_head = np.matmul(self.rob_T_atrac, self.atrac_T_head)
        robPose_link8 = np.matmul(robPose, self.panda_link8_tf)
        ee_T_head = np.matmul(np.linalg.inv(robPose_link8), rob_T_head)
        kin_T_head = np.matmul(np.linalg.inv(self.ee_T_kin), ee_T_head)
        kin_T_tf = np.matmul(kin_T_head, tf)
        return kin_T_tf
    
    #kin_P_point, point :4x1 vec (electrode)
    def transform_point_head_to_kin(self, point: Vector4d, robPose: Matrix4d) -> Vector4d:
        # in:point in head coordinate frame
        rob_T_head = np.matmul(self.rob_T_atrac, self.atrac_T_head)
        robPose_link8 = np.matmul(robPose, self.panda_link8_tf)
        ee_T_head = np.matmul(np.linalg.inv(robPose_link8), rob_T_head)
        kin_T_head = np.matmul(np.linalg.inv(self.ee_T_kin), ee_T_head)
        kin_P_point = np.dot(kin_T_head, point)
        return kin_P_point

    #transfer all gt electrodes (4x1) to single robot pose hence kin_T_allelectrodes
    def tranform_points_head_to_kin(self, robPose: Matrix4d, points: PointList = None) -> PointList:
        if points is None:
            points = self.cap_points
        return [self.transform_point_head_to_kin(point, robPose) for point in points]

    # delaune is a traiangulation technique to create a mesh for scattered points
    def build_model_from_points(self, points: PointList = None, robotPose: Matrix4d = None) -> Model:
        if points is None:
            points = self.cap_points
        if robotPose is not None:
            points = self.tranform_points_head_to_kin(robotPose, points)

        points_arr = np.asarray(points)
        # Build model
        delaunay = Delaunay(points_arr[:, :3]) # stripping homogeneous part 1
        # hull = ConvexHull(points_arr[:, :3])
        # indices = hull.simplices
        indices = delaunay.simplices
        vertices = points_arr[indices]

        #fig = plt.figure()
        #ax = fig.add_subplot(1, 1, 1, projection='3d')

        #ax.plot_trisurf(points_arr[:,0], points_arr[:,1], points_arr[:,2], triangles=hull.simplices, cmap=plt.cm.Spectral)
        #ax.scatter(points_arr[:,0], points_arr[:,1], points_arr[:,2], marker='o')
        #plt.show()
        return vertices

    def bloat_model(self, model: Model, distance: float, center: Vector4d):
        # adds extra points to the model at the vertex center translated by the vertex normal outwards by
        # distance
        new_points: PointList = []
        for vertex in model:
            normal = np.cross(vertex[1][:3]-vertex[0][:3], vertex[2][:3]-vertex[0][:3])
            vertex_center = 0.5 * (vertex[1][:3]-vertex[0][:3]) + vertex[0][:3]
            vertex_center += 0.5 * (vertex[2][:3] - vertex_center)
            vertex_direction = vertex_center - center[:3]
            if np.dot(normal, vertex_direction) < 0:
                normal *= -1
            normal /= np.linalg.norm(normal)
            new_point = vertex_center + normal * distance
            new_points.append(new_point)
            for p in vertex:
                new_points.append(p[:3])

        return self.build_model_from_points(new_points)

    def remove_invisible_points(self, model: Model, points: PointList, max_angle: float, origin: Vector4d = np.asarray([0, 0, 0, 1])) -> PointList:
        res = []
        for point in points:
            if not self.intersect(origin, point, model) and not self.too_far_away(origin, point, points) and not self.electrode_sharp_angled(origin, point, points, max_angle):
                res.append(point)

        return res

    @staticmethod
    def point_in_vertex(point: Vector4d, vertex: PointList) -> bool:
        for v in vertex:
            same = True
            for i, p in enumerate(point):
                if p != v[i]:
                    same = False
                    break
            if same:
                return True
        return False

    def electrode_sharp_angled(self, origin: Vector4d, point: Vector4d, points: PointList, max_angle: float) -> bool:
        center_point = np.copy(points[0])
        for point_l in points[1:]:
            center_point += point_l
        center_point /= len(points)
        direction = point - origin
        point_dir = center_point - point

        angle = self.angle_between(direction[0:3], point_dir[0:3])

        return angle > max_angle
        # return np.abs(avg_angle - np.pi / 2) < np.pi / 32

    @staticmethod
    def unit_vector(vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
        """ Returns the angle in radians between vectors 'v1' and 'v2'::

                >>> self.angle_between((1, 0, 0), (0, 1, 0))
                1.5707963267948966
                >>> self.angle_between((1, 0, 0), (1, 0, 0))
                0.0
                >>> self.angle_between((1, 0, 0), (-1, 0, 0))
                3.141592653589793
        """
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def too_far_away(self, origin: Vector4d, point: Vector4d, points: PointList, mean_fraction: float = 0.95) -> bool:
        mean = 0
        max_p = -float('Inf')
        min_p = float('Inf')
        for p in points:
            dist = np.linalg.norm(p[:3])
            mean += dist
            max_p = max(max_p, dist)
            min_p = min(min_p, dist)
        mean /= len(points)

        if np.linalg.norm(point[:3] - origin[:3]) > mean + (mean_fraction-1) * (max_p - min_p):
            return True
        return False

    def intersect(self, origin: Vector4d, point: Vector4d, model: Model) -> bool:
        for vertex in model:
            epsilon = 0.00001
            e1 = vertex[1]-vertex[0]
            e2 = vertex[2]-vertex[0]
            e1 = e1[0:3]
            e2 = e2[0:3]
            direction = point-origin
            direction = direction[0:3]
            direction /= np.linalg.norm(direction)
            q = np.cross(direction,e2)
            a = np.dot(e1,q)
            if -epsilon < a < epsilon:
                # the vector is parallel to the plane (the intersection is at infinity)
                continue

            f = 1/a
            s = origin[0:3]-vertex[0,0:3]
            u = f*np.dot(s,q)

            if u <= 0.0:
                # the intersection is outside of the triangle
                continue

            r = np.cross(s, e1)
            v = f*np.dot(direction, r)

            if v <= 0.0 or u+v >= 1.0:
                # the intersection is outside of the triangle
                continue

            p = vertex[0,0:3] + u*e1 + v*e2

            # intersects directly at point -> probably the electrode position
            #if np.linalg.norm(p - point[0:3]) < epsilon:
            #    continue
            t = f*np.dot(e2,r) # verified!
            if t+epsilon < np.linalg.norm(point-origin):
                return True

        return False

    def read_marker_in_head_from_file(self, filename: typing.AnyStr):
        self.cap_points = []
        self.cap_names = []
        with open(filename) as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            for row in spamreader:
                arr = np.asarray(row[1:]).astype(np.float)
                arr = np.concatenate((arr, [1]))
                name = row[0]
                self.cap_names.append(name)
                self.cap_points.append(arr)

    def point_to_pixels(self, point: Vector4d):
        pixel_coordinates = self.camera_matrix.dot(point[:3])
        pixel_coordinates[0] *= self.dest_img_size[0] / self.src_img_size[0] / pixel_coordinates[2]
        pixel_coordinates[1] *= self.dest_img_size[1] / self.src_img_size[1] / pixel_coordinates[2]
        #tmp = pixel_coordinates[0]
        #pixel_coordinates[0] = pixel_coordinates[1]
        #pixel_coordinates[1] = tmp
        return pixel_coordinates[:2]

    def box_from_point(self, point: Vector4d) -> Box:
        box = []
        tmp = point.copy()
        tmp[:2] -= self.box_size / 2
        box.append(self.point_to_pixels(tmp))
        tmp = point.copy()
        tmp[:2] += self.box_size / 2
        box.append(self.point_to_pixels(tmp))
        return np.asarray(box).flatten()

    def remove_outside_boxes(self, boxes: typing.List[Box]) -> typing.List[Box]:
        result = []
        for box in boxes:
            if box[0] >= 0 <= box[1] and box[2] < self.dest_img_size[0] and box[3] < self.dest_img_size[1]:
                result.append(box)
        return result

    def points_center(self, points: PointList) -> Vector4d:
        mean_p: Vector4d = np.asarray([0., 0., 0., 0.])
        for p in points:
            mean_p += p
        mean_p /= len(points)
        mean_p[3] = 1.
        return mean_p

    def process_poses(self, filename: typing.AnyStr, electrode_pos_fn: typing.AnyStr, poses: typing.List[np.ndarray], idx, img_fns: typing.AnyStr = None, model_bloat: float = 0., max_angle: float = np.pi, head_tfs_fn: str = None):
        if os.path.isfile(filename) and img_fns is not None:
            os.remove(filename)
        if os.path.isfile(electrode_pos_fn):
            os.remove(electrode_pos_fn)
        if head_tfs_fn is not None and os.path.isfile(head_tfs_fn):
            os.remove(head_tfs_fn)

        all_vis_boxes = []
        for i, pose in enumerate(poses):
            transformed_head_tf = self.transform_tf_head_to_kin(tf=self.head_tf, robPose=pose)
            if head_tfs_fn is not None:
                self.save_head_tf(head_tfs_fn, transformed_head_tf, img_fns.replace("__id__", str(idx[i])), idx[i])
            cap_points = self.tranform_points_head_to_kin(pose)
            model = self.build_model_from_points(cap_points)
            if model_bloat > 0:
                model = self.bloat_model(model, model_bloat, self.points_center(cap_points))
            visible_points = self.remove_invisible_points(model, cap_points, max_angle)
            if electrode_pos_fn is not None:
                self.save_points_to_file(electrode_pos_fn, visible_points, img_fns.replace("__id__", str(idx[i])), idx[i])
            visible_boxes = [self.box_from_point(p) for p in visible_points]
            visible_boxes = self.remove_outside_boxes(visible_boxes)
            all_vis_boxes.append(visible_boxes)
            if img_fns is not None:
                self.save_boxes_to_file(filename, visible_boxes, img_fns.replace("__id__", str(idx[i])), idx[i])

        return all_vis_boxes

    def save_points_to_file(self, filename: str, point_list: PointList, img_filename: typing.AnyStr, img_index: int):
        with open(filename, 'a') as file:
            strings = [str(img_index), img_filename]
            prefix = " ".join(strings)
            suff_strings = []
            for p in point_list:
                for element in p:
                    suff_strings.append(str(element))
            file.write(prefix + " " + " ".join(suff_strings) + "\n")

    def save_head_tf(self, filename: str, tf: Matrix4d, img_filename: str, img_index: int):
        with open(filename, 'a') as file:
            strings = [str(img_index), img_filename]
            for e in np.asarray(tf).flatten():
                strings.append(str(e))
            file.write(" ".join(strings) + "\n")

    def save_boxes_to_file(self, filename: typing.AnyStr, box_list: typing.List[Box], img_filename: typing.AnyStr, img_index: int):
        with open(filename, 'a') as file:
            strings = [str(img_index), img_filename, str(self.dest_img_size[0]), str(self.dest_img_size[1])]
            prefix = " ".join(strings)
            suff_strings = []
            for box in box_list:
                suff_strings.append(str(0))
                for i in range(4):
                    suff_strings.append(str(int(np.round(box[i]))).replace('[', '').replace(']', ''))

            file.write(prefix + " " + " ".join(suff_strings) + "\n")

    @staticmethod
    def _tfInv(pose: Matrix4d):
        #pose = np.asarray(pose)
        rot = pose[0:3, 0:3].T
        trans = -np.dot(rot, pose[0:3, 3])
        transform = np.empty((4, 4))
        transform[:3, :3] = rot
        transform[:3, 3] = trans
        transform[3, :] = [0, 0, 0, 1]
        return transform

    @staticmethod
    def plot_one_box(img, coord, label=None, color=None, line_thickness=None):
        '''
        coord: [x_min, y_min, x_max, y_max] format coordinates.
        img: img to plot on.
        label: str. The label name.
        color: int. color index.
        line_thickness: int. rectangle line thickness.
        '''
        tl = line_thickness or int(round(0.002 * max(img.shape[0:2])))  # line thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(coord[0]), int(coord[1])), (int(coord[2]), int(coord[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl)
        print(coord)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(label, 0, fontScale=float(tl) / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, float(tl) / 3, [0, 0, 0], thickness=tf, lineType=cv2.LINE_AA)


