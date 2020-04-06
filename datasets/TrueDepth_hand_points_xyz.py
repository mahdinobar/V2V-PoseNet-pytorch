import os
import numpy as np
import sys
import struct
from torch.utils.data import Dataset
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt
import open3d as o3d
from PIL import Image


def pixel2world(x, y, z, img_width, img_height, fx, fy, cx, cy):
    w_x = (x - cx) * z / fx
    # w_y = (cy - y) * z / fy
    w_y = (y - cy) * z / fy
    w_z = z
    return w_x, w_y, w_z


def world2pixel(x, y, z, img_width, img_height, fx, fy, cx, cy):
    p_x = x * fx / z + cx
    # p_y = cy - y * fy / z
    p_y = y * fy / z - cy
    return p_x, p_y


def depthmap2points(image, fx, fy, cx, cy):
    h, w = image.shape
    x, y = np.meshgrid(np.arange(w) + 1, np.arange(h) + 1)
    points = np.zeros((h, w, 3), dtype=np.float32)
    points[:, :, 0], points[:, :, 1], points[:, :, 2] = pixel2world(x, y, image, w, h, fx, fy, cx, cy)
    return points


def points2pixels(points, img_width, img_height, fx, fy, cx, cy):
    pixels = np.zeros((points.shape[0], 2))
    pixels[:, 0], pixels[:, 1] = \
        world2pixel(points[:, 0], points[:, 1], points[:, 2], img_width, img_height, fx, fy, cx, cy)
    return pixels


def load_depthmap(filename, img_width, img_height, max_depth):
    # with open(filename, mode='rb') as f:
    # data = f.read()
    # _, _, left, top, right, bottom = struct.unpack('I'*6, data[:6*4])
    # num_pixel = (right - left) * (bottom - top)
    # cropped_image = struct.unpack('f'*num_pixel, data[6*4:])
    #
    # cropped_image = np.asarray(cropped_image).reshape(bottom-top, -1)
    # depth_image = np.zeros((img_height, img_width), dtype=np.float32)
    # depth_image[top:bottom, left:right] = cropped_image
    # depth_image[depth_image == 0] = max_depth

    # return depth_image

    # depth_image = io.imread(filename)
    # # io.imshow(depth_image)
    # # io.show()
    # # resize the input depth map
    # depth_image = resize(depth_image, (240, 320, 3))[:, :, 0]
    # # io.imshow(depth_image, cmap=plt.cm.gray)
    # # io.show()
    # return depth_image * 1000
    # temporary: must be changed ###########################################################################################
    def subtract_depth(pcl):
        shallowest_point = abs(np.asarray(pcl.points)[:, 2]).min()
        arg_not_acceptable = np.argwhere(abs(np.asarray(pcl.points)[:, 2]) > shallowest_point + 0.07)
        _xyz = np.asarray(pcl.points)
        _rgb = np.asarray(pcl.colors)
        _xyz[arg_not_acceptable, :] = np.array((None, None, None))
        _rgb[arg_not_acceptable, :] = np.array((None, None, None))
        _pcl = o3d.geometry.PointCloud()
        _pcl.colors = o3d.utility.Vector3dVector(_rgb)
        _pcl.points = o3d.utility.Vector3dVector(_xyz)
        return _pcl

    color_raw = o3d.io.read_image('/home/mahdi/HVR/hvr/hand_pcl_iPhone/Tom_set_2/iPhone/hand30wall50_color.png')
    depth_raw = o3d.io.read_image(filename)
    color_raw = o3d.geometry.Image(np.asarray(color_raw))
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, depth_scale=0.529, depth_trunc=30.0, convert_rgb_to_intensity=False)
    # iPhone calibration
    h = np.asarray(color_raw).shape[0]  # 480
    w = np.asarray(color_raw).shape[1]  # 640
    iw = 3088.0
    ih = 2316.0
    xscale = h / ih
    yscale = w / iw
    _fx = 2880.0796 * xscale
    _fy = 2880.0796 * yscale
    # _cx = 1546.5824 * xscale
    # _cy = 1153.2035 * yscale
    _cx = 1153.2035 * xscale
    _cy = 1546.5824 * yscale
    setIntrinsic = o3d.camera.PinholeCameraIntrinsic()
    setIntrinsic.set_intrinsics(width=w, height=h, fx=_fx, fy=_fy, cx=_cx, cy=_cy)
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        setIntrinsic)
    pcd = subtract_depth(pcd)
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    z_values = -(np.asarray(pcd.points)[:, 2] * 1000)  # in mm
    depth_map = np.reshape(z_values, (480, 640))
    imgdata = np.asarray(Image.fromarray(depth_map).resize((320, 240)))
    # temporary: must be changed ###########################################################################################
    return np.copy(imgdata)


class HandDataset(Dataset):
    def __init__(self, root, center_dir, mode, test_subject_id, transform=None):
        self.img_width = 320
        self.img_height = 240
        self.min_depth = 100
        self.max_depth = 400

        # iPhone calibration
        iw = 3088.0
        ih = 2316.0
        xscale = self.img_height / ih
        yscale = self.img_width / iw

        # cx and cy maybe would need to be replaced
        self.cx = 1153.2035 * xscale
        self.cy = 1546.5824 * yscale
        self.fx = 2880.0796 * xscale
        self.fy = 2880.0796 * yscale
        self.joint_num = 21
        self.world_dim = 3
        self.folder_list = ['5']
        self.subject_num = 1

        self.root = root
        self.center_dir = center_dir
        self.mode = mode
        self.test_subject_id = test_subject_id
        self.transform = transform

        if not self.mode in ['train', 'test']: raise ValueError('Invalid mode')
        assert self.test_subject_id >= 0 and self.test_subject_id < self.subject_num
        # check if joint.txt exists: '/home/mahdi/HVR/git_repos/V2V-PoseNet-pytorch/datasets/msra-hand/P0/1/joint.txt'
        if not self._check_exists(): raise RuntimeError('Invalid MSRA hand dataset')

        self._load()

    def __getitem__(self, index):
        depthmap = load_depthmap(self.names[index], self.img_width, self.img_height, self.max_depth)
        points = depthmap2points(depthmap, self.fx, self.fy, self.cx, self.cy)
        points = points.reshape((-1, 3))

        sample = {
            'name': self.names[index],
            'points': points,
            'joints': self.joints_world[index],
            'refpoint': self.ref_pts[index]
        }

        # if self.transform: sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.num_samples

    def _load(self):
        self._compute_dataset_size()

        self.num_samples = self.train_size if self.mode == 'train' else self.test_size
        self.joints_world = np.zeros((self.num_samples, self.joint_num, self.world_dim))
        self.ref_pts = np.zeros((self.num_samples, self.world_dim))
        self.names = []

        # Collect reference center points strings
        if self.mode == 'train':
            ref_pt_file = 'center_train_' + str(self.test_subject_id) + '_refined.txt'
        else:
            ref_pt_file = 'center_test_' + str(self.test_subject_id) + '_refined.txt'

        with open(os.path.join(self.center_dir, ref_pt_file)) as f:
            ref_pt_str = [l.rstrip() for l in f]

        #
        file_id = 0
        frame_id = 0

        for mid in range(self.subject_num):
            if self.mode == 'train':
                model_chk = (mid != self.test_subject_id)
            elif self.mode == 'test':
                model_chk = (mid == self.test_subject_id)
            else:
                raise RuntimeError('unsupported mode {}'.format(self.mode))

            if model_chk:
                for fd in self.folder_list:
                    annot_file = os.path.join(self.root, 'P' + str(mid), fd, 'joint.txt')

                    lines = []
                    with open(annot_file) as f:
                        lines = [line.rstrip() for line in f]

                    # skip first line
                    for i in range(1, len(lines)):
                        # referece point
                        splitted = ref_pt_str[file_id].split()
                        if splitted[0] == 'invalid':
                            print('Warning: found invalid reference frame')
                            file_id += 1
                            continue
                        else:
                            self.ref_pts[frame_id, 0] = float(splitted[0])
                            self.ref_pts[frame_id, 1] = float(splitted[1])
                            self.ref_pts[frame_id, 2] = float(splitted[2])

                        # joint point
                        splitted = lines[i].split()
                        for jid in range(self.joint_num):
                            self.joints_world[frame_id, jid, 0] = float(splitted[jid * self.world_dim])
                            self.joints_world[frame_id, jid, 1] = float(splitted[jid * self.world_dim + 1])
                            self.joints_world[frame_id, jid, 2] = -float(splitted[jid * self.world_dim + 2])

                        filename = os.path.join(self.root, 'P' + str(mid), fd, '{:0>6d}'.format(i - 1) + '_depth.png')
                        self.names.append(filename)

                        frame_id += 1
                        file_id += 1

    def _compute_dataset_size(self):
        self.train_size, self.test_size = 0, 0

        for mid in range(self.subject_num):
            num = 0
            for fd in self.folder_list:
                annot_file = os.path.join(self.root, 'P' + str(mid), fd, 'joint.txt')
                with open(annot_file) as f:
                    num = int(f.readline().rstrip())
                if mid == self.test_subject_id:
                    self.test_size += num
                else:
                    self.train_size += num

    def _check_exists(self):
        # Check basic data
        for mid in range(self.subject_num):
            for fd in self.folder_list:
                annot_file = os.path.join(self.root, 'P' + str(mid), fd, 'joint.txt')
                if not os.path.exists(annot_file):
                    print('Error: annotation file {} does not exist'.format(annot_file))
                    return False

        # Check precomputed centers by v2v-hand model's author
        for subject_id in range(self.subject_num):
            center_train = os.path.join(self.center_dir, 'center_train_' + str(subject_id) + '_refined.txt')
            center_test = os.path.join(self.center_dir, 'center_test_' + str(subject_id) + '_refined.txt')
            if not os.path.exists(center_train) or not os.path.exists(center_test):
                print('Error: precomputed center files do not exist')
                return False

        return True
