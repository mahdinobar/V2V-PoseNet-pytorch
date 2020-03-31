import os
import numpy as np
import sys
import struct
from torch.utils.data import Dataset
from skimage import io
from skimage.transform import resize
import matplotlib.pyplot as plt

def pixel2world(x, y, z, img_width, img_height, fx, fy, cx, cy):
    w_x = (x - cx) * z / fx
    w_y = (cy - y) * z / fy
    w_z = z
    return w_x, w_y, w_z


def world2pixel(x, y, z, img_width, img_height, fx, fy, cx, cy):
    p_x = x * fx / z + cx
    p_y = cy - y * fy / z
    return p_x, p_y


def depthmap2points(image, fx, fy, cx, cy):
    h, w = image.shape
    x, y = np.meshgrid(np.arange(w) + 1, np.arange(h) + 1)
    points = np.zeros((h, w, 3), dtype=np.float32)
    points[:,:,0], points[:,:,1], points[:,:,2] = pixel2world(x, y, image, w, h, fx, fy, cx, cy)
    return points


def points2pixels(points, img_width, img_height, fx, fy, cx, cy):
    pixels = np.zeros((points.shape[0], 2))
    pixels[:, 0], pixels[:, 1] = \
        world2pixel(points[:,0], points[:, 1], points[:, 2], img_width, img_height, fx, fy, cx, cy)
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

    depth_image = io.imread(filename)
    io.imshow(depth_image)
    io.show()
    # resize the input depth map
    depth_image = resize(depth_image, (240, 320, 3))[:, :, 0]
    io.imshow(depth_image, cmap=plt.cm.gray)
    io.show()
    return depth_image * 1000

class MARAHandDataset(Dataset):
    def __init__(self, root, center_dir, mode, test_subject_id, transform=None):
        self.img_width = 320
        self.img_height = 240
        self.min_depth = 100
        self.max_depth = 700

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

        if self.transform: sample = self.transform(sample)

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
        if self.mode == 'train': ref_pt_file = 'center_train_' + str(self.test_subject_id) + '_refined.txt'
        else: ref_pt_file = 'center_test_' + str(self.test_subject_id) + '_refined.txt'

        with open(os.path.join(self.center_dir, ref_pt_file)) as f:
                ref_pt_str = [l.rstrip() for l in f]

        #
        file_id = 0
        frame_id = 0

        for mid in range(self.subject_num):
            if self.mode == 'train': model_chk = (mid != self.test_subject_id)
            elif self.mode == 'test': model_chk = (mid == self.test_subject_id)
            else: raise RuntimeError('unsupported mode {}'.format(self.mode))
            
            if model_chk:
                for fd in self.folder_list:
                    annot_file = os.path.join(self.root, 'P'+str(mid), fd, 'joint.txt')

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
                        
                        filename = os.path.join(self.root, 'P'+str(mid), fd, '{:0>6d}'.format(i-1) + '_depth.png')
                        self.names.append(filename)

                        frame_id += 1
                        file_id += 1

    def _compute_dataset_size(self):
        self.train_size, self.test_size = 0, 0

        for mid in range(self.subject_num):
            num = 0
            for fd in self.folder_list:
                annot_file = os.path.join(self.root, 'P'+str(mid), fd, 'joint.txt')
                with open(annot_file) as f:
                    num = int(f.readline().rstrip())
                if mid == self.test_subject_id: self.test_size += num
                else: self.train_size += num

    def _check_exists(self):
        # Check basic data
        for mid in range(self.subject_num):
            for fd in self.folder_list:
                annot_file = os.path.join(self.root, 'P'+str(mid), fd, 'joint.txt')
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
