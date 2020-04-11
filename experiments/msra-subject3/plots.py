import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import argparse
import os

from lib.solver import train_epoch, val_epoch, test_epoch
from lib.sampler import ChunkSampler
from src.v2v_model import V2VModel
from src.v2v_util import V2VVoxelization

from datasets.msra_hand import MARAHandDataset
from datasets.msra_hand_points_xyz import HandDataset

## Some helpers
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Hand Keypoints Estimation Training')
    #parser.add_argument('--resume', 'r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--resume', '-r', default=14, type=int, help='resume after epoch')
    args = parser.parse_args()
    return args


#######################################################################################
## Configurations
print('Warning: disable cudnn for batchnorm first, or just use only cuda instead!')


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
dtype = torch.float

#
args = parse_args()
resume_train = args.resume >= 0
resume_after_epoch = args.resume

save_checkpoint = True
checkpoint_per_epochs = 1
checkpoint_dir = r'/home/mahdi/HVR/git_repos/V2V-PoseNet-pytorch/checkpoint/'

start_epoch = 0
epochs_num = 0

batch_size = 1


#######################################################################################
## Data, transform, dataset and loader
# Data
print('==> Preparing data ..')
# data_dir = r'/home/mahdi/HVR/git_repos/V2V-PoseNet-pytorch/datasets/msra-hand'
# center_dir = r'/home/mahdi/HVR/git_repos/V2V-PoseNet-pytorch/datasets/msra-hand-center'
data_dir = r'/home/mahdi/HVR/git_repos/V2V-PoseNet-pytorch/datasets/msra-hand'
center_dir = r'/home/mahdi/HVR/git_repos/V2V-PoseNet-pytorch/datasets/msra-hand-center'
keypoints_num = 21
test_subject_id = 0
cubic_size = 200


# Transform
voxelization_train = V2VVoxelization(cubic_size=200, augmentation=True)
voxelization_val = V2VVoxelization(cubic_size=200, augmentation=False)


def transform_train(sample):
    points, keypoints, refpoint = sample['points'], sample['joints'], sample['refpoint']
    assert(keypoints.shape[0] == keypoints_num)
    input, heatmap = voxelization_train({'points': points, 'keypoints': keypoints, 'refpoint': refpoint})
    return (torch.from_numpy(input), torch.from_numpy(heatmap))


def transform_val(sample):
    points, keypoints, refpoint = sample['points'], sample['joints'], sample['refpoint']
    assert(keypoints.shape[0] == keypoints_num)
    input, heatmap = voxelization_val({'points': points, 'keypoints': keypoints, 'refpoint': refpoint})
    return (torch.from_numpy(input), torch.from_numpy(heatmap))

## Test
print('==> Testing ..')
voxelize_input = voxelization_train.voxelize
evaluate_keypoints = voxelization_train.evaluate


def transform_test(sample):
    points, refpoint = sample['points'], sample['refpoint']
    input = voxelize_input(points, refpoint)
    return torch.from_numpy(input), torch.from_numpy(refpoint.reshape((1, -1)))

# plot input depth map with estimated joints
test_id = 0 # id of test frame to plot
keypoints_test = np.loadtxt('/home/mahdi/HVR/git_repos/V2V-PoseNet-pytorch/experiments/msra-subject3/results/test_res.txt')
keypoints_test = keypoints_test.reshape(keypoints_test.shape[0], 21, 3)
input_points_xyz = HandDataset(data_dir, center_dir, 'test', test_subject_id, transform_test).__getitem__(test_id)['points']
input_center = HandDataset(data_dir, center_dir, 'test', test_subject_id, transform_test).ref_pts[test_id]
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12,12))
ax = fig.gca(projection='3d')

ax.scatter(input_points_xyz[:, 0], input_points_xyz[:, 1], input_points_xyz[:, 2], marker="o", s=.01, label='depth map')
# keypoints_test = np.squeeze(keypoints_test)
ax.scatter(keypoints_test[test_id, :, 0], keypoints_test[test_id, :, 1], keypoints_test[test_id, :, 2], marker="x", c='red', s=10, label='estimated hand joints')
# ax.scatter(input_center[0], input_center[1], input_center[2], marker="X", c='black', s=10, label='refined hand center')
ax.view_init(elev=90, azim=0)
ax.set_title('V2V hand joint estimator: MSRA15')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.savefig('./results/pcl_joints_{}'.format(test_id))
ax.view_init(elev=0, azim=0)
plt.show()

plt.close()
print('All done ..')
