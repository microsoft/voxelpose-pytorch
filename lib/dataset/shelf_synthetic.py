# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset

import json_tricks as json
import pickle
import logging
import copy
import random
import cv2

import os

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import rotate_points, get_scale
from utils.cameras_cpu import project_pose

logger = logging.getLogger(__name__)

coco_joints_def = {0: 'nose',
                   1: 'Leye', 2: 'Reye', 3: 'Lear', 4: 'Rear',
                   5: 'Lsho', 6: 'Rsho',
                   7: 'Lelb', 8: 'Relb',
                   9: 'Lwri', 10: 'Rwri',
                   11: 'Lhip', 12: 'Rhip',
                   13: 'Lkne', 14: 'Rkne',
                   15: 'Lank', 16: 'Rank'}

LIMBS = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [11, 13], [13, 15],
        [6, 12], [12, 14], [14, 16], [5, 6], [11, 12]]


class ShelfSynthetic(Dataset):
    def __init__(self, cfg, image_set, is_train, transform=None):
        super().__init__()
        self.pixel_std = 200.0
        self.joints_def = coco_joints_def
        self.limbs = LIMBS
        self.num_joints = len(coco_joints_def)
        self.cam_list = [0, 1, 2, 3, 4]
        self.num_views = len(self.cam_list)
        self.maximum_person = cfg.MULTI_PERSON.MAX_PEOPLE_NUM

        self.is_train = is_train

        this_dir = os.path.dirname(__file__)
        dataset_root = os.path.join(this_dir, '../..', cfg.DATASET.ROOT)
        self.dataset_root = dataset_root
        self.image_set = image_set
        self.dataset_name = cfg.DATASET.TEST_DATASET

        self.data_format = cfg.DATASET.DATA_FORMAT
        self.data_augmentation = cfg.DATASET.DATA_AUGMENTATION

        self.color_rgb = cfg.DATASET.COLOR_RGB

        self.target_type = cfg.NETWORK.TARGET_TYPE
        self.image_size = np.array(cfg.NETWORK.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.NETWORK.HEATMAP_SIZE)
        self.sigma = cfg.NETWORK.SIGMA
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1

        self.transform = transform

        self.space_size = np.array(cfg.MULTI_PERSON.SPACE_SIZE)
        self.space_center = np.array(cfg.MULTI_PERSON.SPACE_CENTER)
        self.initial_cube_size = np.array(cfg.MULTI_PERSON.INITIAL_CUBE_SIZE)

        pose_db_file = os.path.join(self.dataset_root, "..", "panoptic_training_pose.pkl")
        self.pose_db = pickle.load(open(pose_db_file, "rb"))
        self.cameras = self._get_cam()

    def _get_cam(self):
        cam_file = osp.join(self.dataset_root, "calibration_shelf.json")
        with open(cam_file) as cfile:
            cameras = json.load(cfile)

        for id, cam in cameras.items():
            for k, v in cam.items():
                cameras[id][k] = np.array(v)

        return cameras

    def __getitem__(self, idx):
        # nposes = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.4, 0.2])
        nposes = np.random.choice(range(1, 6))
        bbox_list = []
        center_list = []

        select_poses = np.random.choice(self.pose_db, nposes)
        joints_3d = np.array([p['pose'] for p in select_poses])
        joints_3d_vis = np.array([p['vis'] for p in select_poses])

        for n in range(0, nposes):
            points = joints_3d[n][:, :2].copy()
            center = (points[11, :2] + points[12, :2]) / 2
            rot_rad = np.random.uniform(-180, 180)

            new_center = self.get_new_center(center_list)
            new_xy = rotate_points(points, center, rot_rad) - center + new_center

            loop_count = 0
            while not self.isvalid(self.calc_bbox(new_xy, joints_3d_vis[n]), bbox_list):
                loop_count += 1
                if loop_count >= 100:
                    break
                new_center = self.get_new_center(center_list)
                new_xy = rotate_points(points, center, rot_rad) - center + new_center

            if loop_count >= 100:
                nposes = n
                joints_3d = joints_3d[:n]
                joints_3d_vis = joints_3d_vis[:n]
            else:
                center_list.append(new_center)
                bbox_list.append(self.calc_bbox(new_xy, joints_3d_vis[n]))
                joints_3d[n][:, :2] = new_xy

        input, target_heatmap, target_weight, target_3d, meta, input_heatmap = [], [], [], [], [], []
        for k, cam in self.cameras.items():
            i, th, tw, t3, m, ih = self._get_single_view_item(joints_3d, joints_3d_vis, cam)
            input.append(i)
            target_heatmap.append(th)
            target_weight.append(tw)
            input_heatmap.append(ih)
            target_3d.append(t3)
            meta.append(m)
        return input, target_heatmap, target_weight, target_3d, meta, input_heatmap

    def __len__(self):
        return 3000
        # return self.db_size // self.num_views

    def _get_single_view_item(self, joints_3d, joints_3d_vis, cam):
        joints_3d = copy.deepcopy(joints_3d)
        joints_3d_vis = copy.deepcopy(joints_3d_vis)
        nposes = len(joints_3d)

        width = 1032
        height = 776
        c = np.array([width / 2.0, height / 2.0], dtype=np.float32)
        # s = np.array(
        #     [width / self.pixel_std, height / self.pixel_std], dtype=np.float32)
        s = get_scale((width, height), self.image_size)
        r = 0

        joints = []
        joints_vis = []
        for n in range(nposes):
            pose2d = project_pose(joints_3d[n], cam)

            x_check = np.bitwise_and(pose2d[:, 0] >= 0,
                                     pose2d[:, 0] <= width - 1)
            y_check = np.bitwise_and(pose2d[:, 1] >= 0,
                                     pose2d[:, 1] <= height - 1)
            check = np.bitwise_and(x_check, y_check)
            vis = joints_3d_vis[n][:, 0] > 0
            vis[np.logical_not(check)] = 0

            joints.append(pose2d)
            joints_vis.append(np.repeat(np.reshape(vis, (-1, 1)), 2, axis=1))

        trans = get_affine_transform(c, s, r, self.image_size)
        input = np.ones((height, width, 3), dtype=np.float32)
        input = cv2.warpAffine(
            input,
            trans, (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            input = self.transform(input)

        for n in range(nposes):
            for i in range(len(joints[0])):
                if joints_vis[n][i, 0] > 0.0:
                    joints[n][i, 0:2] = affine_transform(
                        joints[n][i, 0:2], trans)
                    if (np.min(joints[n][i, :2]) < 0 or
                            joints[n][i, 0] >= self.image_size[0] or
                            joints[n][i, 1] >= self.image_size[1]):
                        joints_vis[n][i, :] = 0

        input_heatmap, _ = self.generate_input_heatmap(
            joints, joints_vis)
        input_heatmap = torch.from_numpy(input_heatmap)
        target_heatmap = torch.zeros_like(input_heatmap)
        target_weight = torch.zeros(len(target_heatmap), 1)

        # make joints and joints_vis having same shape
        joints_u = np.zeros((self.maximum_person, len(joints[0]), 2))
        joints_vis_u = np.zeros((self.maximum_person, len(joints[0]), 2))
        for i in range(nposes):
            joints_u[i] = joints[i]
            joints_vis_u[i] = joints_vis[i]

        joints_3d_u = np.zeros((self.maximum_person, len(joints[0]), 3))
        joints_3d_vis_u = np.zeros((self.maximum_person, len(joints[0]), 3))
        for i in range(nposes):
            joints_3d_u[i] = joints_3d[i][:, 0:3]
            joints_3d_vis_u[i] = joints_3d_vis[i][:, 0:3]

        target_3d = self.generate_3d_target(joints_3d)
        target_3d = torch.from_numpy(target_3d)

        meta = {
            'image': '',
            'num_person': nposes,
            'joints_3d': joints_3d_u,
            'roots_3d': (joints_3d_u[:, 11] + joints_3d_u[:, 12]) / 2.0,
            'joints_3d_vis': joints_3d_vis_u,
            'joints': joints_u,
            'joints_vis': joints_vis_u,
            'center': c,
            'scale': s,
            'rotation': r,
            'camera': cam
        }

        return input, target_heatmap, target_weight, target_3d, meta, input_heatmap

    @staticmethod
    def compute_human_scale(pose, joints_vis):
        idx = joints_vis[:, 0] == 1
        if np.sum(idx) == 0:
            return 0
        minx, maxx = np.min(pose[idx, 0]), np.max(pose[idx, 0])
        miny, maxy = np.min(pose[idx, 1]), np.max(pose[idx, 1])
        return np.clip(np.maximum(maxy - miny, maxx - minx) ** 2, 1.0 / 4 * 96 ** 2, 4 * 96 ** 2)

    def generate_input_heatmap(self, joints, joints_vis):
        '''
        :param joints:  [[num_joints, 3]]
        :param joints_vis: [num_joints, 3]
        :return: input_heatmap
        '''
        nposes = len(joints)
        num_joints = joints[0].shape[0]
        target_weight = np.zeros((num_joints, 1), dtype=np.float32)
        for i in range(num_joints):
            for n in range(nposes):
                if joints_vis[n][i, 0] == 1:
                    target_weight[i, 0] = 1

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros(
                (num_joints, self.heatmap_size[1], self.heatmap_size[0]),
                dtype=np.float32)
            feat_stride = self.image_size / self.heatmap_size

            for n in range(nposes):
                obscured = random.random() < 0.05
                if obscured:
                    continue
                human_scale = 2 * self.compute_human_scale(joints[n] / feat_stride, joints_vis[n])
                if human_scale == 0:
                    continue

                cur_sigma = self.sigma * np.sqrt((human_scale / (96.0 * 96.0)))
                tmp_size = cur_sigma * 3
                for joint_id in range(num_joints):
                    feat_stride = self.image_size / self.heatmap_size
                    mu_x = int(joints[n][joint_id][0] / feat_stride[0])
                    mu_y = int(joints[n][joint_id][1] / feat_stride[1])
                    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                    if joints_vis[n][joint_id, 0] == 0 or \
                            ul[0] >= self.heatmap_size[0] or \
                            ul[1] >= self.heatmap_size[1] \
                            or br[0] < 0 or br[1] < 0:
                        continue

                    size = 2 * tmp_size + 1
                    x = np.arange(0, size, 1, np.float32)
                    y = x[:, np.newaxis]
                    x0 = y0 = size // 2
                    # scale = 1 - np.abs(np.random.randn(1) * 0.25)
                    scale = 0.9 + np.random.randn(1) * 0.03 if random.random() < 0.6 else 1.0
                    if joint_id in [7, 8, 13, 14]:
                        scale = scale * 0.5 if random.random() < 0.1 else scale
                    elif joint_id in [9, 10, 15, 16]:
                        scale = scale * 0.2 if random.random() < 0.1 else scale
                    else:
                        scale = scale * 0.5 if random.random() < 0.05 else scale
                    g = np.exp(
                        -((x - x0) ** 2 + (y - y0) ** 2) / (2 * cur_sigma ** 2)) * scale

                    # Usable gaussian range
                    g_x = max(0,
                              -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                    g_y = max(0,
                              -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                    img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = np.maximum(
                        target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]],
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]])
                target = np.clip(target, 0, 1)

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

    def generate_3d_target(self, joints_3d):
        num_people = len(joints_3d)

        space_size = self.space_size
        space_center = self.space_center
        cube_size = self.initial_cube_size
        grid1Dx = np.linspace(-space_size[0] / 2, space_size[0] / 2, cube_size[0]) + space_center[0]
        grid1Dy = np.linspace(-space_size[1] / 2, space_size[1] / 2, cube_size[1]) + space_center[1]
        grid1Dz = np.linspace(-space_size[2] / 2, space_size[2] / 2, cube_size[2]) + space_center[2]

        target = np.zeros((cube_size[0], cube_size[1], cube_size[2]), dtype=np.float32)
        cur_sigma = 200.0

        for n in range(num_people):
            joint_id = [11, 12]  # mid-hip
            mu_x = (joints_3d[n][joint_id[0]][0] + joints_3d[n][joint_id[1]][0]) / 2.0
            mu_y = (joints_3d[n][joint_id[0]][1] + joints_3d[n][joint_id[1]][1]) / 2.0
            mu_z = (joints_3d[n][joint_id[0]][2] + joints_3d[n][joint_id[1]][2]) / 2.0

            i_x = [np.searchsorted(grid1Dx, mu_x - 3 * cur_sigma),
                   np.searchsorted(grid1Dx, mu_x + 3 * cur_sigma, 'right')]
            i_y = [np.searchsorted(grid1Dy, mu_y - 3 * cur_sigma),
                   np.searchsorted(grid1Dy, mu_y + 3 * cur_sigma, 'right')]
            i_z = [np.searchsorted(grid1Dz, mu_z - 3 * cur_sigma),
                   np.searchsorted(grid1Dz, mu_z + 3 * cur_sigma, 'right')]
            if i_x[0] >= i_x[1] or i_y[0] >= i_y[1] or i_z[0] >= i_z[1]:
                continue

            gridx, gridy, gridz = np.meshgrid(grid1Dx[i_x[0]:i_x[1]], grid1Dy[i_y[0]:i_y[1]], grid1Dz[i_z[0]:i_z[1]],
                                              indexing='ij')
            g = np.exp(-((gridx - mu_x) ** 2 + (gridy - mu_y) ** 2 + (gridz - mu_z) ** 2) / (2 * cur_sigma ** 2))
            target[i_x[0]:i_x[1], i_y[0]:i_y[1], i_z[0]:i_z[1]] = np.maximum(
                target[i_x[0]:i_x[1], i_y[0]:i_y[1], i_z[0]:i_z[1]], g)

        target = np.clip(target, 0, 1)
        return target

    def evaluate(self):
        pass

    @staticmethod
    def get_new_center(center_list):
        if len(center_list) == 0 or random.random() < 0.7:
            new_center = np.array([np.random.uniform(-1000.0, 2000.0), np.random.uniform(-1600.0, 1600.0)])
        else:
            xy = center_list[np.random.choice(range(len(center_list)))]
            new_center = xy + np.random.normal(500, 50, 2) * np.random.choice([1, -1], 2)

        return new_center

    @staticmethod
    def isvalid(bbox, bbox_list):
        if len(bbox_list) == 0:
            return True

        bbox_list = np.array(bbox_list)
        x0 = np.maximum(bbox[0], bbox_list[:, 0])
        y0 = np.maximum(bbox[1], bbox_list[:, 1])
        x1 = np.minimum(bbox[2], bbox_list[:, 2])
        y1 = np.minimum(bbox[3], bbox_list[:, 3])

        intersection = np.maximum(0, (x1 - x0) * (y1 - y0))
        area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        area_list = (bbox_list[:, 2] - bbox_list[:, 0]) * (bbox_list[:, 3] - bbox_list[:, 1])
        iou_list = intersection / (area + area_list - intersection)

        return np.max(iou_list) < 0.01

    @staticmethod
    def calc_bbox(pose, pose_vis):
        index = pose_vis[:, 0] > 0
        bbox = [np.min(pose[index, 0]), np.min(pose[index, 1]),
                np.max(pose[index, 0]), np.max(pose[index, 1])]

        return np.array(bbox)
