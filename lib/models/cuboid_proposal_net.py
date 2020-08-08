# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn

from models.v2v_net import V2VNet
from models.project_layer import ProjectLayer
from core.proposal import nms


class ProposalLayer(nn.Module):
    def __init__(self, cfg):
        super(ProposalLayer, self).__init__()
        self.grid_size = torch.tensor(cfg.MULTI_PERSON.SPACE_SIZE)
        self.cube_size = torch.tensor(cfg.MULTI_PERSON.INITIAL_CUBE_SIZE)
        self.grid_center = torch.tensor(cfg.MULTI_PERSON.SPACE_CENTER)
        self.num_cand = cfg.MULTI_PERSON.MAX_PEOPLE_NUM
        self.root_id = cfg.DATASET.ROOTIDX
        self.num_joints = cfg.NETWORK.NUM_JOINTS
        self.threshold = cfg.MULTI_PERSON.THRESHOLD

    def filter_proposal(self, topk_index, gt_3d, num_person):
        batch_size = topk_index.shape[0]
        cand_num = topk_index.shape[1]
        cand2gt = torch.zeros(batch_size, cand_num)

        for i in range(batch_size):
            cand = topk_index[i].reshape(cand_num, 1, -1)
            gt = gt_3d[i, :num_person[i]].reshape(1, num_person[i], -1)

            dist = torch.sqrt(torch.sum((cand - gt)**2, dim=-1))
            min_dist, min_gt = torch.min(dist, dim=-1)

            cand2gt[i] = min_gt
            cand2gt[i][min_dist > 500.0] = -1.0

        return cand2gt

    def get_real_loc(self, index):
        device = index.device
        cube_size = self.cube_size.to(device=device, dtype=torch.float)
        grid_size = self.grid_size.to(device=device)
        grid_center = self.grid_center.to(device=device)
        loc = index.float() / (cube_size - 1) * grid_size + grid_center - grid_size / 2.0
        return loc

    def forward(self, root_cubes, meta):
        batch_size = root_cubes.shape[0]

        topk_values, topk_unravel_index = nms(root_cubes.detach(), self.num_cand)
        topk_unravel_index = self.get_real_loc(topk_unravel_index)

        grid_centers = torch.zeros(batch_size, self.num_cand, 5, device=root_cubes.device)
        grid_centers[:, :, 0:3] = topk_unravel_index
        grid_centers[:, :, 4] = topk_values

        # match gt to filter those invalid proposals for training/validate PRN
        if self.training and ('roots_3d' in meta[0] and 'num_person' in meta[0]):
            gt_3d = meta[0]['roots_3d'].float()
            num_person = meta[0]['num_person']
            cand2gt = self.filter_proposal(topk_unravel_index, gt_3d, num_person)
            grid_centers[:, :, 3] = cand2gt
        else:
            grid_centers[:, :, 3] = (topk_values > self.threshold).float() - 1.0  # if ground-truths are not available.

        # nms
        # for b in range(batch_size):
        #     centers = copy.deepcopy(topk_unravel_index[b, :, :3])
        #     scores = copy.deepcopy(topk_values[b])
        #     keep = []
        #     keep_s = []
        #     while len(centers):
        #         keep.append(centers[0])
        #         keep_s.append(scores[0])
        #         dist = torch.sqrt(torch.sum((centers[0] - centers)**2, dim=-1))
        #         index = (dist > 500.0) & (scores > 0.1)
        #         centers = centers[index]
        #         scores = scores[index]
        #     grid_centers[b, :len(keep), :3] = torch.stack(keep, dim=0)
        #     grid_centers[b, :len(keep), 3] = 0.0
        #     grid_centers[b, :len(keep), 4] = torch.stack(keep_s, dim=0)

        return grid_centers


class CuboidProposalNet(nn.Module):
    def __init__(self, cfg):
        super(CuboidProposalNet, self).__init__()
        self.grid_size = cfg.MULTI_PERSON.SPACE_SIZE
        self.cube_size = cfg.MULTI_PERSON.INITIAL_CUBE_SIZE
        self.grid_center = cfg.MULTI_PERSON.SPACE_CENTER

        self.project_layer = ProjectLayer(cfg)
        self.v2v_net = V2VNet(cfg.NETWORK.NUM_JOINTS, 1)
        self.proposal_layer = ProposalLayer(cfg)

    def forward(self, all_heatmaps, meta):

        initial_cubes, grids = self.project_layer(all_heatmaps, meta,
                                                  self.grid_size, [self.grid_center], self.cube_size)
        root_cubes = self.v2v_net(initial_cubes)
        root_cubes = root_cubes.squeeze(1)
        grid_centers = self.proposal_layer(root_cubes, meta)

        return root_cubes, grid_centers