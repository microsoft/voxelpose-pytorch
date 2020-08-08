from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import maximum_filter


def get_index(indices, shape):
    batch_size = indices.shape[0]
    num_people = indices.shape[1]
    indices_x = (indices // (shape[1] * shape[2])).reshape(batch_size, num_people, -1)
    indices_y = ((indices % (shape[1] * shape[2])) // shape[2]).reshape(batch_size, num_people, -1)
    indices_z = (indices % shape[2]).reshape(batch_size, num_people, -1)
    indices = torch.cat([indices_x, indices_y, indices_z], dim=2)
    return indices


def max_pool(inputs, kernel=3):
    padding = (kernel - 1) // 2
    max = F.max_pool3d(inputs, kernel_size=kernel, stride=1, padding=padding)
    keep = (inputs == max).float()
    return keep * inputs


def nms(root_cubes, max_num):
    batch_size = root_cubes.shape[0]
    # root_cubes_nms = torch.zeros_like(root_cubes, device=root_cubes.device)
    #
    # for b in range(batch_size):
    #     mx = torch.as_tensor(maximum_filter(root_cubes[b].detach().cpu().numpy(), size=3),
    #                          dtype=torch.float, device=root_cubes.device)
    #     root_cubes_nms[b] = (mx == root_cubes[b]).float() * root_cubes[b]
    root_cubes_nms = max_pool(root_cubes)
    root_cubes_nms_reshape = root_cubes_nms.reshape(batch_size, -1)
    topk_values, topk_index = root_cubes_nms_reshape.topk(max_num)
    topk_unravel_index = get_index(topk_index, root_cubes[0].shape)

    return topk_values, topk_unravel_index
