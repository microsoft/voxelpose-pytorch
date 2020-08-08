# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim

from core.config import get_model_name


def create_logger(cfg, cfg_name, phase='train'):
    this_dir = Path(os.path.dirname(__file__))  ##
    root_output_dir = (this_dir / '..' / '..' / cfg.OUTPUT_DIR).resolve()  ##
    tensorboard_log_dir = (this_dir / '..' / '..' / cfg.LOG_DIR).resolve()
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.TRAIN_DATASET
    model, _ = get_model_name(cfg)
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / model / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = tensorboard_log_dir / dataset / model / \
        (cfg_name + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)

def get_optimizer(cfg, model):
    optimizer = None
    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.TRAIN.LR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=cfg.TRAIN.LR
        )

    return optimizer


def load_model_state(model, output_dir, epoch):
    file = os.path.join(output_dir, 'checkpoint_3d_epoch'+str(epoch)+'.pth.tar')
    if os.path.isfile(file):
        model.module.load_state_dict(torch.load(file))
        print('=> load models state {} (epoch {})'
              .format(file, epoch))
        return model
    else:
        print('=> no checkpoint found at {}'.format(file))
        return model


def load_checkpoint(model, optimizer, output_dir, filename='checkpoint.pth.tar'):
    file = os.path.join(output_dir, filename)
    if os.path.isfile(file):
        checkpoint = torch.load(file)
        start_epoch = checkpoint['epoch']
        precision = checkpoint['precision'] if 'precision' in checkpoint else 0
        model.module.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('=> load checkpoint {} (epoch {})'
              .format(file, start_epoch))

        return start_epoch, model, optimizer, precision

    else:
        print('=> no checkpoint found at {}'.format(file))
        return 0, model, optimizer, 0


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'],
                   os.path.join(output_dir, 'model_best.pth.tar'))


def load_backbone_panoptic(model, pretrained_file):
    this_dir = os.path.dirname(__file__)
    pretrained_file = os.path.abspath(os.path.join(this_dir, '../..', pretrained_file))
    pretrained_state_dict = torch.load(pretrained_file)
    model_state_dict = model.module.backbone.state_dict()

    prefix = "module."
    new_pretrained_state_dict = {}
    for k, v in pretrained_state_dict.items():
        if k.replace(prefix, "") in model_state_dict and v.shape == model_state_dict[k.replace(prefix, "")].shape:
            new_pretrained_state_dict[k.replace(prefix, "")] = v
        elif k.replace(prefix, "") == "final_layer.weight":  # TODO
            print("Reiniting final layer filters:", k)

            o = torch.zeros_like(model_state_dict[k.replace(prefix, "")][:, :, :, :])
            nn.init.xavier_uniform_(o)
            n_filters = min(o.shape[0], v.shape[0])
            o[:n_filters, :, :, :] = v[:n_filters, :, :, :]

            new_pretrained_state_dict[k.replace(prefix, "")] = o
        elif k.replace(prefix, "") == "final_layer.bias":
            print("Reiniting final layer biases:", k)
            o = torch.zeros_like(model_state_dict[k.replace(prefix, "")][:])
            nn.init.zeros_(o)
            n_filters = min(o.shape[0], v.shape[0])
            o[:n_filters] = v[:n_filters]

            new_pretrained_state_dict[k.replace(prefix, "")] = o
    logging.info("load backbone statedict from {}".format(pretrained_file))
    model.module.backbone.load_state_dict(new_pretrained_state_dict)

    return model
