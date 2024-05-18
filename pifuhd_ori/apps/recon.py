# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import time
import json
import numpy as np
import cv2
import random
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
from numpy.linalg import inv

from pifuhd_ori.lib.options import BaseOptions
from pifuhd_ori.lib.mesh_util import save_obj_mesh_with_color, reconstruction
from pifuhd_ori.lib.data import EvalWPoseDataset, EvalDataset
from pifuhd_ori.lib.model import HGPIFuNetwNML, HGPIFuMRNet
from pifuhd_ori.lib.geometry import index
import torch.nn.functional as F
from PIL import Image

parser = BaseOptions()


def gen_mesh(res, net, cuda, image, norm_F=None, norm_B=None, thresh=0.5, use_octree=True, components=False):
    image_tensor_global = image.to(device=cuda)
    image_tensor = F.interpolate(image_tensor_global, (1024, 1024), mode='bilinear', align_corners=False)
    feature = net.filter_global(image_tensor_global, norm_F, norm_B)
    feature_fine = net.filter_local(image_tensor[:, None], norm_F, norm_B)
    return feature, feature_fine


def recon(opt, use_rect=False, image=None, norm_F=None, norm_B=None):
    # load checkpoints
    state_dict_path = None
    if opt.load_netMR_checkpoint_path is not None:
        state_dict_path = opt.load_netMR_checkpoint_path
    elif opt.resume_epoch < 0:
        state_dict_path = '%s/%s_train_latest' % (opt.checkpoints_path, opt.name)
        opt.resume_epoch = 0
    else:
        state_dict_path = '%s/%s_train_epoch_%d' % (opt.checkpoints_path, opt.name, opt.resume_epoch)
    cuda = torch.device('cuda:%d' % opt.gpu_id if torch.cuda.is_available() else 'cpu')

    state_dict = None
    if state_dict_path is not None and os.path.exists(state_dict_path):
        print('Resuming from ', state_dict_path)
        state_dict = torch.load(state_dict_path, map_location=cuda)
        print('Warning: opt is overwritten.')
        dataroot = opt.dataroot
        resolution = opt.resolution
        results_path = opt.results_path
        loadSize = opt.loadSize

        opt = state_dict['opt']
        opt.dataroot = dataroot
        opt.resolution = resolution
        opt.results_path = results_path
        opt.loadSize = loadSize
    else:
        raise Exception('failed loading state dict!', state_dict_path)

    opt_netG = state_dict['opt_netG']
    netG = HGPIFuNetwNML(opt_netG).to(device=cuda)
    netMR = HGPIFuMRNet(opt, netG).to(device=cuda)

    def set_eval():
        netG.eval()

    netMR.load_state_dict(state_dict['model_state_dict'])
    torch.cuda.empty_cache()
    with torch.no_grad():
        set_eval()

        feature, feature_fine = gen_mesh(opt.resolution, netMR, cuda, image, norm_F, norm_B, components=opt.use_compose)

    return feature, feature_fine


def reconWrapper(args=None, use_rect=False, image=None, norm_F=None, norm_B=None):
    opt = parser.parse(args)
    feature, feature_fine = recon(opt, use_rect, image, norm_F, norm_B)

    return feature, feature_fine


if __name__ == '__main__':
    feature, feature_fine = reconWrapper()
