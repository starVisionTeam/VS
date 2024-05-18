import sys

import torch
import numpy as np
import os

import trimesh

from Mr.models.networks import *
from os.path import join
from Mr.util.util import print_network, sample_surface, local_nonuniform_penalty, cal_edge_loss
from Mr.models.loss import chamfer_distance1, BeamGapLoss
from pytorch3d.loss import chamfer_distance
from pytorch3d.ops.sample_points_from_meshes import sample_points_from_meshes


class ClassifierModel:
    """ Class for training Model weights

    :args opt: structure containing configuration params
    e.g.,
    --dataset_mode -> classification / segmentation)
    --arch -> network type
    """

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = join(opt.checkpoints_dir, opt.name)
        self.optimizer = None
        self.edge_features = None
        self.labels = None
        self.mesh = None
        self.num_samples = 50000
        self.label_path_vss = None
        self.label_path_norms = None
        self.part_mesh = None
        self.edge_features = None
        self.part_meshs = None
        self.label_path_vs = None
        self.label_path_norm = None
        self.edge_feature = None
        self.loss = 0
        self.n_loss = 0

        # load/define networks
        self.net = init_net(self.device, self.opt)
        self.net.train(self.is_train)

        if self.is_train:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.scheduler = get_scheduler(self.optimizer, opt)
            print_network(self.net)

        if not self.is_train or opt.continue_train:
            self.load_network(opt.which_epoch)

    def set_input(self, data):
        if self.opt.phase == 'train':
            self.part_meshs = data['part_mesh']
            self.label_path_vss = data['label_path_vs']
            self.label_path_norms = data['label_path_norm']
            self.edge_features = data['edge_feature']
            self.model_id = data['model_id']
            self.view_id = data['view_id']
        else:
            self.part_meshs = [data[0]]
            self.edge_features = [data[1]]
            self.save_path = data[2]

    def forward(self, i):
        pass

    def backward(self):
        pass

    def optimize_parameters(self, i):
        self.optimizer.zero_grad()
        self.forward(i)
        # self.backward()
        self.optimizer.step()

    def load_network(self, which_epoch):
        """load model from disk"""
        save_filename = '%s_net.pth' % which_epoch
        load_path = join(self.save_dir, save_filename)
        net = self.net
        print('loading for net ...', load_path)
        net.load_state_dict(torch.load(load_path, map_location=self.device))

    def save_network(self, which_epoch):
        """save model to disk"""
        save_filename = '%s_net.pth' % (which_epoch)
        save_path = join(self.save_dir, save_filename)
        torch.save(self.net.state_dict(), save_path)

    def update_learning_rate(self):
        """update learning rate (called once every epoch)"""
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def test(self):
        """tests model
        returns: number correct and total number
        """
        with torch.no_grad():
            for B_num, part_mesh in enumerate(self.part_meshs):
                edge_feature = self.edge_features[B_num].unsqueeze(0).to('cuda')
                from timeit import default_timer as timer
                start = timer()
                for part_i, est_verts in enumerate(self.net(edge_feature, part_mesh)):
                    part_mesh.update_verts(est_verts[0], part_i)
                    end = timer()
                    # print(end-start)
                    part_mesh.export(self.save_path)
                    mesh = trimesh.load(self.save_path)
        return mesh
