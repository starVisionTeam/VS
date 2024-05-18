import argparse
import os
from Mr.util import util
import torch

MANIFOLD_DIR = r'/home/amax/Manifold-master/build'  # path to manifold software (https://github.com/hjwdzh/Manifold)


# 0154   ### 0175  0110  0495 0062 0395
class BaseOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        # data params
        # self.parser.add_argument('--datarood', nargs='+', type=str, default='/media/amax/4C76448F76447C28/LYH/mesh_result', help='path to meshes (should have subfolders train, up_sample, test)')
        print("-0--------------------------")
        self.parser.add_argument('--dataset_mode', choices={"classification", "segmentation"}, default='reconstruction')
        self.parser.add_argument('--ninput_edges', type=int, default=750,
                                 help='# of input edges (will include dummy edges)')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                                 help='Maximum number of samples per epoch')
        self.parser.add_argument('--datarood', nargs='+', type=str,
                                 default='/media/gpu/备份/LYH/mesh_result',
                                 help='path to meshes (should have subfolders train, up_sample, test)')
        # network params
        self.parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        self.parser.add_argument('--arch', type=str, default='mconvnet',
                                 help='selects network to use')  # todo add choices
        self.parser.add_argument('--res_blocks', type=int, default=3, help='# of res blocks')
        self.parser.add_argument('--fc_n', type=int, default=100, help='# between fc and nclasses')  # todo make generic
        # self.parser.add_argument('--convs', nargs='+', default=[1024, 512, 256, 128], type=int, help='conv filters')
        # self.parser.add_argument('--convs', nargs='+', default=[32, 64, 64, 128, 64, 64, 32, 16],
        #                          type=int, help='conv filters')
        # self.parser.add_argument('--convs', nargs='+', default=[4096, 2048, 1024, 512],
        #                          type=int, help='conv filters')
        self.parser.add_argument('--convs', nargs='+', default=[1024, 1024, 1024, 2048, 2048, 1024, 512, 256, 128],
                                 type=int, help='conv filters')
        # self.parser.add_argument('--convs', nargs='+', default=[1024,4096, 2048, 1024, 512, 256, 128],
        #                          type=int, help='conv filters')
        # self.parser.add_argument('--convs', nargs='+', default=[512, 512, 1024, 1024, 1280, 2560, 1280, 640, 320, 160],
        #                          type=int, help='conv filters')
        # self.parser.add_argument('--convs', nargs='+', default=[512, 1024, 1024, 1024, 512, 256, 128],
        #                          type=int, help='conv filters')
        # self.parser.add_argument('--convs', nargs='+', default=[256, 32, 62, 64, 128, 64, 64, 32, 16],
        #                          type=int, help='conv filters')
        self.parser.add_argument('--pools', nargs='+', default=[0.0, 0.0, 0.0, 0.0], type=float, help='pooling res')
        self.parser.add_argument('--norm', type=str, default='batch',
                                 help='instance normalization or batch normalization or group normalization')
        self.parser.add_argument('--num_groups', type=int, default=16, help='# of groups for groupnorm')
        self.parser.add_argument('--init_type', type=str, default='normal',
                                 help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--init_gain', type=float, default=0.02,
                                 help='scaling factor for normal, xavier and orthogonal.')
        # general params
        self.parser.add_argument('--num_threads', default=1, type=int, help='# threads for loading data')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='debug',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--checkpoints_dir', type=str, default='../Mr/checkpoints',
                                 help='models are saved here')
        self.parser.add_argument('--serial_batches', action='store_true',
                                 help='if true, takes meshes in order, otherwise takes them randomly')
        self.parser.add_argument('--seed', type=int, help='if specified, uses seed')
        # visualization params
        self.parser.add_argument('--export_folder', type=str, default='',
                                 help='exports intermediate collapses to this folder')
        #
        self.parser.add_argument('--leaky_relu', type=float, metavar='1eN', default=0.01, help='slope for leaky relu')
        self.parser.add_argument('--init_weights', type=float, default=0.002, help='initialize NN with this size')
        self.parser.add_argument('--overlap', type=int, default=0, help='overlap for bfs')
        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        # self.opt, unknown = self.parser.parse_known_args()
        self.opt, _ = self.parser.parse_known_args()
        self.opt.is_train = self.is_train  # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)
        # seed为空
        if self.opt.seed is not None:
            import numpy as np
            import random
            torch.manual_seed(self.opt.seed)
            np.random.seed(self.opt.seed)
            random.seed(self.opt.seed)

        if self.opt.export_folder:
            self.opt.export_folder = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.export_folder)
            util.mkdir(self.opt.export_folder)

        if self.is_train:
            print('------------ Options -------------')
            for k, v in sorted(args.items()):
                print('%s: %s' % (str(k), str(v)))
            print('-------------- End ----------------')

            # save to the disk
            expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
            util.mkdir(expr_dir)

            file_name = os.path.join(expr_dir, 'opt.txt')
            with open(file_name, 'wt') as opt_file:
                opt_file.write('------------ Options -------------\n')
                for k, v in sorted(args.items()):
                    opt_file.write('%s: %s\n' % (str(k), str(v)))
                opt_file.write('-------------- End ----------------\n')
        return self.opt
