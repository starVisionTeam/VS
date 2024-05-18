import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from Mr.models.layers.mesh_conv import MeshConv
import torch.nn.functional as F
from Mr.models.layers.mesh_pool import MeshPool
from Mr.models.layers.mesh_unpool import MeshUnpool

import torch
import torch.nn as nn
from torch.nn import init
from torch import optim
from Mr.models.layers.mesh_conv import MeshConv
import torch.nn.functional as F
from Mr.models.layers.mesh_pool import MeshPool
from Mr.models.layers.mesh_unpool import MeshUnpool
from typing import List


###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance', num_groups=1):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'group':
        norm_layer = functools.partial(nn.GroupNorm, affine=True, num_groups=num_groups)
    elif norm_type == 'none':
        norm_layer = NoNorm
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_norm_args(norm_layer, nfeats_list):
    if hasattr(norm_layer, '__name__') and norm_layer.__name__ == 'NoNorm':
        norm_args = [{'fake': True} for f in nfeats_list]
    elif norm_layer.func.__name__ == 'GroupNorm':
        norm_args = [{'num_channels': f} for f in nfeats_list]
    elif norm_layer.func.__name__ == 'BatchNorm':
        norm_args = [{'num_features': f} for f in nfeats_list]
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_layer.func.__name__)
    return norm_args


class NoNorm(nn.Module):  # todo with abstractclass and pass
    def __init__(self, fake=True):
        self.fake = fake
        super(NoNorm, self).__init__()

    def forward(self, x):
        return x

    def __call__(self, x):
        return self.forward(x)


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, threshold=0.0001, patience=2)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type, init_gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def define_classifier(input_nc, ncf, ninput_edges, nclasses, opt, gpu_ids, arch, init_type, init_gain):
    net = None
    norm_layer = get_norm_layer(norm_type=opt.norm, num_groups=opt.num_groups)

    if arch == 'mconvnet':
        net = MeshConvNet(norm_layer, input_nc, ncf, nclasses, ninput_edges, opt.pool_res, opt.fc_n,
                          opt.resblocks)
    elif arch == 'meshunet':  ############################
        down_convs = [input_nc] + ncf
        up_convs = ncf[::-1] + [nclasses]
        pool_res = [ninput_edges] + opt.pool_res
        net = MeshEncoderDecoder(pool_res, down_convs, up_convs, blocks=opt.resblocks,
                                 transfer_data=True)
    else:
        raise NotImplementedError('Encoder model name [%s] is not recognized' % arch)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_loss(opt):
    if opt.dataset_mode == 'classification':
        loss = torch.nn.CrossEntropyLoss()
    elif opt.dataset_mode == 'reconstruction':
        loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
    return loss


def populate_e(meshes, verts=None):
    mesh = meshes[0]
    if verts is None:
        # verts = torch.rand(len(meshes), mesh.vs.shape[0], 3).to(mesh.vs.device)
        verts = torch.zeros(len(meshes), mesh.vs.shape[0], 3).to(torch.device('cpu'))
    x = verts[:, mesh.edges, :]
    return x.view(len(meshes), mesh.edges_count, -1).permute(0, 2, 1).type(torch.float32)


##############################################################################
# Classes For Classification / Segmentation Networks
##############################################################################

class MeshConvNet(nn.Module):
    """Network for learning a global shape descriptor (classification)
    """

    def __init__(self, norm_layer, nf0, conv_res, nclasses, input_res, pool_res, fc_n,
                 nresblocks=3):
        super(MeshConvNet, self).__init__()
        self.k = [nf0] + conv_res
        self.res = [input_res] + pool_res
        norm_args = get_norm_args(norm_layer, self.k[1:])

        for i, ki in enumerate(self.k[:-1]):
            setattr(self, 'conv{}'.format(i), MResConv(ki, self.k[i + 1], nresblocks))
            setattr(self, 'norm{}'.format(i), norm_layer(**norm_args[i]))
            setattr(self, 'pool{}'.format(i), MeshPool(self.res[i + 1]))

        self.gp = torch.nn.AvgPool1d(self.res[-1])
        # self.gp = torch.nn.MaxPool1d(self.res[-1])
        self.fc1 = nn.Linear(self.k[-1], fc_n)
        self.fc2 = nn.Linear(fc_n, nclasses)

    def forward(self, x, mesh):

        for i in range(len(self.k) - 1):
            x = getattr(self, 'conv{}'.format(i))(x, mesh)
            x = F.relu(getattr(self, 'norm{}'.format(i))(x))
            x = getattr(self, 'pool{}'.format(i))(x, mesh)

        x = self.gp(x)
        x = x.view(-1, self.k[-1])

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class MResConv(nn.Module):
    def __init__(self, in_channels, out_channels, skips=1):
        super(MResConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skips = skips
        self.conv0 = MeshConv(self.in_channels, self.out_channels, bias=False)
        for i in range(self.skips):
            setattr(self, 'bn{}'.format(i + 1), nn.BatchNorm2d(self.out_channels))
            setattr(self, 'conv{}'.format(i + 1),
                    MeshConv(self.out_channels, self.out_channels, bias=False))

    def forward(self, x, mesh):
        x = self.conv0(x, mesh)
        x1 = x
        for i in range(self.skips):
            x = getattr(self, 'bn{}'.format(i + 1))(F.relu(x))
            x = getattr(self, 'conv{}'.format(i + 1))(x, mesh)
        x += x1
        x = F.relu(x)
        return x


def reset_params(model):  # todo replace with my init
    for i, m in enumerate(model.modules()):
        weight_init(m)


def weight_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        # nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)


###########################
def init_net(device, opts):
    # initialize network, weights, and random input tensor
    # init_verts = mesh.vs.clone().detach()
    net = PartNet(init_part_mesh=None, convs=opts.convs,
                  pool=opts.pools, res_blocks=opts.res_blocks,
                  init_verts=None, transfer_data=False,
                  leaky=opts.leaky_relu, init_weights_size=opts.init_weights).to(device)
    # net = torch.nn.DataParallel(net, opts.gpu_ids)

    # optimizer = optim.Adam(net.parameters(), lr=opts.lr)
    # scheduler = get_scheduler(opts.iterations, optimizer)
    # rand_verts = populate_e([mesh])
    return net


class PriorNet(nn.Module):
    """
    network for
    """

    def __init__(self, n_edges, in_ch=562, convs=[32, 64], pool=[], res_blocks=0,
                 init_verts=None, transfer_data=False, leaky=0, init_weights_size=0.002):
        super(PriorNet, self).__init__()
        # check that the number of pools and convs match such that there is a pool between each conv
        # down_convs = [in_ch] + convs
        # up_convs = convs[::-1] + [in_ch]
        # down_convs = [in_ch] + convs[:5]
        # up_convs = convs[4:] + [6]
        down_convs = [in_ch] + convs[:7]
        up_convs = convs[6:] + [6]
        # down_convs = [in_ch] + convs[:2]
        # up_convs = convs[1:] + [6]
        pool_res = [n_edges] + pool
        self.encoder_decoder = MeshEncoderDecoder(pools=pool_res, down_convs=down_convs,
                                                  up_convs=up_convs, blocks=res_blocks,
                                                  transfer_data=transfer_data, leaky=leaky)
        self.last_conv = MeshConv(6, 6)
        init_weights(self, 'kaiming', init_weights_size)
        eps = 1e-8
        self.last_conv.conv.weight.data.uniform_(-1 * eps, eps)
        self.last_conv.conv.bias.data.uniform_(-1 * eps, eps)
        self.init_verts = init_verts
        # 1看输入是怎么处理的 2怎么连接的manifold 3损失是怎么引入的

    def forward(self, x, meshes):
        meshes_new = [i.deep_copy() for i in meshes]
        # x, _ = self.encoder_decoder(x, meshes[0])
        x, _ = self.encoder_decoder(x, meshes_new)
        x = x.squeeze(-1)
        x = self.last_conv(x, meshes_new).squeeze(-1)
        # x = self.last_conv(x, meshes[0]).squeeze(-1)
        est_verts = build_v(x.unsqueeze(0), meshes)
        # assert not torch.isnan(est_verts).any()
        return est_verts.float() + self.init_verts.expand_as(est_verts).to(est_verts.device)
        # return est_verts.float() + self.init_verts.to(est_verts.device)


class PartNet(PriorNet):
    def __init__(self, init_part_mesh, in_ch=562, convs=[32, 64], pool=[], res_blocks=0,
                 init_verts=None, transfer_data=False, leaky=0,
                 init_weights_size=0.002):
        temp = torch.linspace(int(len(convs) / 2) + 1, 1, int(len(convs) / 2) + 1).long().tolist()
        super().__init__(temp[0], in_ch=in_ch, convs=convs, pool=temp[1:], res_blocks=res_blocks,
                         init_verts=init_verts, transfer_data=transfer_data, leaky=leaky,
                         init_weights_size=init_weights_size)
        self.mesh_pools = []
        self.mesh_unpools = []
        self.factor_pools = pool
        for i in self.modules():
            if isinstance(i, MeshPool):
                self.mesh_pools.append(i)
            if isinstance(i, MeshUnpool):
                self.mesh_unpools.append(i)
        self.mesh_pools = sorted(self.mesh_pools, key=lambda x: x._MeshPool__out_target, reverse=True)
        # self.mesh_pools = sorted(self.mesh_pools, key=lambda x: x.__out_target, reverse=True)
        self.mesh_unpools = sorted(self.mesh_unpools, key=lambda x: x.unroll_target, reverse=False)
        self.init_part_verts = None
        #     nn.ParameterList([torch.nn.Parameter(i) for i in init_part_mesh.init_verts])
        # for i in self.init_part_verts:
        #     i.requires_grad = False

    def __set_pools(self, n_edges: int, new_pools: List[int]):
        for i, l in enumerate(self.mesh_pools):
            l._MeshPool__out_target = new_pools[i]
        new_pools = [n_edges] + new_pools
        new_pools = new_pools[:-1]
        new_pools.reverse()
        for i, l in enumerate(self.mesh_unpools):
            l.unroll_target = new_pools[i]

    def forward(self, x, partmesh):
        """
        forward PartNet
        :param x: BXfXn_edges
        :param partmesh:
        :return:
        """

        for i, p in enumerate(partmesh):
            # self.init_part_verts = nn.ParameterList([torch.nn.Parameter(partmesh.init_verts[i])])
            self.init_part_verts = partmesh.init_verts[i]
            for vert in self.init_part_verts:
                vert.requires_grad = False
            n_edges = p.edges_count
            self.init_verts = self.init_part_verts
            temp_pools = [int(n_edges - i) for i in self.make3(PartNet.array_times(n_edges, self.factor_pools))]
            self.__set_pools(n_edges, temp_pools)
            relevant_edges = x[:, :, partmesh.sub_mesh_edge_index[i]]
            results = super().forward(relevant_edges, [p])
            # results = super().forward(relevant_edges, p)
            yield results

    @staticmethod
    def array_times(num: int, iterable):
        return [i * num for i in iterable]

    @staticmethod
    def make3(array):
        diff = [i % 3 for i in array]
        return [array[i] - diff[i] for i in range(len(array))]


class MeshEncoderDecoder(nn.Module):
    """Network for fully-convolutional tasks (segmentation)
    """

    def __init__(self, pools, down_convs, up_convs, blocks=0, transfer_data=True, leaky=0):
        super(MeshEncoderDecoder, self).__init__()
        self.transfer_data = transfer_data
        self.encoder = MeshEncoder(pools, down_convs, blocks=blocks, leaky=leaky)
        unrolls = pools[:-1].copy()
        unrolls.reverse()
        add_convs = down_convs[::-1]
        self.decoder = MeshDecoder(unrolls, up_convs, add_convs, blocks=blocks, transfer_data=transfer_data,
                                   leaky=leaky)
        self.bn = nn.InstanceNorm2d(up_convs[-1])

    def forward(self, x, meshes):
        fe, before_pool = self.encoder((x, meshes))
        fe = self.decoder((fe, meshes), before_pool)
        fe = self.bn(fe.unsqueeze(-1))
        return fe, None


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels, blocks=0, pool=0, leaky=0):
        super(DownConv, self).__init__()
        self.leaky = leaky
        self.bn = []
        self.pool = None
        self.conv1 = ConvBlock(in_channels, out_channels)
        # self.conv2 = []
        # for _ in range(blocks):
        #     self.conv2.append(ConvBlock(out_channels, out_channels))
        #     self.conv2 = nn.ModuleList(self.conv2)
        for _ in range(blocks + 1):
            self.bn.append(nn.InstanceNorm2d(out_channels))
            self.bn = nn.ModuleList(self.bn)
        # if pool:
        #     self.pool = MeshPool(pool)

    def forward(self, x):
        fe, meshes = x[0], x[1]
        x1 = self.conv1(fe, meshes)
        x1 = F.leaky_relu(x1, self.leaky)
        if self.bn:
            x1 = self.bn[0](x1)
        x1 = x1.squeeze(3)
        before_pool = None
        # if self.pool:
        #     before_pool = x1
        #     x1 = self.pool(x1, meshes)
        return x1, before_pool


class ConvBlock(nn.Module):
    def __init__(self, in_feat, out_feat, k=1):
        super(ConvBlock, self).__init__()
        self.lst = [MeshConv(in_feat, out_feat)]
        for i in range(k - 1):
            self.lst.append(MeshConv(out_feat, out_feat))
        self.lst = nn.ModuleList(self.lst)

    def forward(self, input, meshes):
        for c in self.lst:
            input = c(input, meshes)
        return input


class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, blocks=0, unroll=0, residual=True,  ###########加了add
                 batch_norm=True, transfer_data=True, leaky=0):
        super(UpConv, self).__init__()
        self.leaky = leaky
        self.residual = residual
        self.bn = []
        self.unroll = None
        self.transfer_data = transfer_data
        self.up_conv = ConvBlock(in_channels, out_channels)
        # if transfer_data:
        #     self.conv1 = ConvBlock(2*out_channels, out_channels)###################
        # else:
        #     self.conv1 = ConvBlock(out_channels, out_channels)
        # self.conv2 = []
        # for _ in range(blocks):
        #     self.conv2.append(ConvBlock(out_channels, out_channels))
        #     self.conv2 = nn.ModuleList(self.conv2)
        if batch_norm:
            for _ in range(blocks + 1):
                self.bn.append(nn.InstanceNorm2d(out_channels))
            self.bn = nn.ModuleList(self.bn)
        # if unroll:
        #     self.unroll = MeshUnpool(unroll)

    def forward(self, x, from_down=None):
        from_up, meshes = x
        x1 = self.up_conv(from_up, meshes).squeeze(3)
        # if self.unroll:
        #     x1 = self.unroll(x1, meshes)
        # if self.transfer_data:
        #     x1 = torch.cat((x1, from_down), 1)
        # x1 = self.conv1(x1, meshes)
        x1 = F.leaky_relu(x1, self.leaky)
        x1 = x1.unsqueeze(3)
        if self.bn:
            x1 = self.bn[0](x1)
        x1 = x1.squeeze(3)
        return x1


class MeshEncoder(nn.Module):
    def __init__(self, pools, convs, blocks=0, leaky=0):
        super(MeshEncoder, self).__init__()
        self.leaky = leaky
        self.convs = []
        for i in range(len(convs) - 1):
            if i + 1 < len(pools):
                pool = pools[i + 1]
            else:
                pool = 0
            if i == 0:
                self.convs.append(DownConv(convs[i], convs[i + 1], blocks=blocks, pool=pool, leaky=leaky))
            else:
                self.convs.append(DownConv(convs[i], convs[i + 1], blocks=blocks, pool=pool, leaky=leaky))
        self.convs = nn.ModuleList(self.convs)
        reset_params(self)

    def forward(self, x):
        fe, meshes = x
        # encoder_outs = fe[:, 0:512, :]
        encoder_outs = fe
        for i, conv in enumerate(self.convs):
            if i == 0:
                fe, before_pool = conv((fe, meshes))
            else:
                # fe, before_pool = conv((torch.cat([encoder_outs, fe], 1), meshes))
                fe, before_pool = conv((fe, meshes))
            # encoder_outs.append(before_pool)
        return fe, encoder_outs


class MeshDecoder(nn.Module):
    def __init__(self, unrolls, convs, add_convs, blocks=0, batch_norm=True, transfer_data=True, leaky=0):
        super(MeshDecoder, self).__init__()
        self.up_convs = []
        for i in range(len(convs) - 2):
            if i < len(unrolls):
                unroll = unrolls[i]
            else:
                unroll = 0
            self.up_convs.append(UpConv(convs[i] + 562, convs[i + 1], blocks=blocks, unroll=unroll,
                                        batch_norm=batch_norm, transfer_data=transfer_data,
                                        leaky=leaky))  ############加了add_convs[i+1]
        self.final_conv = UpConv(convs[-2] + 562, convs[-1], blocks=blocks, unroll=False,
                                 batch_norm=batch_norm, transfer_data=False, leaky=leaky)  ##############加了add_convs[-1]
        self.up_convs = nn.ModuleList(self.up_convs)
        reset_params(self)

    def forward(self, x, encoder_outs=None):
        fe, meshes = x
        for i, up_conv in enumerate(self.up_convs):
            before_pool = None
            # if encoder_outs is not None:
            #     before_pool = encoder_outs[-(i+2)]
            fe = up_conv((torch.cat([encoder_outs, fe], 1), meshes), before_pool)
            # fe = up_conv((fe, meshes), before_pool)
        fe = self.final_conv((torch.cat([encoder_outs, fe], 1), meshes))
        # fe = self.final_conv((fe, meshes))
        return fe

    def __call__(self, x, encoder_outs=None):
        return self.forward(x, encoder_outs)


def build_v(x, meshes):
    # mesh.edges[mesh.ve[2], mesh.vei[2]]
    mesh = meshes[0]  # b/c all meshes in batch are same
    x = x.reshape(len(meshes), 2, 3, -1)
    vs_to_sum = torch.zeros([len(meshes), len(mesh.vs_in), mesh.max_nvs, 3], dtype=x.dtype, device=x.device)
    x = x[:, mesh.vei, :, mesh.ve_in].transpose(0, 1)
    vs_to_sum[:, mesh.nvsi, mesh.nvsin, :] = x
    vs_sum = torch.sum(vs_to_sum, dim=2)
    nvs = mesh.nvs.to('cuda')
    vs = vs_sum / nvs[None, :, None]
    return vs


def orthogonal(points, calibrations, transforms=None):
    '''
    Compute the orthogonal projections of 3D points into the image plane by given projection matrix
    :param points: [B, 3, N] Tensor of 3D points
    :param calibrations: [B, 4, 4] Tensor of projection matrix
    :param transforms: [B, 2, 3] Tensor of image transform matrix
    :return: xyz: [B, 3, N] Tensor of xyz coordinates in the image plane
    '''
    rot = calibrations[:, :3, :3]
    trans = calibrations[:, :3, 3:4]
    pts = torch.baddbmm(trans, rot, points)  # [B, 3, N]
    if transforms is not None:
        scale = transforms[:2, :2]
        shift = transforms[:2, 2:3]
        pts[:, :2, :] = torch.baddbmm(shift, scale, pts[:, :2, :])
    return pts


def Index(feat, uv):
    '''

    :param feat: [B, C, H, W] image features
    :param uv: [B, 2, N] uv coordinates in the image plane, range [-1, 1]
    :return: [B, C, N] image features at the uv coordinates
    '''
    uv = uv.transpose(1, 2)  # [B, N, 2]
    uv = uv.unsqueeze(2)  # [B, N, 1, 2]
    # NOTE: for newer PyTorch, it seems that training results are degraded due to implementation diff in F.grid_sample
    # for old versions, simply remove the aligned_corners argument.
    samples = torch.nn.functional.grid_sample(feat, uv, align_corners=False)  # [B, C, N, 1]采样点是干嘛的
    return samples[:, :, :, 0]  # [B, C, N]


def rotate_points(pts, view_id):
    angle = (-2 * np.pi * view_id) / 360
    pts_rot = np.zeros_like(pts)
    pts_rot[:, 0] = pts[:, 0] * math.cos(angle) - pts[:, 2] * math.sin(angle)
    pts_rot[:, 1] = pts[:, 1]
    pts_rot[:, 2] = pts[:, 0] * math.sin(angle) + pts[:, 2] * math.cos(angle)

    return pts_rot.astype(np.float32)
