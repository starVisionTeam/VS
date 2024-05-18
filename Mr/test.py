from Mr.options.test_options import TestOptions
from Mr.data import DataLoader
from Mr.models import create_model
from timeit import default_timer as timer
import os
import torch
from Mr.data.base_dataset import BaseDataset
from Mr.util.util import is_mesh_file, load_obj, manifold_upsample, get_num_parts, compute_normal
import numpy as np
from Mr.models.layers.mesh import Mesh, PartMesh, export
from Mr.models.networks import orthogonal, Index, populate_e, rotate_points
import trimesh
from PIL import Image
import torchvision.transforms as transforms
from Mr.visible import Visible

device = torch.device('cpu')


def data_preparation(path, normals_f, normals_b, feature_fine, feature_global, mask_b, save_path):
    mid_path = save_path
    mesh = manifold_upsample(path, mid_path, Mesh, num_faces=40000, res=20000, simplify=True)
    smpl_vs = np.array(mesh.vs)
    smpl_faces = np.array(mesh.faces)
    smpl_norm = compute_normal(smpl_vs, smpl_faces)
    mask_b = Visible(mesh.vs, mesh.faces)
    part_mesh = PartMesh(mesh, num_parts=get_num_parts(len(mesh.faces)), bfs_depth=0)
    projection_matrix = np.identity(4)
    projection_matrix[1, 1] = -1
    calib_former = torch.Tensor(projection_matrix).to(device).float()
    calib = calib_former[None, :, :]
    smpl_vs = mesh.vs
    smpl_vs = smpl_vs - mesh.translations
    smpl_vs = smpl_vs * mesh.scale
    if type(mesh.vs) is np.ndarray:
        vs = torch.from_numpy(smpl_vs).to(device).float()
    else:
        vs = smpl_vs.to(device).float()
    vs = vs[None, :, :].permute(0, 2, 1)
    xyz = orthogonal(vs, calib, transforms=None)
    xy = xyz[:, :2, :]  # B, 2, N
    vs_normals_f = Index(normals_f, xy)  # B, C, N
    vs_normals_f = vs_normals_f.permute(0, 2, 1)  # B, N, C
    vs_normals_b = Index(normals_b, xy)  # B, C, N
    vs_normals_b = vs_normals_b.permute(0, 2, 1)  # B, N, C
    vs_normals = torch.empty((1, vs.size(2), 3))
    for id, back in enumerate(mask_b):
        if back == True:
            vs_normals[0, id, :] = vs_normals_b[0, id, :]
        else:
            vs_normals[0, id, :] = vs_normals_f[0, id, :]
    vs_features_fine = Index(feature_fine, xy)  # B, C, N
    vs_features_fine = vs_features_fine.permute(0, 2, 1)  # B, N, C
    x_fine = vs_features_fine[:, mesh.edges, :]
    edge_feature_fine = x_fine.view(1, mesh.edges_count, -1).permute(0, 2, 1)  # B, C, N
    vs_features_global = Index(feature_global, xy)  # B, C, N
    vs_features_global = vs_features_global.permute(0, 2, 1)  # B, N, C
    x_global = vs_features_global[:, mesh.edges, :]
    edge_feature_global = x_global.view(1, mesh.edges_count, -1).permute(0, 2, 1)  # B, C, N
    smpl_norm = torch.tensor(smpl_norm[None]).to(device).float()
    vs_features = vs_normals
    vs_features = torch.cat([vs_features, smpl_norm], dim=2)
    x = vs_features[:, mesh.edges, :]
    edge_feature = x.view(1, mesh.edges_count, -1).permute(0, 2, 1)  # B, C, N
    rand_verts = populate_e([mesh])
    feature = torch.cat([edge_feature, edge_feature_global, edge_feature_fine, rand_verts], dim=1)
    edge_feature = feature.squeeze(0)

    return part_mesh, edge_feature, save_path


def run_test(path, normals_f, normals_b, feature_fine, feature_global, mask_b, save_path):
    print('Running GCN')
    opt = TestOptions().parse()
    model = create_model(opt)
    data = data_preparation(path, normals_f.cpu(), normals_b.cpu(), feature_fine.cpu(), feature_global.cpu(),
                            mask_b.cpu(), save_path)
    model.set_input(data)
    mesh = model.test()
    return mesh


if __name__ == '__main__':
    tic = timer()
    run_test()
    toc = timer()
    print(toc - tic)
