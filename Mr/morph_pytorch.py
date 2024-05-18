import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from Mr.models.networks import Index
import trimesh
import random

ORIG_WIDTH = 0
ORIG_HEIGHT = 0
TRAIN_EPOCHS = 1000
im_sz = 128
mp_sz = 96
warp_scale = 0.05
mult_scale = 0.4
add_scale = 0.4
add_first = False
np.random.seed(1998)
torch.manual_seed(1998)
torch.cuda.manual_seed(1998)
random.seed(0)


def shift(f_shift, b_shift, vs, faces, mask):
    preds_f = f_shift
    preds_b = b_shift
    f_trans = -preds_f[:, :, :, 2:4] * 2 * 0.05
    b_trans = -preds_b[:, :, :, 2:4] * 2 * 0.05
    f_trans = f_trans.permute(0, 3, 1, 2)
    b_trans = b_trans.permute(0, 3, 1, 2)
    smpl_vs, smpl_faces = vs, faces
    smpl_vs[:, 1] *= -1
    smpl_vs = smpl_vs[None, :, :].permute(0, 2, 1)
    smpl_xy = smpl_vs[:, :2, :]
    f_xy_trans = Index(f_trans, smpl_xy)
    b_xy_trans = Index(b_trans, smpl_xy)
    mask_b = mask
    for id, back in enumerate(mask_b):
        if back == True:
            f_xy_trans[0, :, id] = b_xy_trans[0, :, id]

    smpl_vs[:, 0:1, :] = smpl_xy[:, 0:1, :] + f_xy_trans[:, 0:1, :]
    smpl_vs[:, 1:2, :] = smpl_xy[:, 1:2, :] + f_xy_trans[:, 1:2, :]
    smpl_vs = smpl_vs.permute(0, 2, 1).squeeze().cpu().numpy()
    smpl_vs[:, 1] *= -1
    recon_obj = trimesh.Trimesh(smpl_vs, faces.cpu().numpy(), process=False, maintains_order=True)
    return recon_obj


def warp(origins, targets, preds_org, preds_trg):
    if add_first:
        res_targets = dense_image_warp((origins + preds_org[:, :, :, 3:6] * 2 * add_scale) * torch.maximum(0.1,
                                                                                                           1 + preds_org[
                                                                                                               :, :, :,
                                                                                                               0:3] * mult_scale),
                                       preds_org[:, :, :, 6:8] * im_sz * warp_scale)
        res_origins = dense_image_warp((targets + preds_trg[:, :, :, 3:6] * 2 * add_scale) * torch.maximum(0.1,
                                                                                                           1 + preds_trg[
                                                                                                               :, :, :,
                                                                                                               0:3] * mult_scale),
                                       preds_trg[:, :, :, 6:8] * im_sz * warp_scale)
    else:
        res_targets = dense_image_warp(origins, preds_org[:, :, :, 0:2] * im_sz * warp_scale)
        res_origins = dense_image_warp(targets, preds_trg[:, :, :, 0:2] * im_sz * warp_scale)
    return res_targets, res_origins


def create_grid(scale):
    grid = np.mgrid[0:scale, 0:scale] / (scale - 1) * 2 - 1
    grid = np.swapaxes(grid, 0, 2)
    grid = np.expand_dims(grid, axis=0)
    grid = torch.from_numpy(grid)
    grid = grid.permute(0, 3, 1, 2)
    return grid


def dense_image_warp(image, flow):
    scale = 128
    grid = np.mgrid[0:scale, 0:scale]
    grid = np.expand_dims(grid, axis=0)
    grid = torch.from_numpy(grid)
    grid = grid.permute(0, 3, 2, 1)
    query_points_on_grid = grid.cuda() - flow
    query_points_on_grid = query_points_on_grid.float()
    query_points_on_grid = query_points_on_grid / 127.5 * 2 - 1
    warped_image = F.grid_sample(image.cuda(), query_points_on_grid, align_corners=False)
    return warped_image


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(8, 64, kernel_size=15)
        self.act1 = nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=13)
        self.act2 = nn.LeakyReLU(negative_slope=0.2)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=11)
        self.act3 = nn.LeakyReLU(negative_slope=0.2)
        self.conv4 = nn.Conv2d(128, 64, kernel_size=7)
        self.act4 = nn.LeakyReLU(negative_slope=0.2)
        self.conv5 = nn.Conv2d(64, 4, kernel_size=3)
        self.act5 = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, maps):
        x = F.interpolate(maps, size=(mp_sz, mp_sz), mode='bilinear', align_corners=True)
        x = self.conv1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.act3(x)
        x = self.conv4(x)
        x = self.act4(x)
        x = self.conv5(x)
        x = self.act5(x)
        return x


def produce_warp_maps(origins, targets, save_path=None):
    maps = create_grid(im_sz)
    maps = torch.cat((maps.cuda(), origins * 0.1, targets * 0.1), dim=1).float()
    maps = maps.clone().detach()
    origins = origins.clone().detach()
    targets = targets.clone().detach()
    model = MyModel().cuda()
    loss_object = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    train_loss = torch.tensor(0.0, requires_grad=False)
    pbar = tqdm(total=TRAIN_EPOCHS, position=0, leave=True)
    for epoch in range(TRAIN_EPOCHS):
        optimizer.zero_grad()
        preds = model(maps)
        preds = F.interpolate(preds, size=(im_sz, im_sz), mode='bilinear', align_corners=True)
        preds = preds.permute(0, 3, 2, 1)
        res_targets_, res_origins_ = warp(origins, targets, preds[:, :, :, :2], preds[:, :, :, 2:])
        res_map = dense_image_warp(maps, preds[:, :, :, 0:2] * im_sz * warp_scale)
        res_map = dense_image_warp(res_map, preds[:, :, :, 2:4] * im_sz * warp_scale)
        loss = (loss_object(maps, res_map) * 1.0 + loss_object(res_targets_, targets) * 0.3 + loss_object(res_origins_,
                                                                                                          origins) * 0.3)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        pbar.set_description(f"Epoch {epoch},Train Loss:{loss.item()}")
        pbar.update(1)
    pbar.close()
    with torch.no_grad():
        preds = model(maps)
        preds = F.interpolate(preds, size=(im_sz, im_sz), mode='bilinear', align_corners=True)
    return preds.permute(0, 3, 2, 1).detach()


def morph_image(source_image_path, target_image_path):
    origins = source_image_path
    targets = target_image_path
    preds_shift = produce_warp_maps(origins, targets)
    return preds_shift


if __name__ == "__main__":
    source = "/media/gpu/dataset_SSD/code/Cloth-shift/result-6/0295_full_body_norm_B.png"
    target = "/media/gpu/dataset_SSD/code/Cloth-shift/result-6/0295_rgb_norm_B.png"
    # save_path = "/media/gpu/dataset_SSD/code/Cloth-shift/result-6/0295_rgb_norm_B.npy"

    morph_image(source, target)
