# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.


from pifuhd_ori.apps.recon import reconWrapper
import argparse


def feature_capture(image=None, norm_F=None, norm_B=None):
    ckpt_path = '../pifuhd_ori/pifuhd.pt'
    cmd = ['--load_netMR_checkpoint_path', ckpt_path, ]
    feature, feature_fine = reconWrapper(cmd, False, image, norm_F, norm_B)
    return feature, feature_fine


if __name__ == '__main__':
    feature_capture()
