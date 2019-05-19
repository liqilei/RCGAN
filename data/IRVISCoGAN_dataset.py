import os

import numpy as np
import scipy.misc as misc
import cv2

import torch
import torch.utils.data as data

from data import common


class IRVISCoGAN_dataset(data.Dataset):
    def name(self):
        return 'IRVISCoGAN_dataset'

    def __init__(self, opt):
        super(IRVISCoGAN_dataset, self).__init__()
        # self.args = args
        self.opt = opt

        self.train = (opt['phase'] == 'train')
        self.split = 'train' if self.train else 'test'

        self.repeat = 1
        # read image list from lmdb or image files
        self.VIS_env, self.paths_VIS = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_VI'])
        self.IR_env, self.paths_IR = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_IR'])
        self.FUS_env, self.paths_PF = common.get_image_paths(self.opt['data_type'], self.opt['dataroot_PF'])

    def __getitem__(self, idx):
        if self.opt['phase'] == 'train' or self.opt['phase'] == 'val':
            ir, vis, fus, ir_path, vis_path, fus_path, = self._load_file(idx)
            ir_tensor, vis_tensor, pf_tensor = common.np2Tensor([ir, vis, fus], self.opt['rgb_range'])
            ir_pad_tensor, vis_pad_tensor, fus_pad_tensor = common._padding_board([ir_tensor, vis_tensor, pf_tensor])
            cat_tensor = torch.cat([ir_pad_tensor, vis_pad_tensor], 0)
            return {'IR': ir_tensor, 'VIS': vis_tensor, 'CAT': cat_tensor, 'PreFUS': pf_tensor, 'VIS_path': vis_path}

        else:
            ir, vis, ir_path, vis_path, = self._load_file(idx)
            ir_tensor, vis_tensor = common.np2Tensor([ir, vis], self.opt['rgb_range'])
            ir_pad_tensor, vis_pad_tensor = common._padding_board([ir_tensor, vis_tensor])
            cat_tensor = torch.cat([ir_pad_tensor, vis_pad_tensor], 0)
            return {'IR': ir_tensor, 'VIS': vis_tensor, 'CAT': cat_tensor, 'VIS_path': vis_path}

    def __len__(self):
        if self.train:
            return len(self.paths_VIS) * self.repeat
        else:
            return len(self.paths_IR)

    def _get_index(self, idx):
        if self.train:
            return idx % len(self.paths_VIS)
        else:
            return idx

    def _load_file(self, idx):
        idx = self._get_index(idx)
        ir_path = self.paths_IR[idx]
        vis_path = self.paths_VIS[idx]
        ir = common.read_img(self.IR_env, ir_path, self.opt['data_type'])
        vis = common.read_img(self.VIS_env, vis_path, self.opt['data_type'])

        if self.opt['phase'] == 'train' or self.opt['phase'] == 'val':
            fus_path = self.paths_PF[idx]
            fus = common.read_img(self.VIS_env, fus_path, self.opt['data_type'])
            return ir, vis, fus, ir_path, vis_path, fus_path
        else:
            return ir, vis, ir_path, vis_path

    def _get_patch(self, ir, vis):
        Label_Size = self.opt['Label_Size']
        if self.train:
            ir, vis = common.get_patch(
                ir, vis, Label_Size, self.scale)
            ir, vis = common.augment([ir, vis])
            ir = common.add_noise(ir, self.opt['noise'])

        return ir, vis


if __name__ == '__main__':
    a = IRVISCoGAN_dataset()
