import os
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from utils import util
from utils.torchsummary import summary as tc_summary
from utils.util import gradient
from .base_solver import BaseSolver
from .modules import loss
from .networks import create_model
from .networks import init_weights


class RCGANModel(BaseSolver):
    def __init__(self, opt):
        super(RCGANModel, self).__init__(opt)
        self.train_opt = opt['train']
        self.img_vis = self.Tensor()
        self.img_ir = self.Tensor()
        self.img_cat = self.Tensor()
        self.img_pf = self.Tensor()
        self.results = {'train_G_loss1': [],
                        'train_G_loss2': [],
                        'train_D_loss1': [],
                        'train_D_loss2': [],
                        'val_G_loss': [],
                        'psnr': [],
                        'ssim': []
                        }
        self.best_prec = np.inf
        self.model = create_model(opt)

        if self.is_train:
            self.model['netG'].train()
            self.model['netD'].train()

            # Content loss 1 : infared image loss
            pix_loss_type = self.train_opt['pixel_criterion']
            if pix_loss_type == 'l1':
                self.criterion_pix = nn.L1Loss()
            elif pix_loss_type == 'l2':
                self.criterion_pix = nn.MSELoss()
            else:
                raise NotImplementedError('[ERROR] Loss type [%s] is not implemented!' % pix_loss_type)

            if self.use_gpu:
                self.criterion_pix = self.criterion_pix.cuda()
            self.criterion_pix_weight = self.train_opt['pixel_weight']

            # Content loss 2 : visible image gradient information loss
            feat_loss_type = self.train_opt['feature_criterion']
            if feat_loss_type == 'l1':
                self.criterion_feat = nn.L1Loss()
            elif feat_loss_type == 'l2':
                self.criterion_feat = nn.MSELoss()
            else:
                raise NotImplementedError('[ERROR] Loss type [%s] is not implemented!' % pix_loss_type)

            if self.use_gpu:
                self.criterion_feat = self.criterion_feat.cuda()
            self.criterion_feat_weight = self.train_opt['feature_weight']

            # Set GAN Loss
            GAN_loss_type = self.train_opt['gan_type']
            self.criterion_gan = loss.GANLoss(GAN_loss_type)
            self.criterion_gan_weight = self.train_opt['gan_weight']
            self.criterion_lambda = self.train_opt['lambda']
            self.criterion_epsilon = self.train_opt['epsilon']

            # Weight decay for G
            weight_decay_G = self.train_opt['weight_decay_G'] if self.train_opt['weight_decay_G'] else 0

            # Weight decay for D
            weight_decay_D = self.train_opt['weight_decay_D'] if self.train_opt['weight_decay_D'] else 0

            # Set G optimizer
            optim_type = self.train_opt['type'].upper()
            if optim_type == "SGD":
                self.optim_G = optim.SGD(self.model['netG'].parameters(), lr=self.train_opt['lr_G'],
                                         momentum=self.train_opt['beta1_G'], weight_decay=weight_decay_G)
            elif optim_type == "ADAM":
                self.optim_G = optim.Adam(self.model['netG'].parameters(), lr=self.train_opt['lr_G'],
                                          weight_decay=weight_decay_G)
            else:
                raise NotImplementedError('[ERROR] Loss type [%s] is not implemented!' % optim_type)

            # Set D optimizer
            if optim_type == "SGD":
                self.optim_D = optim.SGD(self.model['netD'].parameters(), lr=self.train_opt['lr_D'],
                                         momentum=self.train_opt['beta1_G'], weight_decay=weight_decay_D)
            elif optim_type == "ADAM":
                self.optim_D = optim.Adam(self.model['netD'].parameters(), lr=self.train_opt['lr_D'],
                                          weight_decay=weight_decay_D)
            else:
                raise NotImplementedError('[ERROR] Loss type [%s] is not implemented!' % optim_type)

            print('[Model Initialized]')

    def net_init(self, init_type='truncated_normal'):
        for m in self.model:
            init_weights(self.model[m], init_type)

    def feed_data(self, batch):
        img_vis, img_ir, img_cat = batch['VIS'], batch['IR'], batch['CAT']
        self.img_vis.resize_(img_vis.size()).copy_(img_vis)
        self.img_ir.resize_(img_ir.size()).copy_(img_ir)
        self.img_cat.resize_(img_cat.size()).copy_(img_cat)
        if self.opt['is_train']:
            img_pf = batch['PreFUS']
            self.img_pf.resize_(img_pf.size()).copy_(img_pf)

    def summary(self, input_g_size, input_d_size):
        print('========================= Model Summary ========================')

        print('========================= Generator Summary ========================')
        print(self.model['netG'])
        print('================================================================')
        print('Input Size: %s' % str(input_g_size))
        tc_summary(self.model['netG'], input_g_size)

        print('========================= Discriminator Summary ========================')
        print(self.model['netD'])
        print('================================================================')
        print('Input Size: %s' % str(input_d_size))
        tc_summary(self.model['netD'], [input_d_size, input_d_size])
        print('================================================================')

    def train_step(self):
        self.model['netG'].train()
        self.model['netD'].train()

        # optim D
        train_D_loss1 = 0.
        train_D_loss2 = 0.
        for d_index in range(self.train_opt['D_step']):
            self.optim_D.zero_grad()
            loss_d_total = 0.
            self.img_fuse1, self.img_fuse2 = self.model['netG'](self.img_cat)

            score_d_real_1, score_d_real_2 = self.model['netD']([self.img_vis, self.img_ir])
            score_d_fake_1, score_d_fake_2 = self.model['netD']([self.img_fuse1.detach(), self.img_fuse2.detach()])

            # d_loss 1
            loss_d_gan_real_1 = self.criterion_gan(score_d_real_1 - torch.mean(score_d_fake_1), True)
            loss_d_gan_fake_1 = self.criterion_gan(score_d_fake_1 - torch.mean(score_d_real_1), False)

            loss_d_1 = (loss_d_gan_real_1 + loss_d_gan_fake_1) / 2

            # d_loss 2
            loss_d_gan_real_2 = self.criterion_gan(score_d_real_2 - torch.mean(score_d_fake_2), True)
            loss_d_gan_fake_2 = self.criterion_gan(score_d_fake_2 - torch.mean(score_d_real_2), False)

            loss_d_2 = (loss_d_gan_real_2 + loss_d_gan_fake_2) / 2

            loss_d_total = loss_d_1 + loss_d_2

            loss_d_total.backward()
            self.optim_D.step()

            train_D_loss1 += loss_d_1.item()
            train_D_loss2 += loss_d_2.item()

        train_G_loss1 = 0.
        train_G_loss2 = 0.
        for g_index in range(self.train_opt['G_step']):
            # optim G
            self.optim_G.zero_grad()
            self.img_fuse1, self.img_fuse2 = self.model['netG'](self.img_cat)
            # g_content loss
            loss_g_content_1 = self.criterion_lambda * self.criterion_pix(self.img_fuse1, self.img_pf) + \
                               self.criterion_lambda * self.criterion_pix(self.img_fuse1, self.img_ir)
            loss_g_content_2 = self.criterion_lambda * self.criterion_pix(self.img_fuse2, self.img_pf) + \
                               self.criterion_lambda * self.criterion_epsilon * (
                                   self.criterion_feat(gradient(self.img_fuse2), gradient(self.img_vis)))
            loss_g_1 = loss_g_content_1
            loss_g_2 = loss_g_content_2

            # g_gan loss
            score_g_fake_1, score_g_fake_2 = self.model['netD']([self.img_fuse1, self.img_fuse2])
            score_g_real_1, score_g_real_2 = self.model['netD']([self.img_vis, self.img_ir])

            # g_gan_loss 1
            loss_g_gan_real_1 = self.criterion_gan(score_g_fake_1 - torch.mean(score_g_real_1), True)
            loss_g_gan_fake_1 = self.criterion_gan(score_g_real_1 - torch.mean(score_g_fake_1), False)
            loss_g_1 += self.criterion_gan_weight * (loss_g_gan_real_1 + loss_g_gan_fake_1) / 2

            # g_gan_loss 2
            loss_g_gan_real_2 = self.criterion_gan(score_g_fake_2 - torch.mean(score_g_real_2), True)
            loss_g_gan_fake_2 = self.criterion_gan(score_g_real_2 - torch.mean(score_g_fake_2), False)
            loss_g_2 += self.criterion_gan_weight * (loss_g_gan_real_2 + loss_g_gan_fake_2) / 2

            loss_g_total = loss_g_1 + loss_g_2
            loss_g_total.backward()
            self.optim_G.step()

            train_G_loss1 += loss_g_1.item()
            train_G_loss2 += loss_g_2.item()

        return [train_G_loss1, train_G_loss2], [train_D_loss1, train_D_loss2]

    def test(self):
        self.model['netG'].eval()
        self.model['netD'].eval()

        with torch.no_grad():
            self.img_fuse1, self.img_fuse2 = self.model['netG'](self.img_cat)

        if self.is_train:
            loss_g_total = 0.
            loss_g_1 = self.criterion_lambda * self.criterion_pix(self.img_fuse1, self.img_pf) + \
                       self.criterion_lambda * self.criterion_pix(self.img_fuse1, self.img_ir)
            loss_g_total += loss_g_1
            loss_g_2 = self.criterion_lambda * self.criterion_pix(self.img_fuse2, self.img_pf) + \
                       self.criterion_lambda * self.criterion_epsilon * \
                       self.criterion_feat(gradient(self.img_fuse2), gradient(self.img_vis))
            loss_g_total += loss_g_2

            return loss_g_total.item()

    def save(self, epoch, is_best):
        filename = os.path.join(self.checkpoint_dir, 'checkpoint.pth')
        print('[Saving checkpoint to %s ...]' % filename)
        state = {
            'epoch': epoch,
            'state_dict_G': self.model['netG'].state_dict(),
            'state_dict_D': self.model['netD'].state_dict(),
            'optimizer_G': self.optim_G.state_dict(),
            'optimizer_D': self.optim_D.state_dict(),
            'best_prec': self.best_prec,
            'results': self.results
        }
        torch.save(state, filename)
        if is_best:
            torch.save(state, os.path.join(self.checkpoint_dir, 'best_checkpoint.pth'))
        print(['=> Done.'])

    def load(self):
        checkpoint_path = os.path.join(self.checkpoint_dir, 'checkpoint.pth')
        print('[Loading checkpoint from %s...]' % checkpoint_path)
        checkpoint = torch.load(checkpoint_path)
        self.model['netG'].load_state_dict(checkpoint['state_dict_G'])
        self.model['netD'].load_state_dict(checkpoint['state_dict_D'])
        start_epoch = checkpoint['epoch'] + 1  # Because the last state had been saved
        self.optim_G.load_state_dict(checkpoint['optimizer_G'])
        self.optim_D.load_state_dict(checkpoint['optimizer_D'])
        self.best_prec = checkpoint['best_prec']
        self.results = checkpoint['results']
        print('=> Done.')

        return start_epoch

    def get_current_visual(self):
        out_dict = OrderedDict()
        out_dict['img_vis'] = self.img_vis.data[0].float().cpu()
        out_dict['img_ir'] = self.img_ir.data[0].float().cpu()
        if self.is_train:
            out_dict['img_pf'] = self.img_pf.data[0].float().cpu()
        out_dict['img_fuse1'] = self.img_fuse1.data[0].float().cpu()
        out_dict['img_fuse2'] = self.img_fuse2.data[0].float().cpu()
        out_dict['img_fuse'] = 0.5 * (self.img_fuse1.data[0].float().cpu()
                                      + self.img_fuse2.data[0].float().cpu())

        return out_dict

    def get_current_visual_list(self):
        vis_list = []
        vis_list.append(util.quantize(self.img_vis.data[0].float().cpu()))
        vis_list.append(util.quantize(self.img_ir.data[0].float().cpu()))
        vis_list.append(util.quantize(self.img_fuse1.data[0].float().cpu()))
        vis_list.append(util.quantize(self.img_fuse2.data[0].float().cpu()))
        vis_list.append(util.quantize(0.5 * (self.img_fuse1.data[0].float().cpu()
                                             + self.img_fuse2.data[0].float().cpu())))
        if self.opt['is_train']:
            vis_list.append(util.quantize(self.img_pf.data[0].float().cpu()))
        return vis_list