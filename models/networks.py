import torch
import torch.nn as nn
from torch.nn import init
import functools


####################
# initialize
####################

def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    print('initializing [%s] ...' % classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        print('initializing [%s] ...' % classname)
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        m.weight.data *= scale
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print('initializing [%s] ...' % classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='truncated_normal', scale=1, std=1e-3):
    # scale for 'kaiming', std for 'normal'.
    print('initialization method [%s]' % init_type)
    if init_type == 'truncated_normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


####################
# define network
####################
def create_model(opt):
    netG = define_G(opt['networks']['G'])
    netD = define_D(opt['networks']['D'])
    return {"netG": netG, "netD": netD}


# Generator
def define_G(opt):
    import models.modules.fusCoGAN_arch as fusCoGAN_arch
    netG = fusCoGAN_arch.CoGAN_G(2, 1)

    if torch.cuda.is_available():
        netG = nn.DataParallel(netG).cuda()
    return netG


# Discriminator
def define_D(opt):
    import models.modules.fusCoGAN_arch as fusCoGAN_arch
    netD = fusCoGAN_arch.CoGAN_D(1, 1)

    if torch.cuda.is_available():
        netD = nn.DataParallel(netD).cuda()
    return netD
