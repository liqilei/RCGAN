import os
import math
import subprocess
from datetime import datetime
import numpy as np
from PIL import Image
from skimage.measure import compare_ssim

import cv2
import torch, gc
import torch.nn.functional as F
from torchvision.utils import make_grid
from torchvision.transforms import Compose, ToTensor, ToPILImage, CenterCrop

from data.common import rgb2ycbcr


####################
# miscellaneous
####################
def gradient(input):
    filters = torch.Tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]]).view((1, 1, 3, 3)).cuda()
    return F.conv2d(input, filters, padding=1)


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [%s]' % new_name)
        os.rename(path, new_name)
    os.makedirs(path)


####################
# image convert
####################

def tensor2img_np(tensor, out_type=np.uint8, min_max=(0, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    tensor = tensor.float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received tensor with dimension: %d' % n_dim)
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def save_img_np(img_np, img_path, mode='RGB'):
    if img_np.ndim == 2:
        mode = 'L'
    img_pil = Image.fromarray(img_np, mode=mode)
    img_pil.save(img_path)


def display_transform():
    return Compose([
        ToPILImage(),
        CenterCrop(400),
        ToTensor()
    ])


def quantize(img):
    # return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)
    return img.mul(127.5).add(127.5).clamp(0, 255).round()


####################
# metric
####################
"""
Ours

SSIM is a little different
"""


def calc_psnr(img1, img2):
    img1 = rgb2ycbcr(img1).astype(np.float32)
    img2 = rgb2ycbcr(img2).astype(np.float32)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def calc_ssim(img1, img2):
    img1 = rgb2ycbcr(img1).astype(np.float32)
    img2 = rgb2ycbcr(img2).astype(np.float32)
    return compare_ssim(img1, img2, gaussian_weights=True, sigma=1.5,
                        use_sample_covariance=False)  # same as Wang et al. 2004
    # return compare_ssim(img1, img2)


####################
# gpu debug
####################

def gpu_dbg_tensor_alloc():
    for obj in gc.get_objects():
        if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
            print(type(obj), obj.size())


def gpu_dbg_mem_use():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    nvsmi = 'C:/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe' if os.name == 'nt' else 'nvidia-smi'
    result = subprocess.check_output(
        [
            nvsmi, '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], universal_newlines=True)
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    print(gpu_memory_map)
    # return gpu_memory_map


if __name__ == '__main__':
    # read images
    import numpy as np
    from scipy import misc

    img1 = misc.imread('/home/ruby/RCAN-master/RCAN_TestCode/HR/Set5/x2/woman_HR_x2.png')
    img2 = misc.imread('/home/ruby/RCAN-master/RCAN_TestCode/SR/BI/RCAN/Set5/x2/woman_RCAN_x2.png')
    crop_size = 2
    crop_img1 = img1[crop_size:-crop_size, crop_size:-crop_size, :]
    crop_img2 = img2[crop_size:-crop_size, crop_size:-crop_size, :]
    psnr = calc_psnr(crop_img1, crop_img2)
    ssim = calc_ssim(crop_img1, crop_img2)
    print("PSNR is %f" % psnr)
    print("SSIM is %f" % ssim)
