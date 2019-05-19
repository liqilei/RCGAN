import argparse, os
import scipy.misc as misc
import numpy as np
from tqdm import tqdm
import torchvision.utils as thutil
import torch
import options.options as option
from utils import util
from models.RCGANModel import RCGANModel

from data import create_dataloader
from data import create_dataset


def main():
    parser = argparse.ArgumentParser(description='Test RCGAN model')
    parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
    opt = option.parse(parser.parse_args().opt)
    opt = option.dict_to_nonedict(opt)

    # create test dataloader
    dataset_opt = opt['datasets']['test']
    if dataset_opt is None:
        raise ValueError("test dataset_opt is None!")
    test_set = create_dataset(dataset_opt)
    test_loader = create_dataloader(test_set, dataset_opt)

    if test_loader is None:
        raise ValueError("The test data does not exist")

    solver = RCGANModel(opt)
    solver.model_pth = opt['model_path']
    solver.results_dir = os.path.join(opt['model_path'], 'results')
    solver.cmp_dir = os.path.join(opt['model_path'], 'cmp')

    # load model
    model_pth = os.path.join(solver.model_pth, 'RCGAN_model.pth')
    if model_pth is None:
        raise ValueError("model_pth' is required.")
    print('[Loading model from %s...]' % model_pth)
    model_dict = torch.load(model_pth)
    solver.model['netG'].load_state_dict(model_dict['state_dict_G'])

    print('=> Done.')
    print('[Start Testing]')

    test_bar = tqdm(test_loader)
    fused_list = []
    path_list = []

    if not os.path.exists(solver.cmp_dir):
        os.makedirs(solver.cmp_dir)

    for iter, batch in enumerate(test_bar):
        solver.feed_data(batch)
        solver.test()
        visuals_list = solver.get_current_visual_list()  # fetch current iteration results as cpu tensor
        visuals = solver.get_current_visual()  # fetch current iteration results as cpu tensor
        images = torch.stack(visuals_list)
        saveimg = thutil.make_grid(images, nrow=3, padding=5)
        saveimg_nd = saveimg.byte().permute(1, 2, 0).numpy()
        img_name = os.path.splitext(os.path.basename(batch['VIS_path'][0]))[0]
        misc.imsave(os.path.join(solver.cmp_dir, 'comp_%s.bmp' % (img_name)), saveimg_nd)
        fused_img = visuals['img_fuse']
        fused_img = np.transpose(util.quantize(fused_img).numpy(), (1, 2, 0)).astype(np.uint8).squeeze()
        fused_list.append(fused_img)
        path_list.append(img_name)

    save_img_path = solver.results_dir
    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)

    for img, img_name in zip(fused_list, path_list):
        misc.imsave(os.path.join(solver.results_dir, img_name + '.bmp'), img)

    test_bar.close()


if __name__ == '__main__':
    main()
