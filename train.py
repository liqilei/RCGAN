import argparse
import os
import random

import pandas as pd
import scipy.misc as misc
import torch
import torchvision.utils as thutil
from tqdm import tqdm

import options.options as option
from data import create_dataloader
from data import create_dataset
from models.RCGANModel import RCGANModel
from utils import util


def main():
    parser = argparse.ArgumentParser(description='Train Super Resolution Models')
    parser.add_argument('-opt', type=str, required=True, help='Path to options JSON file.')
    opt = option.parse(parser.parse_args().opt)

    if opt['train']['resume'] is False:
        util.mkdir_and_rename(opt['path']['exp_root'])  # rename old experiments if exists
        util.mkdirs((path for key, path in opt['path'].items() if not key == 'exp_root' and \
                     not key == 'pretrain_G' and not key == 'pretrain_D'))
        option.save(opt)
        opt = option.dict_to_nonedict(opt)  # Convert to NoneDict, which return None for missing key.
    else:
        opt = option.dict_to_nonedict(opt)
        if opt['train']['resume_path'] is None:
            raise ValueError("The 'resume_path' does not declarate")

    NUM_EPOCH = int(opt['train']['num_epochs'])

    # random seed
    seed = opt['train']['manual_seed']
    if seed is None:
        seed = random.randint(1, 10000)
    print("Random Seed: ", seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # create train and val dataloader
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_loader = create_dataloader(train_set, dataset_opt)
            print('Number of train images in [%s]: %d' % (dataset_opt['name'], len(train_set)))
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            print('Number of val images in [%s]: %d' % (dataset_opt['name'], len(val_set)))
        elif phase == 'test':
            pass
        else:
            raise NotImplementedError("Phase [%s] is not recognized." % phase)

    if train_loader is None:
        raise ValueError("The training data does not exist")

    solver = RCGANModel(opt)

    solver.summary(train_set[0]['CAT'].size(), train_set[0]['IR'].size())
    solver.net_init()
    print('[Start Training]')

    start_epoch = 1
    if opt['train']['resume']:
        start_epoch = solver.load()

    for epoch in range(start_epoch, NUM_EPOCH + 1):
        # Initialization
        train_loss_g1 = 0.0
        train_loss_g2 = 0.0
        train_loss_d1 = 0.0
        train_loss_d2 = 0.0

        train_bar = tqdm(train_loader)
        # Train model
        for iter, batch in enumerate(train_bar):
            solver.feed_data(batch)
            loss_g_total, loss_d_total = solver.train_step()
            cur_batch_size = batch['CAT'].size(0)
            train_loss_g1 += loss_g_total[0] * cur_batch_size
            train_loss_g2 += loss_g_total[1] * cur_batch_size
            train_loss_d1 += loss_d_total[0] * cur_batch_size
            train_loss_d2 += loss_d_total[1] * cur_batch_size
            train_bar.set_description(desc='[%d/%d] G-Loss: %.4f D-Loss: %.4f' % (
                epoch, NUM_EPOCH, loss_g_total[0] + loss_g_total[1], loss_d_total[0] + loss_d_total[1]))

        solver.results['train_G_loss1'].append(train_loss_g1 / len(train_set))
        solver.results['train_G_loss2'].append(train_loss_g2 / len(train_set))
        solver.results['train_D_loss1'].append(train_loss_d1 / len(train_set))
        solver.results['train_D_loss2'].append(train_loss_d2 / len(train_set))
        print('Train G-Loss: %.4f' % ((train_loss_g1 + train_loss_g2) / len(train_set)))
        print('Train D-Loss: %.4f' % ((train_loss_d1 + train_loss_d2) / len(train_set)))

        train_bar.close()

        if epoch % solver.val_step == 0 and epoch != 0:
            print('[Validating...]')
            vis_index = 1
            val_loss = 0.0
            for iter, batch in enumerate(val_loader):
                solver.feed_data(batch)
                loss_total = solver.test()
                batch_size = batch['VIS'].size(0)
                vis_list = solver.get_current_visual_list()
                images = torch.stack(vis_list)
                saveimg = thutil.make_grid(images, nrow=3, padding=5)
                saveimg_nd = saveimg.byte().permute(1, 2, 0).numpy()
                misc.imsave(os.path.join(solver.vis_dir, 'epoch_%d_%d.png' % (epoch, vis_index)), saveimg_nd)
                vis_index += 1
                val_loss += loss_total * batch_size

            solver.results['val_G_loss'].append(val_loss / len(val_set))
            print('Valid Loss: %.4f' % (val_loss / len(val_set)))
            # statistics
            is_best = False
            if solver.best_prec > solver.results['val_G_loss'][-1]:
                solver.best_prec = solver.results['val_G_loss'][-1]
                is_best = True

            solver.save(epoch, is_best)

    data_frame = pd.DataFrame(
        data={'train_G_loss1': solver.results['train_G_loss1'],
              'train_G_loss2': solver.results['train_G_loss2'],
              'train_D_loss1': solver.results['train_D_loss1'],
              'train_D_loss2': solver.results['train_D_loss2'],
              'val_G_loss': solver.results['val_G_loss']
              },
        index=range(1, NUM_EPOCH + 1)
    )

    data_frame.to_csv(os.path.join(solver.results_dir, 'train_results.csv'),
                      index_label='Epoch')


if __name__ == '__main__':
    main()
