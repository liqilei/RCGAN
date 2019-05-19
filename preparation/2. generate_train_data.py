import os
from os.path import join
import numpy as np
import cv2
import glob


def main():
    ori_dir = '../Demo_dataset/train/ori_img'
    sub_dir = '../Demo_dataset/train/sub_img'
    stride = 14
    sub_img_size = 120

    Train_ir_list = glob.glob(join(join(ori_dir, 'ir'), '*.*'))
    Train_vi_list = glob.glob(join(join(ori_dir, 'vi'), '*.*'))
    Train_pf_list = glob.glob(join(join(ori_dir, 'pf'), '*.*'))

    if not os.path.exists(join(join(sub_dir, 'ir'))):
        os.makedirs(join(join(sub_dir, 'ir')))
    if not os.path.exists(join(join(sub_dir, 'vi'))):
        os.makedirs(join(join(sub_dir, 'vi')))
    if not os.path.exists(join(join(sub_dir, 'pf'))):
        os.makedirs(join(join(sub_dir, 'pf')))

    assert len(Train_ir_list) == len(Train_vi_list), \
        'len(Train_ir_list) != len(Train_vis_list)'
    count = 0
    for idx in range(len(Train_vi_list)):
        ir_img = cv2.imread(Train_ir_list[idx], cv2.IMREAD_GRAYSCALE)
        vi_img = cv2.imread(Train_vi_list[idx], cv2.IMREAD_GRAYSCALE)
        pf_img = cv2.imread(Train_pf_list[idx], cv2.IMREAD_GRAYSCALE)

        assert ir_img.shape == vi_img.shape, 'shape is not same, ir is {}, vis is {}' \
            .format(ir_img.shape, vi_img.shape)
        h, w = ir_img.shape
        h_mark = np.arange(0, h - sub_img_size + 1, stride)
        w_mark = np.arange(0, w - sub_img_size + 1, stride)

        flag = 0
        for h_step in h_mark:
            for w_step in w_mark:
                sub_ir = ir_img[h_step:h_step + sub_img_size, w_step:w_step + sub_img_size]
                sub_vi = vi_img[h_step:h_step + sub_img_size, w_step:w_step + sub_img_size]
                sub_pf = pf_img[h_step:h_step + sub_img_size, w_step:w_step + sub_img_size]
                cv2.imwrite(join(join(sub_dir, 'ir'), 'sub_ir_' + str(idx) + '_' + str(flag) + '.bmp'), sub_ir)
                cv2.imwrite(join(join(sub_dir, 'vi'), 'sub_vi_' + str(idx) + '_' + str(flag) + '.bmp'), sub_vi)
                cv2.imwrite(join(join(sub_dir, 'pf'), 'sub_pf_' + str(idx) + '_' + str(flag) + '.bmp'), sub_pf)
                flag += 1
                count += 1
        print(flag)
    print('count is {}'.format(count))


if __name__ == '__main__':
    main()
