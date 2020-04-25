%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Pre-fuse visible and infrared images for training or validation
% This code comes from Li S , Kang X , Hu J . Image Fusion With Guided Filtering[J]. 
%IEEE Transactions on Image Processing, 2013, 22(7):2864-2875.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc,clear,close all;
input_path = '../../Demo_dataset/train/ori_img';
IR_path = fullfile(input_path, 'ir');
VI_path = fullfile(input_path, 'vi');
PF_path = fullfile(input_path, 'pf');

if ~exist(PF_path, 'dir')
    mkdir(PF_path)
end

Track_path = './SourceImage/track';
IR_list = dir(fullfile(IR_path, '*.bmp'));
VI_list = dir(fullfile(VI_path, '*.bmp'));

for ii = 1 : length(IR_list)
    fprintf('pre-fusing %d-th image; \n', ii)
    ir_img_path = fullfile(IR_path,IR_list(ii).name);
    ir_img = imread(ir_img_path);
    if size(ir_img, 3) == 3
        ir_img = rgb2gray(ir_img);
    end
    vi_img_path = fullfile(VI_path,VI_list(ii).name);
    vi_img = imread(vi_img_path);
    if size(vi_img, 3) == 3
        vi_img = rgb2gray(vi_img);
    end
    I = [];
    I(:,:,1) = ir_img;
    I(:,:,2) = vi_img;
    [F, ~] = GFF(I);
    temp = split(VI_list(ii).name,'.');
    save_name = temp{1};
    imwrite(F,fullfile(PF_path,[save_name, '.bmp']))
end